import time
import logging
import warnings
import mxnet as mx
import mxnet.ndarray as nd
import mxnet.metric as metric
from mxnet.model import _update_params, _update_params_on_kvstore, BatchEndParam
import pdb

def add_batched_train_args(argument_group):
    """
    argument_group : argparse.ArgumentParser.add_argument_group
    return a argument_group added with args required by batched_train
    """
    argument_group.add_argument('--update-step', type=int, default=1,
                                help='steps to sync update fc parameters')

def batch_update(model, merged_grad_arrays):
    """Updates parameters according to the installed optimizer and the gradients computed
    in the previous forward-backward batch.

    See Also
    ----------
    :meth:`BaseModule.update`.
    """
    assert model.binded and model.params_initialized and model.optimizer_initialized

    model._params_dirty = True
    if model._update_on_kvstore:
        _update_params_on_kvstore(model._exec_group.param_arrays,
                                  merged_grad_arrays,
                                  model._kvstore,
                                  model._exec_group.param_names)
    else:
        _update_params(model._exec_group.param_arrays,
                       merged_grad_arrays,
                       updater=model._updater,
                       num_device=len(model._context),
                       kvstore=model._kvstore,
                       params_names=model._exec_group.param_names)

def init_merged_grad_array(model):
    """
    init merged_grad_array used to store internal grad before update
    """
    merged_grad_arrays = []
    for i in range(len(model._exec_group.grad_arrays)):
        single_grad = []
        for j in range(len(model._exec_group.grad_arrays[i])):
            single_grad.append(nd.zeros(model._exec_group.grad_arrays[i][j].shape,
                                        model._exec_group.grad_arrays[i][j].context))
        merged_grad_arrays.append(single_grad)
    return merged_grad_arrays

def update_merged_grad_array(model, merged_grad_arrays, update_step, cur_step):
    """
    update param by merged_grad_array
    """
    if cur_step % update_step == 0:
        for i in range(len(model._exec_group.grad_arrays)):
            for j in range(len(model._exec_group.grad_arrays[i])):
                merged_grad_arrays[i][j] = model._exec_group.grad_arrays[i][j].copy()
    else:
        for i in range(len(model._exec_group.grad_arrays)):
            for j in range(len(model._exec_group.grad_arrays[i])):
                merged_grad_arrays[i][j] += model._exec_group.grad_arrays[i][j]

def batched_fit(model,
                train_data,
                eval_data=None,
                eval_metric='acc',
                epoch_end_callback=None,
                batch_end_callback=None,
                kvstore='local',
                optimizer='sgd',
                optimizer_params=None,
                eval_end_callback=None,
                eval_batch_end_callback=None,
                initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
                arg_params=None,
                aux_params=None,
                allow_missing=False,
                force_rebind=False,
                force_init=False,
                begin_epoch=0,
                num_epoch=None,
                validation_metric=None,
                monitor=None,
                update_step=1):
    """
    Trains the module parameters.

    Checkout `Module Tutorial <http://mxnet.io/tutorials/basic/module.html>`_ to see
    a end-to-end use-case.

    Parameters
    ----------
    model : Module
        class Module
    update_step : int
        how many steps to sync update parameters
    train_data : DataIter
        Train DataIter.
    eval_data : DataIter
        If not ``None``, will be used as validation set and the performance
        after each epoch will be evaluated.
    eval_metric : str or EvalMetric
        Defaults to 'accuracy'. The performance measure used to display during training.
        Other possible predefined metrics are:
        'ce' (CrossEntropy), 'f1', 'mae', 'mse', 'rmse', 'top_k_accuracy'.
    epoch_end_callback : function or list of functions
        Each callback will be called with the current `epoch`, `symbol`, `arg_params`
        and `aux_params`.
    batch_end_callback : function or list of function
        Each callback will be called with a `BatchEndParam`.
    kvstore : str or KVStore
        Defaults to 'local'.
    optimizer : str or Optimizer
        Defaults to 'sgd'.
    optimizer_params : dict
        Defaults to ``(('learning_rate', 0.01),)``. The parameters for
        the optimizer constructor.
        The default value is not a dict, just to avoid pylint warning on dangerous
        default values.
    eval_end_callback : function or list of function
        These will be called at the end of each full evaluation, with the metrics over
        the entire evaluation set.
    eval_batch_end_callback : function or list of function
        These will be called at the end of each mini-batch during evaluation.
    initializer : Initializer
        The initializer is called to initialize the module parameters when they are
        not already initialized.
    arg_params : dict
        Defaults to ``None``, if not ``None``, should be existing parameters from a trained
        model or loaded from a checkpoint (previously saved model). In this case,
        the value here will be used to initialize the module parameters, unless they
        are already initialized by the user via a call to `init_params` or `fit`.
        `arg_params` has a higher priority than `initializer`.
    aux_params : dict
        Defaults to ``None``. Similar to `arg_params`, except for auxiliary states.
    allow_missing : bool
        Defaults to ``False``. Indicates whether to allow missing parameters when `arg_params`
        and `aux_params` are not ``None``. If this is ``True``, then the missing parameters
        will be initialized via the `initializer`.
    force_rebind : bool
        Defaults to ``False``. Whether to force rebinding the executors if already bound.
    force_init : bool
        Defaults to ``False``. Indicates whether to force initialization even if the
        parameters are already initialized.
    begin_epoch : int
        Defaults to 0. Indicates the starting epoch. Usually, if resumed from a
        checkpoint saved at a previous training phase at epoch N, then this value should be
        N+1.
    num_epoch : int
        Number of epochs for training.
    validation_metric : str or EvalMetric
        Defaults to 'accuracy'. The performance measure used to display during validation.
        Other possible predefined metrics are:
        'ce' (CrossEntropy), 'f1', 'mae', 'mse', 'rmse', 'top_k_accuracy'.
    monitor:
        time monitor
    """
    assert num_epoch is not None, 'please specify number of epochs'

    model.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label,
               for_training=True, force_rebind=force_rebind)
    if monitor is not None:
        model.install_monitor(monitor)
    model.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                      allow_missing=allow_missing, force_init=force_init)

    optimizer_params['rescale_grad'] = 1.0 / (model._exec_group.batch_size * update_step)
    optimizer_params['learning_rate'] = optimizer_params['learning_rate'] * update_step

    model.init_optimizer(kvstore=kvstore, optimizer=optimizer,
                         optimizer_params=optimizer_params)

    if validation_metric is None:
        validation_metric = eval_metric
    if not isinstance(eval_metric, metric.EvalMetric):
        eval_metric = metric.create(eval_metric)

    merged_grad_arrays = init_merged_grad_array(model)
    ################################################################################
    # training loop
    ################################################################################
    for epoch in range(begin_epoch, num_epoch):
        tic = time.time()
        eval_metric.reset()
        nbatch = 0
        data_iter = iter(train_data)
        end_of_batch = False
        next_data_batch = next(data_iter)
        while not end_of_batch:
            data_batch = next_data_batch
            if monitor is not None:
                monitor.tic()
            model.forward_backward(data_batch)

            if update_step > 1:
                update_merged_grad_array(model, merged_grad_arrays, update_step, nbatch)
                if (nbatch + 1) % update_step == 0:
                    batch_update(model, merged_grad_arrays)
            else:
                model.update()

            try:
                # pre fetch next batch
                next_data_batch = next(data_iter)
                model.prepare(next_data_batch)
            except StopIteration:
                end_of_batch = True

            model.update_metric(eval_metric, data_batch.label)

            if monitor is not None:
                monitor.toc_print()

            if batch_end_callback is not None:
                batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                 eval_metric=eval_metric,
                                                 locals=locals())
                for callback in as_list(batch_end_callback):
                    callback(batch_end_params)
            nbatch += 1

        # one epoch of training is finished
        for name, val in eval_metric.get_name_value():
            model.logger.info('Epoch[%d] Train-%s=%f', epoch, name, val)
        toc = time.time()
        model.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))

        # sync aux params across devices
        arg_params, aux_params = model.get_params()
        model.set_params(arg_params, aux_params)

        if epoch_end_callback is not None:
            for callback in as_list(epoch_end_callback):
                callback(epoch, model.symbol, arg_params, aux_params)

        #----------------------------------------
        # evaluation on validation set
        if eval_data:
            res = model.score(eval_data, validation_metric,
                              score_end_callback=eval_end_callback,
                              batch_end_callback=eval_batch_end_callback, epoch=epoch)
            for name, val in res:
                model.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)

        # end of 1 epoch, reset the data-iter for another epoch
        train_data.reset()


def as_list(obj):
    """A utility function that treat the argument as a list.

    Parameters
    ----------
    obj : object

    Returns
    -------
    If `obj` is a list, return it. Otherwise, return `[obj]` as a single-element list.
    """
    if isinstance(obj, list):
        return obj
    else:
        return [obj]
