"""
Benchmark the scoring performance on various CNNs
"""
from common import find_mxnet
from common.util import get_gpus
import mxnet as mx
from importlib import import_module
import logging
import time
import numpy as np
import pdb
import argparse

logging.basicConfig(level=logging.DEBUG)

def channel_symbol(num_group, cudnn_off):
    ## define alexnet
    data = mx.symbol.Variable(name="data")
    if cudnn_off == 0:
        tmp = True
    else:
        tmp = False
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), pad=(1, 1), num_filter=32, cudnn_off=tmp, num_group=num_group, name="conv1_1")
    return conv1

def score(dev, batch_size, num_batches, num_group, size, cudnn_off):
    # get mod
    sym = channel_symbol(num_group, cudnn_off)
    image_shape = (3,size,size)
    data_shape = [('data', (batch_size,)+image_shape)]
    mod = mx.mod.Module(symbol=sym, context=dev)
    mod.bind(for_training     = False,
             inputs_need_grad = False,
             data_shapes      = data_shape)
    a=np.array([[1,2,3],[4,5,6],[7,8,9]])
    b=mx.ndarray.array(a)
    c=b.broadcast_to((32,3,3,3))
    c1=mx.ndarray.ones((32))
    d={'conv1_1_bias':c1, 'conv1_1_weight':c}
    mod.init_params(arg_params=d, aux_params=None)

    # get data
    data = [mx.random.uniform(1.0, 1.0, shape=shape, ctx=dev) for _, shape in mod.data_shapes]
    batch = mx.io.DataBatch(data, []) # empty label

    # run
    tic = time.time()
    for i in range(num_batches):
        mod.forward(batch, is_train=False)
        for output in mod.get_outputs():
            output.wait_to_read()
            pdb.set_trace()
            logging.info("ouput max:c %f", output.asnumpy().max())
            logging.info("ouput min:c %f", output.asnumpy().min())

    # return num images per second
    return num_batches*batch_size/(time.time() - tic)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train cifar10", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_group', type=int, default=1, help='initial learning rate')
    parser.add_argument('--size', type=int, default=112, help='initial learning rate')
    parser.add_argument('--cudnn_off', type=int, default=1, help='initial learning rate')
    args = parser.parse_args()
    batch_sizes = [1]
    for b in batch_sizes:
        speed = score(dev=mx.gpu(0), batch_size=b, num_batches=1, num_group = args.num_group, size = args.size, cudnn_off = args.cudnn_off)
        logging.info('batch size %2d, image/sec: %f', b, speed)
