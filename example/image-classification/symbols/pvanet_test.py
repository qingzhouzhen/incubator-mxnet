"""
Benchmark the scoring performance on various CNNs
"""


import logging
import time
import mxnet as mx
from pvanet import *

logging.basicConfig(level=logging.DEBUG)

def test_inception(dev, batch_size, num_batches):
    data = mx.sym.Variable('data')
    image_shape = (128, 132, 80)
    # get mod
    sym = inception(data=data, middle_filter=[64, [48, 128], [24, 48, 48], 128], num_filter=256, kernel=(3, 3),stride=(2, 2), proj=True, name='conv4_1', suffix='')
    data_shape = [('data', (batch_size,)+image_shape)]
    mod = mx.mod.Module(symbol=sym, context=dev)
    mod.bind(for_training     = False,
             inputs_need_grad = False,
             data_shapes      = data_shape)
    mod.init_params(initializer=mx.init.Xavier(magnitude=2.))

    # get data
    data = [mx.random.uniform(-1.0, 1.0, shape=shape, ctx=dev) for _, shape in mod.data_shapes]
    batch = mx.io.DataBatch(data, []) # empty label

    # run
    dry_run = 5                 # use 5 iterations to warm up
    for i in range(num_batches):
        tic = time.time()
        mod.forward(batch, is_train=False)
        for output in mod.get_outputs():
            output.wait_to_read()
            print mod.data_shapes
            print mod.output_shapes

    # return num images per second
    return num_batches*batch_size/(time.time() - tic)

def test_res_crelu(dev, batch_size, num_batches):
    data = mx.sym.Variable('data')
    image_shape = (3, 264, 160)
    # get mod
    sym = res_crelu(data=data, middle_filter=[24, 24], num_filter=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1), bsr=True, proj=True, name='test', suffix='')
    data_shape = [('data', (batch_size,)+image_shape)]
    #sym, data_shape = get_symbol(batch_size)
    mod = mx.mod.Module(symbol=sym, context=dev)
    mod.bind(for_training     = False,
             inputs_need_grad = False,
             data_shapes      = data_shape)
    mod.init_params(initializer=mx.init.Xavier(magnitude=2.))

    # get data
    data = [mx.random.uniform(-1.0, 1.0, shape=shape, ctx=dev) for _, shape in mod.data_shapes]
    batch = mx.io.DataBatch(data, []) # empty label

    # run
    dry_run = 5                 # use 5 iterations to warm up
    for i in range(num_batches):
        tic = time.time()
        mod.forward(batch, is_train=False)
        for output in mod.get_outputs():
            output.wait_to_read()
            print mod.data_shapes
            print mod.output_shapes

    # return num images per second
    return num_batches*batch_size/(time.time() - tic)
def test_all(dev, batch_size, num_batches):

    image_shape = (3, 1056, 640)
    # get mod
    sym = get_symbol(num_class=1000)

    slice = mx.sym.slice_axis(data=sym, axis=1, begin=0, end=256, name='slice')

    data_shape = [('data', (batch_size,)+image_shape)]
    mod = mx.mod.Module(symbol=sym, context=dev)
    mod.bind(for_training     = False,
             inputs_need_grad = False,
             data_shapes      = data_shape)
    mod.init_params(initializer=mx.init.Xavier(magnitude=2.))

    # get data
    data = [mx.random.uniform(-1.0, 1.0, shape=shape, ctx=dev) for _, shape in mod.data_shapes]
    batch = mx.io.DataBatch(data, []) # empty label

    # run
    dry_run = 5                 # use 5 iterations to warm up
    for i in range(num_batches):
        tic = time.time()
        mod.forward(batch, is_train=False)
        for output in mod.get_outputs():
            output.wait_to_read()
            print mod.data_shapes
            print mod.output_shapes

    # return num images per second
    return num_batches*batch_size/(time.time() - tic)

if __name__ == '__main__':

    batch_sizes = [32]

    for b in batch_sizes:
        #speed = test_res_crelu(dev=mx.gpu(0), batch_size=b, num_batches=1)
        #speed = test_inception(dev=mx.gpu(0), batch_size=b, num_batches=1)
        speed = test_all(dev=mx.gpu(1), batch_size=b, num_batches=1)
        logging.info('batch size %2d, image/sec: %f', b, speed)
