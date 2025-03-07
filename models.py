from bitfusion.src.benchmarks.benchmarks import (
    fc, conv,
    get_bench_numbers
)

from dnnweaver2.graph import Graph, get_default_graph
from dnnweaver2.tensorOps.cnn import (
    conv2D, maxPool, flatten, matmul, 
    addBias, batch_norm, reorg, concat, 
    leakyReLU, add, globalAvgPool
)
from dnnweaver2 import get_tensor
from dnnweaver2.scalar.dtypes import FQDtype, FixedPoint, Dtype

import logging

BATCH_SIZE = 16
class Int(Dtype):
    def __init__(self, bits):
        self.bits = bits
        self.int_bits = bits
        self.frac_bits = 0
        self.op_str = 'INT{}'.format(self.bits)

def int2dtype(q):
    return Int(q)

def mlp_mnist(wq: list, aq: list):
    g = Graph("MLP", "MNIST", log_level=logging.INFO)

    ATYPE = list(map(int2dtype, aq))
    WTYPE = list(map(int2dtype, wq))

    with g.as_default():
        with g.name_scope('input'):
            x = get_tensor(
                shape=(BATCH_SIZE, 256), name='x', dtype = ATYPE[0],
                trainable=False)
        with g.name_scope('fc1'):
            fc1 = fc(x, 64, f_dtype=ATYPE[0], w_dtype=WTYPE[0])
        with g.name_scope('fc2'):
            fc2 = fc(fc1, 64, f_dtype=ATYPE[1], w_dtype=WTYPE[1])
        with g.name_scope('fc3'):
            fc3 = fc(fc2, 64, f_dtype=ATYPE[2], w_dtype=WTYPE[2])
        with g.name_scope('fc5'):
            fc4 = fc(fc3, 10, f_dtype=ATYPE[3], w_dtype=WTYPE[3])
    return g

def lenet_mnist(wq: list, aq: list):
    g = Graph("LeNet5", "MNIST", log_level=logging.INFO)

    ATYPE = list(map(int2dtype, aq))
    WTYPE = list(map(int2dtype, wq))

    with g.as_default():
        with g.name_scope('inputs'):
            i = get_tensor(shape=(BATCH_SIZE, 28, 28,1), 
                           name='data', dtype=ATYPE[0], trainable=False)

        with g.name_scope('conv0'):
            conv0 = conv(i, filters=6, kernel_size=5, pad='SAME', 
                    c_dtype=ATYPE[0], w_dtype=WTYPE[0])
            
        with g.name_scope('pool0'):
            pool0 = maxPool(conv0, pooling_kernel=(1,2,2,1),
                             stride=(1,2,2,1), pad='VALID')

        with g.name_scope('conv1'):
            conv1 = conv(pool0, filters=16, kernel_size=5, pad='SAME',
                    c_dtype=ATYPE[1], w_dtype=WTYPE[1])
        with g.name_scope('pool1'):
            pool1 = maxPool(conv1, pooling_kernel=(1,2,2,1), 
                            stride=(1,2,2,1), pad='VALID')

        with g.name_scope('flatten1'):
            flatten1 = flatten(pool1)

        with g.name_scope('fc1'):
            fc1 = fc(flatten1, output_channels=64,
                     f_dtype=ATYPE[2], w_dtype=WTYPE[2])

        with g.name_scope('fc2'):
            fc2 = fc(fc1, output_channels=32,
                    f_dtype=ATYPE[3], w_dtype=WTYPE[3])
        
        with g.name_scope('fc3'):
            fc3 = fc(fc2, output_channels=10,
                    f_dtype=ATYPE[4], w_dtype=WTYPE[4])
    return g

def vgg_cifar10(wq: list, aq: list):
    g = Graph("VGG", "CIFAR-10", log_level=logging.INFO)

    ATYPE = list(map(int2dtype, aq))
    WTYPE = list(map(int2dtype, wq))
    with g.as_default():
        with g.name_scope('inputs'):
            i = get_tensor(shape=(BATCH_SIZE, 32, 32, 3), 
                           name='data', dtype=ATYPE[0], trainable=False)

        with g.name_scope('conv0'):
            conv0 = conv(i, filters=64, kernel_size=3, pad='SAME', 
                    c_dtype=ATYPE[0], w_dtype=WTYPE[0])
        with g.name_scope('pool0'):
            pool0 = maxPool(conv0, pooling_kernel=(1,2,2,1),
                             stride=(1,2,2,1), pad='VALID')

        with g.name_scope('conv1'):
            conv1 = conv(pool0, filters=128, kernel_size=3, pad='SAME',
                    c_dtype=ATYPE[1], w_dtype=WTYPE[1])
        with g.name_scope('pool1'):
            pool1 = maxPool(conv1, pooling_kernel=(1,2,2,1), 
                            stride=(1,2,2,1), pad='VALID')

        with g.name_scope('conv2'):
            conv2 = conv(pool1, filters=256, kernel_size=3, pad='SAME',
                    c_dtype=ATYPE[2], w_dtype=WTYPE[2])
        with g.name_scope('pool2'):
            pool2 = maxPool(conv2, pooling_kernel=(1,2,2,1), 
                            stride=(1,2,2,1), pad='VALID')

        with g.name_scope('conv3'):
            conv3 = conv(pool2, filters=256, kernel_size=3, pad='SAME',
                    c_dtype=ATYPE[3], w_dtype=WTYPE[3])
        with g.name_scope('pool3'):
            pool3 = maxPool(conv3, pooling_kernel=(1,2,2,1), 
                            stride=(1,2,2,1), pad='VALID')

        with g.name_scope('flatten1'):
            flatten1 = flatten(pool3)

        with g.name_scope('fc1'):
            fc1 = fc(flatten1, output_channels=4096,
                     f_dtype=ATYPE[4], w_dtype=WTYPE[4])

        with g.name_scope('fc2'):
            fc2 = fc(fc1, output_channels=4096,
                    f_dtype=ATYPE[5], w_dtype=WTYPE[5])

        with g.name_scope('fc3'):
            fc3 = fc(fc2, output_channels=10,
                    f_dtype=ATYPE[6], w_dtype=WTYPE[6])

    return g

# TODO: Add more models here
def squeezeNet_cifar100(wq: list, aq: list):
    pass

def resnet18_cifar100(wq: list, aq: list):
    pass