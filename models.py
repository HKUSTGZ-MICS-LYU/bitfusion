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

def matmul_test(wq: list, aq: list, N=16, K=256, M=256):
    g = Graph("MatMul", "MNIST", log_level=logging.INFO)

    ATYPE = list(map(int2dtype, aq))
    WTYPE = list(map(int2dtype, wq))

    with g.as_default():
        with g.name_scope('input'):
            x = get_tensor(
                shape=(N, K), name='x', dtype = ATYPE[0],
                trainable=False)
        with g.name_scope('matmul'):
            matmul1 = fc(x, M, f_dtype=ATYPE[0], w_dtype=WTYPE[0])
    return g

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

    assert (len(ATYPE) == 4) and (len(WTYPE) == 4), "Invalid quantization scheme"

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

    assert (len(ATYPE) == 7) and (len(WTYPE) == 7), "Invalid quantization scheme"

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

def cmsiscnn_cifar10(wq: list, aq: list):
    g = Graph("CMSIS-CNN", "CIFAR-10", log_level=logging.INFO)
    
    ATYPE = list(map(int2dtype, aq))
    WTYPE = list(map(int2dtype, wq))
    
    assert (len(ATYPE) == 4) and (len(WTYPE) == 4), "Invalid quantization scheme"
    
    with g.as_default():
        with g.name_scope('inputs'):
            i = get_tensor(shape=(BATCH_SIZE, 32, 32, 3), 
                           name='data', dtype=ATYPE[0], trainable=False)
        
        # First conv block
        with g.name_scope('conv0'):
            conv0 = conv(i, filters=32, kernel_size=5, pad='SAME', 
                    c_dtype=ATYPE[0], w_dtype=WTYPE[0])
        with g.name_scope('pool0'):
            pool0 = maxPool(conv0, pooling_kernel=(1,2,2,1),
                            stride=(1,2,2,1), pad='VALID')
        
        # Second conv block
        with g.name_scope('conv1'):
            conv1 = conv(pool0, filters=32, kernel_size=5, pad='SAME',
                    c_dtype=ATYPE[1], w_dtype=WTYPE[1])
        with g.name_scope('pool1'):
            pool1 = maxPool(conv1, pooling_kernel=(1,2,2,1),
                            stride=(1,2,2,1), pad='VALID')
        
        # Third conv block
        with g.name_scope('conv2'):
            conv2 = conv(pool1, filters=64, kernel_size=5, pad='SAME',
                    c_dtype=ATYPE[2], w_dtype=WTYPE[2])
        with g.name_scope('pool2'):
            pool2 = maxPool(conv2, pooling_kernel=(1,2,2,1),
                            stride=(1,2,2,1), pad='VALID')
        
        # Flatten and fully connected layer
        with g.name_scope('flatten'):
            flatten1 = flatten(pool2)
        
        with g.name_scope('fc'):
            fc1 = fc(flatten1, output_channels=10,
                    f_dtype=ATYPE[3], w_dtype=WTYPE[3])
    
    return g

def resnet8_cifar100(wq: list, aq: list):
    g = Graph("ResNet-8", "CIFAR-100", log_level=logging.INFO)
    
    ATYPE = list(map(int2dtype, aq))
    WTYPE = list(map(int2dtype, wq))
    
    assert (len(ATYPE) == 13) and (len(WTYPE) == 13), "Invalid quantization scheme"
    
    with g.as_default():
        with g.name_scope('inputs'):
            i = get_tensor(shape=(BATCH_SIZE, 32, 32, 3), 
                           name='data', dtype=ATYPE[0], trainable=False)
        
        # Initial convolution
        with g.name_scope('conv1'):
            conv1 = conv(i, filters=64, kernel_size=3, pad='SAME',
                    c_dtype=ATYPE[0], w_dtype=WTYPE[0])
            
        # Layer 1 (1 BasicBlock)
        with g.name_scope('layer1_block1_conv1'):
            l1b1_conv1 = conv(conv1, filters=64, kernel_size=3, pad='SAME',
                          c_dtype=ATYPE[1], w_dtype=WTYPE[1])
        with g.name_scope('layer1_block1_conv2'):
            l1b1_conv2 = conv(l1b1_conv1, filters=64, kernel_size=3, pad='SAME',
                          c_dtype=ATYPE[2], w_dtype=WTYPE[2])
        
        # Identity shortcut (no parameter needed)
        l1b1_out = add((l1b1_conv2, conv1), dtype=ATYPE[2])
        
        # Layer 2 (1 BasicBlock with downsampling)
        with g.name_scope('layer2_block1_conv1'):
            l2b1_conv1 = conv(l1b1_out, filters=128, kernel_size=3, 
                          stride=(1,2,2,1), pad='SAME',
                          c_dtype=ATYPE[3], w_dtype=WTYPE[3])
        with g.name_scope('layer2_block1_conv2'):
            l2b1_conv2 = conv(l2b1_conv1, filters=128, kernel_size=3, pad='SAME',
                          c_dtype=ATYPE[4], w_dtype=WTYPE[4])
        
        # Shortcut with 1x1 conv due to dimension change
        with g.name_scope('layer2_shortcut'):
            l2b1_shortcut = conv(l1b1_out, filters=128, kernel_size=1, 
                             stride=(1,2,2,1), pad='SAME',
                             c_dtype=ATYPE[5], w_dtype=WTYPE[5])
        
        l2b1_out = add((l2b1_conv2, l2b1_shortcut), dtype=ATYPE[5])
        
        # Layer 3 (1 BasicBlock with downsampling)
        with g.name_scope('layer3_block1_conv1'):
            l3b1_conv1 = conv(l2b1_out, filters=256, kernel_size=3, 
                          stride=(1,2,2,1), pad='SAME',
                          c_dtype=ATYPE[6], w_dtype=WTYPE[6])
        with g.name_scope('layer3_block1_conv2'):
            l3b1_conv2 = conv(l3b1_conv1, filters=256, kernel_size=3, pad='SAME',
                          c_dtype=ATYPE[7], w_dtype=WTYPE[7])
        
        # Shortcut with 1x1 conv due to dimension change
        with g.name_scope('layer3_shortcut'):
            l3b1_shortcut = conv(l2b1_out, filters=256, kernel_size=1, 
                             stride=(1,2,2,1), pad='SAME',
                             c_dtype=ATYPE[8], w_dtype=WTYPE[8])
        
        l3b1_out = add((l3b1_conv2, l3b1_shortcut), dtype=ATYPE[8])
        
        # Layer 4 (1 BasicBlock with downsampling)
        with g.name_scope('layer4_block1_conv1'):
            l4b1_conv1 = conv(l3b1_out, filters=512, kernel_size=3, 
                          stride=(1,2,2,1), pad='SAME',
                          c_dtype=ATYPE[9], w_dtype=WTYPE[9])
        with g.name_scope('layer4_block1_conv2'):
            l4b1_conv2 = conv(l4b1_conv1, filters=512, kernel_size=3, pad='SAME',
                          c_dtype=ATYPE[10], w_dtype=WTYPE[10])
        
        # Shortcut with 1x1 conv due to dimension change
        with g.name_scope('layer4_shortcut'):
            l4b1_shortcut = conv(l3b1_out, filters=512, kernel_size=1, 
                             stride=(1,2,2,1), pad='SAME',
                             c_dtype=ATYPE[11], w_dtype=WTYPE[11])
        
        l4b1_out = add((l4b1_conv2, l4b1_shortcut), dtype=ATYPE[11])
        
        # Global average pooling
        with g.name_scope('avg_pool'):
            avg_pool = globalAvgPool(l4b1_out)
            
        # FC layer
        with g.name_scope('fc'):
            fc_out = fc(avg_pool, output_channels=100,
                      f_dtype=ATYPE[12], w_dtype=WTYPE[12])
        
    return g

def resnet18_cifar100(wq: list, aq: list):
    g = Graph("ResNet-18", "CIFAR-100", log_level=logging.INFO)
    
    ATYPE = list(map(int2dtype, aq))
    WTYPE = list(map(int2dtype, wq))
    
    assert (len(ATYPE) == 21) and (len(WTYPE) == 21), "Invalid quantization scheme"
    
    # Track the quantization index
    q_idx = 0
    
    with g.as_default():
        with g.name_scope('inputs'):
            i = get_tensor(shape=(BATCH_SIZE, 32, 32, 3), 
                           name='data', dtype=ATYPE[0], trainable=False)
        
        # Initial convolution
        with g.name_scope('conv1'):
            conv1 = conv(i, filters=64, kernel_size=3, pad='SAME',
                    c_dtype=ATYPE[q_idx], w_dtype=WTYPE[q_idx])
            q_idx += 1
        
        # Track the current tensor and number of channels
        current = conv1
        in_channels = 64
        
        # Define the block configurations: [(num_blocks, out_channels), ...]
        blocks_config = [(2, 64), (2, 128), (2, 256), (2, 512)]
        
        # Create all ResNet blocks
        for layer_idx, (num_blocks, out_channels) in enumerate(blocks_config):
            for block_idx in range(num_blocks):
                # First block in layer 2, 3, 4 has stride 2 for downsampling
                stride = (1,2,2,1) if layer_idx > 0 and block_idx == 0 else (1,1,1,1)
                
                with g.name_scope(f'layer{layer_idx+1}_block{block_idx+1}'):
                    # Residual path
                    with g.name_scope(f'conv1'):
                        res_conv1 = conv(current, filters=out_channels, kernel_size=3, 
                                        stride=stride, pad='SAME',
                                        c_dtype=ATYPE[q_idx], w_dtype=WTYPE[q_idx])
                        q_idx += 1
                    
                    with g.name_scope(f'conv2'):
                        res_conv2 = conv(res_conv1, filters=out_channels, kernel_size=3, 
                                        pad='SAME',
                                        c_dtype=ATYPE[q_idx], w_dtype=WTYPE[q_idx])
                        q_idx += 1
                    
                    # Shortcut path
                    if stride != (1,1,1,1) or in_channels != out_channels:
                        # Need 1x1 conv for shortcut due to dimension change
                        with g.name_scope(f'shortcut'):
                            shortcut = conv(current, filters=out_channels, kernel_size=1, 
                                           stride=stride, pad='SAME',
                                           c_dtype=ATYPE[q_idx], w_dtype=WTYPE[q_idx])
                            q_idx += 1
                    else:
                        shortcut = current
                    
                    # Add residual and shortcut
                    current = add((res_conv2, shortcut), dtype=ATYPE[q_idx-1])
                
                # Update in_channels for next block
                in_channels = out_channels
        
        # Global average pooling
        with g.name_scope('avg_pool'):
            avg_pool = globalAvgPool(current)
            
        # FC layer
        with g.name_scope('fc'):
            fc_out = fc(avg_pool, output_channels=100,
                      f_dtype=ATYPE[q_idx], w_dtype=WTYPE[q_idx])
        
    return g

def squeeze_cifar(wq: list, aq: list):
    pass