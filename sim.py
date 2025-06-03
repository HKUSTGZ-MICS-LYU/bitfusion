import os
import logging
import warnings
import dnnweaver2
from dnnweaver2.graph import Graph, get_default_graph
from dnnweaver2.tensorOps.cnn import (
    conv2D, maxPool, flatten, matmul, 
    addBias, batch_norm, reorg, concat, 
    leakyReLU, add
)
from dnnweaver2 import get_tensor
from dnnweaver2.scalar.dtypes import FQDtype, FixedPoint

from bitfusion.src.simulator.stats import Stats
from bitfusion.src.benchmarks.benchmarks import (
    get_bench_numbers
)
from bitfusion.src.simulator.simulator import Simulator
from sim_utils import sim_results

if __name__ == "__main__":
    import sys
    import json
    import time
    from models import (
        mlp_mnist, lenet_mnist, vgg_cifar10, cmsiscnn_cifar10,
        resnet8, resnet18,
        matmul_test, conv2d_test, conv2d_2l_test
    )
    
    sim_config_file = sys.argv[1]

    with open(sim_config_file, 'r') as f:
        sim_config = json.load(f)

    # print(get_default_simulator())

    start = time.time()
    model_name = sim_config['model_name']
    if model_name == 'mlp':
        g = mlp_mnist(sim_config['wq'], sim_config['aq'])
    elif model_name == 'lenet':
        g = lenet_mnist(sim_config['wq'], sim_config['aq'])
    elif model_name == 'vgg':
        g = vgg_cifar10(sim_config['wq'], sim_config['aq'])
    elif model_name == 'cmsiscnn':
        g = cmsiscnn_cifar10(sim_config['wq'], sim_config['aq'])
    elif model_name == 'resnet8_cifar10':
        g = resnet8(sim_config['wq'], sim_config['aq'], 10)
    elif model_name == 'resnet8_cifar100':
        g = resnet8(sim_config['wq'], sim_config['aq'], 100)
    elif model_name == 'resnet18_cifar100':
        g = resnet18(sim_config['wq'], sim_config['aq'], 100)
    elif model_name == 'matmul':
        g = matmul_test(sim_config['wq'], sim_config['aq'], 
                        sim_config['n'], sim_config['m'], sim_config['k'])
    elif model_name == 'conv2d':
        g = conv2d_test(sim_config['wq'], sim_config['aq'], 
                   sim_config['h'], sim_config['w'], sim_config['c'],
                sim_config['k'], sim_config['ks'])
    else:
        raise ValueError(f"Model {model_name} not supported.")
    res = sim_results(g)
    print("-"*50)
    for k, v in res.items():
        print(f"{k}: {v}")
    print("Simulation Time:", time.time()-start)