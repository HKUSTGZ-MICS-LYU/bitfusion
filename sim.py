
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

warnings.filterwarnings('ignore')
def get_default_simulator():
    bf_sim = Simulator('bitfusion.ini', verbose=False)
    return bf_sim

def sim_results(g, bf_sim = None, batch_size = 16):
    if bf_sim is None:
        bf_sim = get_default_simulator()

    stats = get_bench_numbers(g, bf_sim, batch_size=batch_size)

    total_cycles = 0
    total_stall = 0
    total_energy = 0
    total_reads = 0
    total_writes = 0

    for layer in stats:
        cycles = stats[layer].total_cycles
        reads = stats[layer].reads['dram']
        writes = stats[layer].writes['dram']
        stalls = stats[layer].mem_stall_cycles
        # print("Layer -", layer)
        # print("Cycles:", cycles)
        cc_energy, mem_energy = stats[layer].get_energy(bf_sim.get_energy_cost())
        # print("Energy:", cc_energy, mem_energy)
        total_cycles += cycles
        total_stall += stalls
        total_energy += cc_energy + mem_energy
        total_reads += reads
        total_writes += writes
    return {
        'Cycles': total_cycles,
        'Stall': total_stall,
        'Energy': total_energy,
        'Reads': total_reads,
        'Writes': total_writes
    }

if __name__ == "__main__":
    import sys
    import json
    import time
    from models import mlp_mnist, lenet_mnist, vgg_cifar10, matmul_test
    
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
    elif model_name == 'matmul':
        g = matmul_test(sim_config['wq'], sim_config['aq'], 
                        sim_config['n'], sim_config['m'], sim_config['k'])
    else:
        raise ValueError(f"Model {model_name} not supported.")
    res = sim_results(g)
    print("-"*50)
    for k, v in res.items():
        print(f"{k}: {v}")
    print("Simulation Time:", time.time()-start)