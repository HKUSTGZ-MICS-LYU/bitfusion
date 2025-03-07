
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
    fc,
    get_bench_numbers,
    get_cifar10_qnn
)
from bitfusion.src.simulator.simulator import Simulator

warnings.filterwarnings('ignore')

config_file = 'bitfusion.ini'
# config_file = 'bf_e_conf.ini'
bf_sim = Simulator(config_file, verbose=False)
print(bf_sim)
print("Area:", bf_sim.get_area())

batch_size = 16
ATYPE = FQDtype.FXP8
WTYPE = FQDtype.FXP2
# Create a Graph
g = Graph("MLP", "MNIST", log_level=logging.INFO)
with g.as_default():
    with g.name_scope('input'):
        x = get_tensor(
            shape=(batch_size, 256), name='x', dtype = ATYPE,
            trainable=False)
    with g.name_scope('fc1'):
        fc1 = fc(x, 256, f_dtype=ATYPE, w_dtype=WTYPE)
    with g.name_scope('fc2'):
        fc2 = fc(fc1, 256, f_dtype=ATYPE, w_dtype=WTYPE)
    with g.name_scope('fc3'):
        fc3 = fc(fc2, 256, f_dtype=ATYPE, w_dtype=WTYPE)
    with g.name_scope('fc4'):
        fc4 = fc(fc3, 256, f_dtype=ATYPE, w_dtype=WTYPE)
    with g.name_scope('fc5'):
        fc5 = fc(fc4, 10, f_dtype=ATYPE, w_dtype=WTYPE)

def sim_results(bf_sim, batch_size, g):
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

import time
start = time.time()
res = sim_results(bf_sim, batch_size, g)
print("-"*50)
print(res)
print("Simulation Time:", time.time()-start)