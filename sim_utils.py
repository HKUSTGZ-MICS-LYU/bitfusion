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

warnings.filterwarnings('ignore')
def get_default_simulator():
    bf_path = os.path.dirname(os.path.abspath(__file__))
    bf_sim = Simulator( bf_path + '/bitfusion.ini', verbose=False)
    return bf_sim

def sim_results(g: Graph, bf_sim = None, batch_size = 16):
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