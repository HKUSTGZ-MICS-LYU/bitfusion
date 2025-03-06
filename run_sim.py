import pandas
import configparser
import os
import numpy as np

import matplotlib
from pandas import DataFrame

import dnnweaver2
import bitfusion.src.benchmarks.benchmarks as benchmarks
from bitfusion.src.simulator.stats import Stats
from bitfusion.src.simulator.simulator import Simulator
from bitfusion.src.sweep.sweep import SimulatorSweep, check_pandas_or_run
from bitfusion.src.utils.utils import *
from bitfusion.src.optimizer.optimizer import optimize_for_order, get_stats_fast
from bitfusion.graph_plot.barchart import BarChart
from bitfusion.src.simulator.stats import Stats

import warnings
warnings.filterwarnings('ignore')

## use a batch size of 16
batch_size = 128

results_dir = './results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

fig_dir = './fig'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

## establish systolic arrray
data = {'Max Precision (bits)': [8, 8],
        'Min Precision (bits)': [2, 2],
        'M': [32, 4],
        'N': [16, 4],
        'Area (um^2)': [1, 1],
        'Dynamic Power (nW)': [1, 1],
        'Frequency': [1, 1],
        'Leakage Power (nW)': [1, 1]
        }

df = DataFrame(data, columns=list(data.keys()))
export_csv = df.to_csv (r'./results/systolic_array_synth.csv', index = None, header=True)
# print (df)

config_file = 'bf_e_conf.ini'
# config_file = 'conf.ini'

# Create simulator object
verbose = False
bf_e_sim = Simulator(config_file, verbose)
bf_e_energy_costs = bf_e_sim.get_energy_cost()
print(bf_e_sim)

energy_tuple = bf_e_energy_costs
print('')
print('*'*50)
print(energy_tuple)

# bench_list = benchmarks.FracTrain_benchlist
# bench_list = benchmarks.L2A_benchlist
bench_list = benchmarks.benchlist

sim_sweep_columns = ['N', 'M',
        'Max Precision (bits)', 'Min Precision (bits)',
        'Network', 'Layer',
        'Cycles', 'Memory wait cycles',
        'WBUF Read', 'WBUF Write',
        'OBUF Read', 'OBUF Write',
        'IBUF Read', 'IBUF Write',
        'DRAM Read', 'DRAM Write',
        'Bandwidth (bits/cycle)',
        'WBUF Size (bits)', 'OBUF Size (bits)', 'IBUF Size (bits)',
        'Batch size']

# bf_e_sim_sweep_csv = os.path.join(results_dir, 'bitfusion-eyeriss-sim-sweep.csv')
# if os.path.exists(bf_e_sim_sweep_csv):
#     bf_e_sim_sweep_df = pandas.read_csv(bf_e_sim_sweep_csv)
# else:
#     bf_e_sim_sweep_df = pandas.DataFrame(columns=sim_sweep_columns)
bf_e_sim_sweep_csv = os.path.join(results_dir, 'trial.csv')
if os.path.exists(bf_e_sim_sweep_csv):
    os.remove(bf_e_sim_sweep_csv)
bf_e_sim_sweep_df = pandas.DataFrame(columns=sim_sweep_columns)
print('Got BitFusion Eyeriss, Numbers')

bf_e_results = check_pandas_or_run(bf_e_sim, bf_e_sim_sweep_df, bf_e_sim_sweep_csv, list_bench=bench_list, batch_size=batch_size, config_file='./conf.ini')
bf_e_results = bf_e_results.groupby('Network',as_index=False).agg(np.sum)
export_csv = bf_e_results.to_csv (r'./results/trial_stat.csv', index = None, header=True)
area_stats = bf_e_sim.get_area()

def df_to_stats(df):
    stats = Stats()
    stats.total_cycles = float(df['Cycles'])
    stats.mem_stall_cycles = float(df['Memory wait cycles'])
    stats.reads['act'] = float(df['IBUF Read'])
    stats.reads['out'] = float(df['OBUF Read'])
    stats.reads['wgt'] = float(df['WBUF Read'])
    stats.reads['dram'] = float(df['DRAM Read'])
    stats.writes['act'] = float(df['IBUF Write'])
    stats.writes['out'] = float(df['OBUF Write'])
    stats.writes['wgt'] = float(df['WBUF Write'])
    stats.writes['dram'] = float(df['DRAM Write'])
    return stats

print('BitFusion-Eyeriss comparison')
eyeriss_area = 3.5*3.5*45*45/65./65.
print('Area budget = {}'.format(eyeriss_area))


print(area_stats)
if abs(sum(area_stats)-eyeriss_area)/eyeriss_area > 0.1:
    print('Warning: BitFusion Area is outside 10% of eyeriss')
print('total_area = {}, budget = {}'.format(sum(area_stats), eyeriss_area))
bf_e_area = sum(area_stats)

baseline_data = []
for bench in bench_list:
    lookup_dict = {'Benchmark': bench}

    # eyeriss_cycles = float(lookup_pandas_dataframe(eyeriss_data_bench, lookup_dict)['time(ms)'])
    # eyeriss_time = eyeriss_cycles / 500.e3 / 16
    # eyeriss_energy = get_eyeriss_energy(lookup_pandas_dataframe(eyeriss_data_bench, lookup_dict))
    # eyeriss_power = eyeriss_energy / eyeriss_time * 1.e-9

    # eyeriss_speedup = eyeriss_time / eyeriss_time
    # eyeriss_energy_efficiency = eyeriss_energy / eyeriss_energy

    # eyeriss_ppa = eyeriss_speedup / eyeriss_area / (eyeriss_speedup / eyeriss_area)
    # eyeriss_ppw = eyeriss_speedup / eyeriss_power / (eyeriss_speedup / eyeriss_power)

    bf_e_stats = df_to_stats(bf_e_results.loc[bf_e_results['Network'] == bench])
    bf_e_cycles = bf_e_stats.total_cycles * (batch_size / 16.)
    bf_e_time = bf_e_cycles / 500.e3 #/ 16
    cc_energy, mem_energy = bf_e_stats.get_energy(bf_e_sim.get_energy_cost())
    cc_energy = cc_energy * (batch_size / 16.)
    mem_energy = mem_energy * (batch_size / 16.)
    print('cc_energy: {}, mem_energy: {}'.format(cc_energy*1.e-9, mem_energy*1.e-9))
    bf_e_energy = cc_energy + mem_energy
    bf_e_power = bf_e_energy / bf_e_time * 1.e-9

    # bf_e_speedup = eyeriss_time / bf_e_time
    # bf_e_energy_efficiency = eyeriss_energy / bf_e_energy

    # bf_e_ppa = bf_e_speedup / bf_e_area / (eyeriss_speedup / eyeriss_area)
    # bf_e_ppw = bf_e_speedup / bf_e_power / (eyeriss_speedup / eyeriss_power)

    # baseline_data.append(['Performance', bench, bf_e_speedup])
    # baseline_data.append(['Energy reduction', bench, bf_e_energy_efficiency])
    # baseline_data.append(['Performance-per-Watt', bench, bf_e_ppw])
    # baseline_data.append(['Performance-per-Area', bench, bf_e_ppa])

    print('*'*50)
    print('Benchmark: {}'.format(bench))
    # print('Eyeriss time: {} ms'.format(eyeriss_time))
    print('BitFusion time: {} ms'.format(bf_e_time))
    # print('Eyeriss power: {} mWatt'.format(eyeriss_power*1.e3*16))
    print('BitFusion power: {} mWatt'.format(bf_e_power*1.e3*16))
    # print('BitFusion energy: {} mJ'.format(bf_e_time*bf_e_power*1.e3*16 / 1.e3))
    print('BitFusion energy: {} J'.format(bf_e_energy*1.e-9))
    print('*'*50)