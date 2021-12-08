from piece import build
# from line_profiler import LineProfiler
# lprofiler = LineProfiler()
# lp_wrapper = lprofiler(build)
# @profile
dur_tot = 24 * 60
f = 0.6
num_of_modes = 16
noc = 14

build(use_pickles=False, dur_tot=dur_tot, f=f, num_of_modes=num_of_modes, noc=noc)
# lp_wrapper(save_pickle=True)
