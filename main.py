from piece import build
# from line_profiler import LineProfiler
# lprofiler = LineProfiler()
# lp_wrapper = lprofiler(build)
# @profile
dur_tot = 20 * 60
build(use_pickles=False, dur_tot=dur_tot)
# lp_wrapper(save_pickle=True)
