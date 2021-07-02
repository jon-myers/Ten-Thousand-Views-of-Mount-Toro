from piece import build
from line_profiler import LineProfiler
# lprofiler = LineProfiler()
# lp_wrapper = lprofiler(build)
# @profile
build(save_pickle=True)
# lp_wrapper(save_pickle=True)
