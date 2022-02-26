from piece import build
import sys, json
# from line_profiler import LineProfiler
# lprofiler = LineProfiler()
# lp_wrapper = lprofiler(build)
# @profile



# f = 0.6
# dur_tot = 24 * 60
# noc = 14
# num_of_modes = 16
#
# f = float(sys.argv[1])
# dur_tot = float(sys.argv[2])
# noc = int(float(sys.argv[3]))
# num_of_modes = int(float(sys.argv[4]))
# print(f, dur_tot, noc, num_of_modes)

meta_params = json.load(open('JSON/meta_params.JSON', 'rb'))
print(meta_params[81])
f, dur_tot, noc, num_of_modes = meta_params[81]
noc = int(noc)
num_of_modes = int(num_of_modes)

# build(True, False, True, dur_tot=dur_tot, f=f, num_of_modes=num_of_modes, noc=noc)
# lp_wrapper(save_pickle=True)
