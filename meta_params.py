import numpy as np
from harmony_tools import utils as h_tools
import json


f_bins = np.linspace(-0.5, 0.5, 11)
dur_bins = np.linspace(0, 1, 11)

# f_spread = 0.5 * (2 ** np.linspace(-0.5, 0.5, 11))
# dur_spread = 20 * 60 * 2 ** np.linspace(0, 1, 11)
cycles_spread = np.arange(6, 16)
chords_spread = np.arange(12, 22)

f_idxs = h_tools.dc_alg(10, 10000)
dur_idxs = h_tools.dc_alg(10, 10000)
cycles_idxs = h_tools.dc_alg(10, 10000)
chords_idxs = h_tools.dc_alg(10, 10000)

f = 0.5 * (2 ** np.random.uniform(f_bins[f_idxs], f_bins[f_idxs+1]))
dur = 20 * 60 * (2 ** np.random.uniform(dur_bins[dur_idxs], dur_bins[dur_idxs+1]))
# f = f_spread[f_idxs]
# dur = dur_spread[dur_idxs]
cycles = cycles_spread[cycles_idxs]
chords = chords_spread[chords_idxs]

meta_params = np.array((f, dur, cycles, chords)).T


json.dump(meta_params, open('JSON/meta_params.JSON', 'w'), cls=h_tools.NpEncoder)
# print(len(test))
