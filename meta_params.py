import numpy as np
from harmony_tools import utils as h_tools

f_spread = 0.5 * (2 ** np.linspace(-0.5, 0.5, 10))
dur_spread = 20 * 60 * 2 ** np.linspace(0, 1, 10)
cycles_spread = np.arange(6, 16)
chords_spread = np.arange(8, 18)
