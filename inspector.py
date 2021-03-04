import json
from fractions import Fraction
from harmony_tools import utils as ht
from harmony_tools import plot as hp
import numpy as np
modes = json.load(open('modes.json', 'r'))
# mode = modes[4]
# mode = [i/mode[0] for i in mode]
# primes = np.array((3, 5, 7, 11, 13), dtype=float)
# hsvs = ht.gen_ratios_to_hsv(mode, primes)
# print(hsvs)
# hp.make_4d_plot(hsvs[:, :-1], 'test_plots/pre')
# hsvs = ht.fix_collection(hsvs)[0]
# hp.make_4d_plot(hsvs[:, :-1], 'test_plots/post')
funds = np.array([mode[0] for mode in modes])[:18]
quotients = [funds[i+1] / funds[i] for i in range(len(funds) - 1)]
quotients = np.array(quotients)
quotients = np.where(quotients<1, quotients* 2, quotients)
print(np.round(funds, 3))
print(np.round(quotients, 3))
# print(funds)
# print(quotients)
# lim = 1.05
# print(np.nonzero(funds<lim))
