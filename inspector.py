import json
from fractions import Fraction
from harmony_tools import utils as ht
from harmony_tools import plot as hp
import numpy as np
modes = json.load(open('modes.json', 'r'))

from rhythm_tools import rhythmic_sequence_maker as rsm
from rhythm_tools import nCVI

a = rsm(10, 8)
offset = np.random.rand()

def event_times_maker(events, nCVI, dur_tot, offset=None):
    if offset == None or offset >= 1 or offset < 0:
        offset = np.random.rand()
    durs = rsm(events, nCVI)
    starts = np.cumsum(durs)[:-1]

    starts = np.insert(starts, 0, 0)
    starts = starts + offset
    starts = np.where(starts >= 1, starts - 1, starts)
    starts = np.sort(starts) * dur_tot
    return starts

def get_internal_nCVI(starts):
    internal_durs = [starts[i+1] - starts[i] for i in range(len(starts) - 1)]
    return nCVI(internal_durs)

def offset_durs(durs, offset):
    starts = np.cumsum(durs[:-1])
    starts = np.insert(starts, 0, 0)
    starts = starts + offset
    starts = np.where(starts >= 1, starts - 1, starts)
    starts = np.sort(starts)
    return starts

def event_times_fitter(A, events, nCVI, dur_tot):
    durs = rsm(events, nCVI)
    offsets = np.linspace(0, 1, 1000)
    nCVIs = []
    for offset in offsets:
        B = offset_durs(durs, offset) * dur_tot
        both = np.sort(np.concatenate((A, B)))
        nCVI_ = get_internal_nCVI(both)
        nCVIs.append(nCVI_)
    nCVIs = np.array(nCVIs)
    best_index = np.argmin(np.abs(nCVIs - nCVI))
    offset = offsets[best_index]
    best_fit = nCVIs[best_index]
    print(best_fit)
    B = offset_durs(durs, offset) * dur_tot
    return B


A = event_times_maker(10, 20, 1)

B = event_times_fitter(A, 4, 20, 1)
print(get_internal_nCVI(A))
print(get_internal_nCVI(B))

# B = event_times_maker(7, 10, 1)
# starts = np.sort(np.concatenate((A, B)))
# print(starts)
# internal_durs = [starts[i+1] - starts[i] for i in range(len(starts) - 1)]
# print(nCVI(internal_durs))
