import numpy as np
import matplotlib.pyplot as plt
from rhythm_tools import rhythmic_sequence_maker as rsm





def sequence_from_sample(samples, bounds):
    A, B, C = samples
    seq = []
    for bound in bounds:
        a_bool = np.all((A >= bound[0], A < bound[1]), axis=0)
        a_count = np.count_nonzero(a_bool)
        b_bool = np.all((B >= bound[0], B < bound[1]), axis=0)
        b_count = np.count_nonzero(b_bool)
        c_bool = np.all((C >= bound[0], C < bound[1]), axis=0)
        c_count = np.count_nonzero(c_bool)
        maxs = np.argmax((a_count, b_count, c_count), axis=0)
        seq.append(maxs)
    return seq
    
starts = rsm(4, 8, start_times=True)
starts = np.append(starts, 1)
bounds = [(starts[i], starts[i+1]) for i in range(len(starts)-1)]


sample_a = np.random.uniform(size=150)
sample_a = np.append(sample_a, np.random.uniform(0, 0.01, size=100))
sample_b = np.random.uniform(size=150)
sample_c = np.random.uniform(size=150)
samples = (sample_a, sample_b, sample_c)

# seq = sequence_from_sample(samples, bounds)
# print(seq)

bounds = [(0, 0.5), (0.5, 1)]
seq = sequence_from_sample(samples, bounds)
print(seq)

bounds = [(0, 1/3), (1/3, 2/3), (2/3, 1)]
seq = sequence_from_sample(samples, bounds)
print(seq)

bounds = [(0, 0.25), (0.25, 0.50), (0.5, 0.75), (0.75, 1)]
seq = sequence_from_sample(samples, bounds)
print(seq)

bounds = [(0, 0.2), (0.2, 0.40), (0.4, 0.6), (0.6, 0.8), (0.8, 1)]
seq = sequence_from_sample(samples, bounds)
print(seq)
