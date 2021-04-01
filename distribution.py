import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
def iterate(x):
    return (1.21 * x) % 1


# seed = np.random.rand()
#
# samples = [seed]
# for i in range(10000):
#     samples.append(iterate(samples[-1]))
#
#
#
# A = np.random.normal(size=10000)
# A_cumulative = np.cumsum(A)
#
# B = np.random.normal(size=10000)
# B_cumulative = np.cumsum(B)
# C = np.arctan2(B_cumulative, A_cumulative)
# print(C)


# to generate rhythms, for a given (fixed) gamut, you need a distribution of
# chord sizes, a distribution of eahc element, nCVI, temporal density.

#basically, for an
def weights_from_samples(sample, bins):
    weight = np.histogram(sample, bins=bins, range=(0, 1), density=True)[0]
    weight = weight / np.sum(weight)
    return weight

A = np.random.uniform(size=29)
C = weights_from_samples(A, 5)
D = weights_from_samples(A, 6)
print(C)



class 
