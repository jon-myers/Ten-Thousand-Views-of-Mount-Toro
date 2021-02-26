from harmony_tools import utils as h_tools
import numpy as np
import json


def normed_inverse_geometric_series(mult, size):
    series = [1/(mult ** i) for i in range(size)]
    series = [i / sum(series) for i in series]
    return series

def make_bass_line(steps, primes, prime_weights, hs_weights):
    """
    hs_contour_weights: three item array with relative weight of going "up" in harmonic
    space, "down" in harmonic space, staying the same, or going 'diagonal'

    """
    contour = np.zeros((steps, len(primes)))

    hs_contour = np.random.choice(np.arange(4), size=steps, p=hs_weights)
    for i, move_type in enumerate(hs_contour):
        vec = np.zeros(len(primes))
        if move_type == 0:
            prime = np.random.choice(np.arange(len(primes)), p=prime_weights)
            vec[prime] = -1
        elif move_type == 1:
            prime = np.random.choice(np.arange(len(primes)), p=prime_weights)
            vec[prime] = 1
        elif move_type == 3:
            prime = np.random.choice(np.arange(len(primes)), size=2, p=prime_weights, replace=False)
            vec[prime[0]] = 1
            vec[prime[1]] = -1
        contour[i] = vec
    return contour







golden = (1 + 5 ** 0.5) / 2
primes = np.array((3, 5, 7, 11, 13, 17), dtype=float)
prime_weights = normed_inverse_geometric_series(golden, len(primes))
contour_weights = normed_inverse_geometric_series(golden, 4)
bass_line_motion = make_bass_line(20, primes, prime_weights, contour_weights)
bass_line = np.cumsum(bass_line_motion, axis=0)
rats = h_tools.hsv_to_gen_ratios(bass_line, primes)
json.dump(rats, open('bass_line.json', 'w'), cls=h_tools.NpEncoder)
