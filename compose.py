from harmony_tools import utils as h_tools
import numpy as np
import json, math
import numpy_indexed as npi
from fractions import Fraction


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


def hsv_to_frac(hsv, primes):
    pos_hsv = np.where(hsv >= 0, hsv, 0)
    neg_hsv = np.where(hsv <= 0, hsv, 0)
    if len(np.shape(pos_hsv)) == 2:
        numerator = np.prod(primes ** pos_hsv, axis=1)
        denominator = np.prod(primes ** (-1 * neg_hsv), axis=1)
        return [(numerator[i], denominator[i]) for i in range(len(numerator))]
    else:
        numerator = np.prod(primes ** pos_hsv)
        denominator = np.prod(primes ** (-1 * neg_hsv))
        return (numerator, denominator)

def harmonic_distance(frac):
    return np.log(frac[0] * frac[1])

def harmonic_distance_mixed(ratio1, ratio2):
    """Gets harmonic distance between two decimal ratios"""
    f1 = Fraction(ratio1).limit_denominator(10000)
    f2 = Fraction(ratio2).limit_denominator(10000)
    int1 = f1.numerator * f2.denominator
    int2 = f2.numerator * f1.denominator
    gcd = math.gcd(int1, int2)
    int1 /= gcd
    int2 /= gcd
    return harmonic_distance((int1, int2))

def get_possible_ratios(max_hd=3):
    primes = np.array((3, 5, 7, 11, 13), dtype=float)
    eye = np.eye(len(primes), dtype=int)
    p = np.zeros_like(primes)
    one_step = np.concatenate((eye, -1*eye))
    a = np.tile(one_step, 10).reshape((100, 5))
    c = np.broadcast_to(one_step, (10, 10, 5)).reshape((100, 5))
    two_step = a + c
    a = np.tile(one_step, 100).reshape((1000, 5))
    c = np.broadcast_to(two_step, (10, 100, 5)).reshape((1000, 5))
    three_step = a + c
    intervals = np.concatenate((one_step, two_step, three_step))
    intervals = npi.unique(intervals)
    fracs = hsv_to_frac(intervals, primes)
    hds = np.array([harmonic_distance(frac) for frac in fracs])
    hd_sorts = np.argsort(hds)
    hd_sorts = hd_sorts[hds[hd_sorts] <=5]
    sorted_hsvs = intervals[hd_sorts]
    sorted_fracs = [fracs[i] for i in hd_sorts]
    rats = h_tools.hsv_to_gen_ratios(sorted_hsvs, primes)
    monophonic_rats = np.sort(rats)
    return monophonic_rats


def mistuned_filter(possible, mistuned_int, p_int, origin, oct=False):
    mt_min = h_tools.cents_to_hz(-mistuned_int, p_int)
    mt_max = h_tools.cents_to_hz(mistuned_int, p_int)
    condition_1 = origin * possible <= mt_min
    condition_2 = origin * possible == p_int
    condition_3 = origin * possible >= mt_max
    if oct == True:
        filter = np.any((condition_1, condition_2, condition_3), axis=0)
    else:
        filter = np.any((condition_1, condition_3), axis=0)
    return filter

def build_mode(ratios, lower_third_lims=[7/6, 9/7], upper_third_lims=[7/6, 9/7],
               mistuned_5th=50, mistuned_8ve=66, last_mode=None, bass_motion=1,
               last_mode_weight=0.25, alpha=8):
    ratios = np.concatenate((ratios, 2 * ratios))
    mode = [bass_motion]
    for i in range(6):
        if i < 2:
            lower_lim = lower_third_lims[0] * mode[-1]
            upper_lim = lower_third_lims[1] * mode[-1]
        else:
            lower_lim = upper_third_lims[0] * mode[-1]
            upper_lim = upper_third_lims[1] * mode[-1]
        # while upper_lim >= 2:
        #     upper_lim /= 2
        # while lower_lim >= 2:
        #     lower_lim /= 2
        above = ratios >= lower_lim
        below = ratios <= upper_lim
        possible = ratios[np.all((above, below), axis=0)]
        if i > 0:
            p_fifth = 1.5 * mode[-2]
            filter = mistuned_filter(possible, mistuned_5th, p_fifth, mode[-1])
            possible = possible[filter]

        if i == 2:
            oct = 2 * mode[-3]
            filter = mistuned_filter(possible, mistuned_8ve, oct, mode[-1], True)
            possible = possible[filter]

        if i > 2:
            octs = [2 * mode[-3], 2 * mode[-4]]
            filter_A = mistuned_filter(possible, mistuned_8ve, octs[0], mode[-1], True)
            filter_B = mistuned_filter(possible, mistuned_8ve, octs[1], mode[-1], True)

            filter = np.all((filter_A, filter_B), axis=0)
            possible = possible[filter]

        choice_hds = []
        for choice in possible:
            aggregate_hd = []
            for current in mode:
                hd = harmonic_distance_mixed(current, choice)
                aggregate_hd.append(hd)
            if np.all(last_mode != None):
                for rat in last_mode:
                    hd = harmonic_distance_mixed(rat, choice) * last_mode_weight
                    aggregate_hd.append(hd)
            choice_hds.append(sum(aggregate_hd))
        alpha = 4
        weight = [1/(i**alpha) for i in choice_hds]
        weight = [i/sum(weight) for i in weight]
        mode.append(np.random.choice(possible, p=weight))
    return mode






golden = (1 + 5 ** 0.5) / 2
primes = np.array((3, 5, 7, 11, 13, 17), dtype=float)
prime_weights = normed_inverse_geometric_series(golden, len(primes))
contour_weights = normed_inverse_geometric_series(golden, 4)
bass_line_motion = make_bass_line(20, primes, prime_weights, contour_weights)
bass_line = np.cumsum(bass_line_motion, axis=0)
bl_ratios = h_tools.hsv_to_gen_ratios(bass_line, primes)
bl_ratios = [(4/3)**i for i in range(10)]
bl_ratios = (4/3) ** np.arange(10)
while np.any(bl_ratios >= 2):
    bl_ratios = np.where(bl_ratios >= 2, bl_ratios / 2, bl_ratios)
print(bl_ratios)
ratios=get_possible_ratios()
modes = [build_mode(ratios)]
for i in range(len(bl_ratios)):
    mode = build_mode(ratios * bl_ratios[i], bass_motion=bl_ratios[i],
                      last_mode=[j/bl_ratios[i] for j in modes[-1]])
    modes.append(mode)
# mode_2 = build_mode(ratios*bl_ratios[0], last_mode=[i / bl_ratios[0] for i in mode_1], bass_motion=bl_ratios[0])
for mode in modes:
    print(mode, '\n\n')
# print(mode_2)
json.dump(modes, open('modes.json', 'w'), cls=h_tools.NpEncoder)
