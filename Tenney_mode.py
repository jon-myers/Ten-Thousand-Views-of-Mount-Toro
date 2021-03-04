from mode import harmonic_distance_mixed
from harmony_tools import utils as h_tools
import numpy as np
import json

def get_aggregate_hd(mode, trial):
    hd = 0
    for note in mode:
        hd += harmonic_distance_mixed(note, trial)
    return hd

# def mistuned_filter(possible, mistuned_int, p_int, origin, oct=False):
#     mt_min = h_tools.cents_to_hz(-mistuned_int, p_int)
#     mt_max = h_tools.cents_to_hz(mistuned_int, p_int)
#     condition_1 = origin * possible <= mt_min
#     condition_2 = origin * possible == p_int
#     condition_3 = origin * possible >= mt_max
#     if oct == True:
#         filter = np.any((condition_1, condition_2, condition_3), axis=0)
#     else:
#         filter = np.any((condition_1, condition_3), axis=0)
#     return filter


def filter_mistuned_5ths(choices, origin, dev=50):
    choices = np.array(choices)
    low = h_tools.cents_to_hz(-dev, 1.5 * origin)
    high = h_tools.cents_to_hz(dev, 1.5 * origin)
    c1 = origin * choices < low
    c2 = np.round(choices, 5) == np.round(origin * 1.5, 5)
    c3 = origin * choices > high
    choices = choices[np.any((c1, c2, c3), axis=0)]
    return choices

def filter_8ves(choices, mode, dev=67):
    if len(mode) >= 4:
        origin = mode[-4]
        diff = np.array([abs(h_tools.hz_to_cents(origin, i)) for i in choices])
        choices = choices[diff > dev]

        origin = mode[-3]
        diff = np.array([abs(h_tools.hz_to_cents(origin, i)) for i in choices])
        choices = choices[diff > dev]
    else:
        origin = mode[-3]
        diff = np.array([abs(h_tools.hz_to_cents(origin, i)) for i in choices])

        choices = choices[diff > dev]

    return choices


def filter_by_limit(choices, lim, note):
    low = note * lim[0]
    high = note * lim[1]
    choices = np.array(choices)
    c1 = np.round(choices, 5) >= np.round(low, 5)
    c2 = np.round(choices, 5) <= np.round(high, 5)
    out_choices = choices[np.all((c1, c2), axis=0)]
    if len(out_choices) == 0: breakpoint()
    return out_choices

def get_weight(mode, choices, alpha, last_mode=None):
    lm_weighting = [1/3, 2/3]
    weight = [1 / (get_aggregate_hd(mode, i) ** alpha) for i in choices]
    weight = [i / sum(weight) for i in weight]
    weight = np.array(weight)
    if np.all(last_mode != None):
        last_mode_weight = get_weight(last_mode, choices, alpha)
        weight = lm_weighting[0] * last_mode_weight + lm_weighting[1] * weight
    return weight

def build_mode(degrees, lower_lim, upper_lim, alpha, last_mode=None):
    lm_weighting = [1/3, 2/3] # means harmonic distance weighting based on last
    # mode is worth 1/3, while hd weighting from current mode is worth 2/3

    mode = [1]
    # third
    _3rds = degrees['3rds']
    weight = get_weight(mode, _3rds, alpha, last_mode)
    third = np.random.choice(_3rds, p=weight)
    mode.append(third)


    # fifth
    _5ths = degrees['5ths']
    if np.round(third, 5) == np.round(6/5, 5):
        fifth = 3/2
    if np.round(third, 5) != np.round(5/4, 5):
        if 25/16 in _5ths:
            _5ths.remove(25/16)
    _5ths = filter_by_limit(_5ths, lower_lim, mode[-1])
    _5ths = filter_mistuned_5ths(_5ths, mode[0])
    weight = get_weight(mode, _5ths, alpha, last_mode)
    fifth = np.random.choice(_5ths, p=weight)
    mode.append(fifth)

    # seventh
    _7ths = degrees['7ths']
    _7ths = filter_by_limit(_7ths, upper_lim, mode[-1])
    _7ths = filter_mistuned_5ths(_7ths, mode[1])
    _7ths = filter_8ves(_7ths, mode)
    weight = get_weight(mode, _7ths, alpha, last_mode)
    seventh = np.random.choice(_7ths, p=weight)
    mode.append(seventh)

    # ninth
    _9ths = degrees['9ths']
    _9ths = filter_by_limit(_9ths, upper_lim, mode[-1]/2)
    # print(len(_9ths))
    _9ths = filter_mistuned_5ths(_9ths, mode[2]/2)
    _9ths = filter_8ves(_9ths, mode)
    weight = get_weight(mode, _9ths, alpha, last_mode)
    ninth = np.random.choice(_9ths, p=weight)
    mode.append(ninth)

    # eleventh
    _11ths = degrees['11ths']
    _11ths = filter_by_limit(_11ths, upper_lim, mode[-1])
    _11ths = filter_mistuned_5ths(_11ths, mode[3]/2)
    _11ths = filter_8ves(_11ths, mode)
    weight = get_weight(mode, _11ths, alpha, last_mode)
    eleventh = np.random.choice(_11ths, p=weight)
    mode.append(eleventh)

    # thirteenths
    _13ths = degrees['13ths']
    _13ths = filter_by_limit(_13ths, upper_lim, mode[-1])
    _13ths = filter_mistuned_5ths(_13ths, mode[4])
    _13ths = filter_8ves(_13ths, mode)
    weight = get_weight(mode, _13ths, alpha, last_mode)
    thirteenth = np.random.choice(_13ths, p=weight)
    mode.append(thirteenth)
    return np.array(mode)


def bass_motion(mode, alpha=4):
    primes = np.array((3, 5, 7), dtype=float)
    choices = -1 * np.eye(len(primes))
    ratios = h_tools.hsv_to_gen_ratios(choices, primes=primes)
    # ratios = np.prod(primes ** choices, axis=1)
    weight = get_weight(mode, ratios, alpha)
    bass_motion = np.random.choice(ratios, p=weight)
    return bass_motion

# potential_adds = [13/8, 16/13, 13/12, 24/13, 39/32, 64/39,
# 13/10, 20/13, 65/64, 128/65,
# 13/7, 14/13, 91/64, 128/91,
# 13/11, 22/13,
# 143/128, 256/143
# 13/9, 18/13,
# 117/64, 128/117]

_3rds = [7/6, 75/64, 6/5, 11/9, 5/4, 9/7] #1.17, 1.29
_5ths = [7/5, 45/32, 3/2, 25/16] #1.4, 1.56
_7ths = [7/4, 9/5, 11/6, 15/8] #1.75, 1.875
_9ths = [21/20, 135/128, 77/72, 16/15, 15/14, 35/32, 12/11, 9/8, 75/64, 7/6]#1.05, 1.17
_11ths = [21/16, 11/8, 45/32, 7/5]#1.31, 1.4
_13ths = [77/48, 8/5, 105/64, 18/11, 5/3, 27/16, 55/32, 12/7]#1.6, 1.71

_3_adds = [16/13, 39/32, 13/11]
_5_adds = [20/13, 91/64, 128/91, 13/9]
_7_adds = [13/7, 256/143, 117/64]
_9_adds = [14/13, 143/128, 128/117]
_11_adds = [13/10, 18/13]
_13_adds = [13/8, 64/39, 22/13]

degrees = {
    '3rds': _3rds,
    '5ths': _5ths,
    '7ths': _7ths,
    '9ths': _9ths,
    '11ths': _11ths,
    '13ths': _13ths
    }

with_13 = False
if with_13:
    degrees['3rds'] += _3_adds
    degrees['5ths'] += _5_adds
    degrees['7ths'] += _7_adds
    degrees['9ths'] += _9_adds
    degrees['11ths'] += _11_adds
    degrees['13ths'] += _13_adds



lower_lim = [7/6, 9/7]
upper_lim = [9/8, 4/3]

# modes = [build_mode(degrees, lower_lim, upper_lim, 4) for i in range(10)]
# json.dump(modes, open('modes.json', 'w'), cls=h_tools.NpEncoder)



modes = [build_mode(degrees, lower_lim, upper_lim, 4)]
for i in range(50):
    bm = bass_motion(modes[-1])
    mode = build_mode(degrees, lower_lim, upper_lim, 4, modes[-1]/bm)
    mode = mode * bm * modes[-1][0]
    while mode[0] >=2:
        mode /= 2
    modes.append(mode)
json.dump(modes, open('modes.json', 'w'), cls=h_tools.NpEncoder)

funds = np.array([mode[0] for mode in modes])
lim = 1.05
inds = np.nonzero(funds<lim)[0]
funds = funds[inds][1:]
cents = [h_tools.hz_to_cents(f, 1) for f in funds]
spread = [cents[i] / inds[i] for i in range(len(funds))]
for i, ind in enumerate(inds):
    if ind >= 10 and ind < 20:
        print(ind)
        print(round(spread[i], 2))
# 
# print(inds)
# print(funds)
# print([round(i, 2) for i in cents])
# print([round(i, 2) for i in spread])
