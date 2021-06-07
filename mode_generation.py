from harmony_tools import utils as h_tools
from harmony_tools.utils import dc_alg_step as dc_step
import numpy as np
import json
from fractions import Fraction
import math
import numpy_indexed as npi
from numpy.random import default_rng
rng = default_rng()
Golden = (1 + 5 ** 0.5) / 2

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

def build_mode(alpha, last_mode=None, lm_weighting=[1/3, 2/3]):
    # lm_weighting of [1/3, 2/3] means harmonic distance weighting based on last
    # mode is worth 1/3, while hd weighting from current mode is worth 2/3.

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
    weight = get_weight(mode, ratios, alpha)
    bass_motion = np.random.choice(ratios, p=weight)
    return bass_motion

def insert_mode(preceding_mode, next_mode, after_next_mode, alpha=4):
    primes = np.array((3, 5, 7), dtype=float)
    frac = Fraction(next_mode[0] / preceding_mode[0]).limit_denominator(100)
    choice_indexes = np.where(frac.denominator != primes)[0]
    next_frac = Fraction(after_next_mode[0] / next_mode[0]).limit_denominator(100)
    nf_index = np.where(next_frac.denominator == primes)[0][0]
    choice_indexes = choice_indexes[np.where(choice_indexes != nf_index)[0]]
    ci = np.random.choice(choice_indexes)
    choices = np.eye(len(primes))
    choice = np.product(primes ** choices[ci])
    while choice > 2:
        choice /= 2
    insert_bass = choice * next_mode[0]
    side_arm = Fraction(preceding_mode[0] / insert_bass).limit_denominator(100)
    target_pitches = np.concatenate((preceding_mode, next_mode)) / side_arm
    mode = build_mode(alpha, target_pitches, [0.5, 0.5])
    return mode * side_arm * preceding_mode[0]

def get_alt_modes(preceding_mode, next_mode, alpha=4):
    """Given a preceding mode and a next mode, finds two 'alt' modes, whose root
    is contained by the next mode, and whose pitches are (stochastically) as
    near as possible in HD to both preceding and next mode. These can be
    inserted in between the two modes as things slow down."""
    primes = np.array((3, 5, 7), dtype=float)
    frac = Fraction(next_mode[0] / preceding_mode[0]).limit_denominator(100)
    alt_indexes = np.where(frac.denominator != primes)[0]
    choices = np.eye(len(primes))
    alt_root_0 = np.product(primes ** choices[alt_indexes[0]])
    while alt_root_0 > 2:
        alt_root_0 /= 2
    alt_root_0 *= next_mode[0]

    alt_root_1 = np.product(primes ** choices[alt_indexes[1]])
    while alt_root_1 > 2:
        alt_root_1 /= 2
    alt_root_1 *= next_mode[0]
    side_ratio_0 = Fraction(preceding_mode[0] / alt_root_0).limit_denominator(50)
    side_ratio_1 = Fraction(preceding_mode[0] / alt_root_1).limit_denominator(50)
    target_0 = np.concatenate((preceding_mode, next_mode)) / side_ratio_0
    target_1 = np.concatenate((preceding_mode, next_mode)) / side_ratio_1
    alt_mode_0 = build_mode(alpha, target_0, [0.5, 0.5])
    alt_mode_1 = build_mode(alpha, target_1, [0.5, 0.5])
    alt_mode_0 = alt_mode_0 * side_ratio_0 * preceding_mode[0]
    alt_mode_1 = alt_mode_1 * side_ratio_1 * preceding_mode[1]
    return (alt_mode_0, alt_mode_1)





def make_mode_sequence(size_lims=(6, 30), alpha=4):

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

    len_inds = 0
    while len_inds < 1:
        modes = [build_mode(alpha)]
        for i in range(50):
            bm = bass_motion(modes[-1])
            mode = build_mode(alpha, modes[-1]/bm)
            mode = mode * bm * modes[-1][0]
            while mode[0] >=2:
                mode /= 2
            modes.append(mode)
        # json.dump(modes, open('modes.json', 'w'), cls=h_tools.NpEncoder)
        funds = np.array([mode[0] for mode in modes])
        lim = 0.05
        inds = np.nonzero(np.logical_or(funds < 1 + lim, funds > 2 - lim))[0]
        funds = funds[inds][1:]
        cents = [h_tools.hz_to_cents(f, 1) for f in funds]
        inds = inds[np.where(inds >= size_lims[0])]
        inds = inds[np.where(inds < size_lims[1])]
        len_inds = len(inds)

    funds = [i[0] for i in modes][:inds[0]+1]
    off = funds[-1]
    if off > 1.5: off /= 2
    # base = math.e ** (math.log(1/off) / (2 * (len(funds) - 1)))
    # mult = base ** np.arange(2 * inds[0])
    modes = np.array(modes[:inds[0]])




    # find two alt modes for between each mode
    alt_modes = []
    for i in range(len(modes)):
        if (i+1) % len(modes) == 0:
            next_mode = modes[0] * off
        else:
            next_mode = modes[i+1]
        alts = get_alt_modes(modes[i], next_mode)
        alt_modes.append(alts)

    base = math.e ** (math.log(1/off) / (len(funds)-1))
    mult = base ** np.arange(len(funds)-1)
    mult = np.expand_dims(mult, 1)
    modes = np.array(modes) * mult
    var_0 = np.array(alt_modes)[:,0] * mult
    var_1 = np.array(alt_modes)[:, 1] * mult


    concat_modes = np.array((modes, var_0, var_1))
    return concat_modes


def make_melody(modes, variations):
    """Returns a target pitch that overlaps with mode variations."""
    melody = []
    for i in range(len(modes)):
        A = h_tools.gen_ratios_to_hsv(modes[i]/modes[i][0], [3, 5, 7])
        B = h_tools.gen_ratios_to_hsv(variations[0][i]/modes[i][0], [3, 5, 7])
        C = h_tools.gen_ratios_to_hsv(variations[1][i]/modes[i][0], [3, 5, 7])

        abc = npi.intersection(A, B, C)
        ab = npi.intersection(A, B)
        bc = npi.intersection(B, C)
        ac = npi.intersection(A, C)
        if len(abc) > 0:
            melody_note = abc[np.random.choice(np.array(len(abc)))]
            index = np.nonzero(np.all(np.equal(A, melody_note), axis=1))[0][0]
            melody.append((0, index))
        elif len(ab) > 0:
            melody_note = ab[np.random.choice(np.array(len(ab)))]
            index = np.nonzero(np.all(np.equal(A, melody_note), axis=1))[0][0]
            melody.append((0, index))
        elif len(ac) > 0:
            melody_note = ac[np.random.choice(np.array(len(ac)))]
            index = np.nonzero(np.all(np.equal(A, melody_note), axis=1))[0][0]
            melody.append((0, index))
        elif len(bc) > 0:
            melody_note = bc[np.random.choice(np.array(len(bc)))]
            index = np.nonzero(np.all(np.equal(B, melody_note), axis=1))[0][0]
            melody.append((1, index))
        else:
            melody.append((0, 0))
    return melody


class Note_Stream:
    """For a given mode and statistical profile, generate chords of various
    sizes and containing notes in a weighted dc_alg manner. """

    def __init__(self, mode, fund, weight=None, chord_sizes=[2, 3, 4, 5],
                 cs_weight=None):
        self.mode = mode
        self.fund = fund
        self.wt = weight
        self.cts = None
        self.cs = chord_sizes
        self.cs_wt = cs_weight
        self.cs_cts = None
        # self.gamut_size = gamut_size
        # self.gamut = []
        # self.make_gamut()
        # self.g_cts = None

    def cs_step(self):
        cs_i, self.cs_cts = dc_step(len(self.cs), self.cs_cts, 2, self.cs_wt)
        chord_size = self.cs[cs_i]
        self.next_cs = chord_size

    def note_step(self):
        self.mode_idxs = []
        for i in range(self.next_cs):
            mode_idx, self.cts = dc_step(len(self.mode), self.cts, 2, self.wt)
            while mode_idx in self.mode_idxs:
                mode_idx, self.cts = dc_step(len(self.mode), self.cts, 2, self.wt)
            self.mode_idxs.append(mode_idx)
        if len(self.mode_idxs) > len(list(set(self.mode_idxs))):
            breakpoint()
        self.mode_idxs = np.array(self.mode_idxs)

    def next_chord(self, register=None):
        """Register is a tuple (min, max) of frequencies"""
        self.cs_step()
        self.note_step()
        chord = self.mode[self.mode_idxs] * self.fund
        if np.all(register != None):
            chord = self.registrate(chord, register)
        return chord
        
    def registrate(self, chord, register):
        for i, note in enumerate(chord):
            min_exp = np.ceil(np.log2(register[0]/note))
            max_exp = np.floor(np.log2(register[1]/note))
            exp_options = np.arange(min_exp, max_exp+1)
            exp = np.random.choice(exp_options)
            chord[i] *= 2 ** exp
        return chord
    
    # def make_gamut(self):
    # 
    #     for i in range(self.gamut_size):
    #         self.gamut.append(self.next_chord())
    # 
    # def next_gamut_chord(self, register):
    #     next_idx, self.g_cts = dc_step(self.gamut_size, self.g_cts, 2)
    #     return self.gamut[next_idx]
    # 




def get_sub_mode(mode, num_of_pitches, weight=None):
    if np.all(weight==None):
        weight = 1 / (Golden ** (np.arange(len(mode))/2))
    weight /= np.sum(weight)
    sub_mode = rng.choice(mode, num_of_pitches, p=weight, replace=False)
    sub_mode = np.sort(sub_mode)
    return sub_mode
