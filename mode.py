import numpy as np
import math
import numpy_indexed as npi
from harmony_tools import utils as h_tools
from fractions import Fraction

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
    

intervals = np.concatenate((one_step, two_step, three_step))
intervals = npi.unique(intervals)
fracs = hsv_to_frac(intervals, primes)
hds = np.array([harmonic_distance(frac) for frac in fracs])
hd_sorts = np.argsort(hds)
max_hd = 5
#filter out anything beneath max hd
hd_sorts = hd_sorts[hds[hd_sorts] <=5]
sorted_hsvs = intervals[hd_sorts]
sorted_fracs = [fracs[i] for i in hd_sorts]
rats = h_tools.hsv_to_gen_ratios(sorted_hsvs, primes)

monophonic_rats = np.sort(rats)


# Tenney's 
# prime to third and third to fifth
low_third_range = [7/6, 9/7]
high_third_range = [9/8, 4/3]


def mistuned_filter(possible, mistuned_int, p_int, origin):
    mt_min = h_tools.cents_to_hz(-mistuned_int, p_int)
    mt_max = h_tools.cents_to_hz(mistuned_int, p_int)
    condition_1 = origin * possible <= mt_min
    condition_2 = origin * possible == p_int
    condition_3 = origin * possible >= mt_max
    filter = np.any((condition_1, condition_2, condition_3), axis=0)
    return filter
    
    
    
    
def build_mode(ratios, third_lims=[7/6, 9/7], mistuned_5th=50, mistuned_8ve=66):
    ratios = np.concatenate((ratios, 2 * ratios))
    mode = [1]
    for i in range(6):
        lower_lim = third_lims[0] * mode[-1]
        upper_lim = third_lims[1] * mode[-1]
        # while upper_lim >= 2:
        #     upper_lim /= 2
        # while lower_lim >= 2:
        #     lower_lim /= 2
        above = ratios >= lower_lim
        below = ratios <= upper_lim
        possible = ratios[np.all((above, below), axis=0)]
        # print(possible, '\n\n')
        if i > 0:
            p_fifth = 1.5 * mode[-2]
            filter = mistuned_filter(possible, mistuned_5th, p_fifth, mode[-1])
            possible = possible[filter]
        
        if i == 2:
            oct = 2 * mode[-3]
            filter = mistuned_filter(possible, mistuned_8ve, oct, mode[-1])
            possible = possible[filter]
            
        if i > 2:
            octs = [2 * mode[-3], 2 * mode[-4]]
            filter_A = mistuned_filter(possible, mistuned_8ve, octs[0], mode[-1])
            filter_B = mistuned_filter(possible, mistuned_8ve, octs[1], mode[-1])

            filter = np.all((filter_A, filter_B), axis=0)
            possible = possible[filter]
            
        choice_hds = []
        for choice in possible:
            aggregate_hd = []
            for current in mode:
                hd = harmonic_distance_mixed(current, choice)
                aggregate_hd.append(hd)
            choice_hds.append(sum(aggregate_hd))
        alpha = 4
        weight = [1/(i**alpha) for i in choice_hds]
        weight = [i/sum(weight) for i in weight]
        print([round(i, 2) for i in weight])
        print()
        mode.append(np.random.choice(possible, p=weight))
            
    return mode
        
        
        # test remaining possible, stochastically choose with weights based on
        # aggregate harmonic distance
        
# print(rats)
mode = build_mode(monophonic_rats)
print(mode)
    
    
