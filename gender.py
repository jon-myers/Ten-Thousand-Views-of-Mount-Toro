import numpy as np
from scipy.signal import find_peaks
import collections, json
from harmony_tools import utils as h_tools
from rhythm_tools import rhythmic_sequence_maker as rsm

def elaborate_phrase(starting, target, tot_events=10, mode_size=5):
    s = starting
    melody = [s + i * np.sign(target - s) for i in range(np.abs(target-s) + 1)]
    if len(melody) == 1:
        melody = [melody[0], melody[0]]
    while len(melody) < tot_events:
        choices = np.random.choice(np.arange(3), size=tot_events-2)
        counts = collections.Counter(choices)
        while counts[0] + counts[1] + counts[2] > 0:
            choice = np.random.choice(np.arange(3))
            if counts[choice] > 0:
                if choice == 0:
                    if len(melody) - 1 == 1:
                        loc = 1
                    else:
                        loc = np.random.choice(np.arange(1, len(melody)-1))
                    degree = np.random.choice(np.arange(1, 4), p = [0.75, 0.2, 0.05])
                    melody = add_turn(melody, loc, degree)
                    counts[choice] -= 2
                elif choice == 1:
                    if len(melody) - 1 == 1:
                        loc = 1
                    else:
                        loc = np.random.choice(np.arange(1, len(melody)-1))
                    melody = add_jump(melody, loc)
                    counts[choice] -= 1
                elif choice == 2:
                    smaller = counts[choice]
                    if smaller > len(melody):
                        smaller = len(melody)        
                    atom_size = np.random.choice(np.arange(1, smaller+1))
                    if len(melody) - atom_size == 0:
                        loc = 0
                    else:
                        loc = np.random.choice(np.arange(0, len(melody) - atom_size))
                    melody = add_repeat(melody, (loc, loc + atom_size))
                    counts[choice] -= atom_size
    
    while len(melody) > tot_events:
        remove_index = np.random.choice(np.arange(1, len(melody)-1))
        del melody[remove_index]
        
    # make sure target is present at least once before arrival. if not present, 
    # change the most recent peak that is not immidiately before target.
    if np.count_nonzero(np.array(melody) == melody[-1]) < 2:
        peaks = find_peaks(melody)[0]
        peaks = peaks[np.where(peaks != len(melody))]
        peaks = peaks[np.where(peaks != len(melody)-1)]
        if len(peaks) > 0: 
            index = peaks[-1]
            melody[index] = target
        elif tot_events > 3: # if less than 4, don't change, just leave as is
            index = -2
            melody[index] = target
    # make sure penultimate note is not the same as target. If same, toss and 
    # recursively elaborate.
    if melody[-2] == melody[-1]:
        if melody[-3] < melody[-1]:
            melody[-2] = melody[-1] + 1
        elif melody[-3] > melody[-1]:
            melody[-2] = melody[-1] - 1
        else:
            melody[-2] = np.random.choice((melody[-1] + 1, melody[-1] - 1))
    return melody



        
def add_turn(melody, loc=1, degree=1, dir=1):
    """Adds two pitches to a melody at a position specified by `loc`, a single 
    `degree` away from `from_pitch`, on side of `from_pitch` specified by dir.
    If dir is 1, goes toward next pitch. If dir is -1, goes away from it. 
    Loc can't be zero, since turn must take place amidst melody."""
    from_pitch = melody[loc-1]
    to_pitch = melody[loc]
    dir = np.sign(to_pitch - from_pitch) * dir
    inserted_notes = [from_pitch - dir * degree, from_pitch]
    melody[loc:loc] = inserted_notes
    return melody
    
def add_jump(melody, loc=1, degree=1):
    """Adds one pitch to a melody at a certain position specified by `loc`, a 
    pitch one degree away from `to_pitch`, on the opposite side of `to_pitch`
    from `from_pitch`. Loc can't be zero, since turn must take place amidst 
    melody."""
    from_pitch = melody[loc-1]
    to_pitch = melody[loc]
    dir = np.sign(to_pitch - from_pitch)
    insert_note = [to_pitch + dir * degree]
    melody[loc:loc] = insert_note
    return melody
    
def add_repeat(melody, loc=(0, 1), reps=1):
    """Takes a chunk of the melody and inserts repeats of that chunk a given 
    number of times, specified by `reps`."""
    chunk = melody[loc[0]:loc[1]]
    for rep in range(reps):
        melody[loc[1]:loc[1]] = chunk
    return melody
    
def reconcile_differences(melody1, melody2, mode_size=5):
    """1. No octaves except at ends."""
    m1 = np.array(melody1)
    m2 = np.array(melody2)
    mels = (m1, m2)
    oct_locs = np.nonzero(m2 - m1 == mode_size)[0][:-1]
    while len(oct_locs) > 0:
        index = np.random.choice(np.arange(2))
        mels[index][oct_locs[-1]] += 2 * np.random.choice(np.arange(2))
        oct_locs = np.nonzero(m2 - m1 == mode_size)[0][:-1]
        
    # diff = np.abs(oct_locs[1:] - oct_locs[:-1])
    # print(oct_locs, diff)
    return list(m1), list(m2)


def add_contour_teehi(melody1, melody2, size=3, reps=2):
    for rep in range(reps):
        melody1 += melody1[-size:]
        melody2 += melody2[-size:]
    return melody1, melody2
        
    
start = np.random.choice(np.arange(5))
end = np.random.choice(np.arange(5))

melody1 = elaborate_phrase(start, end, 10)
melody2 = elaborate_phrase(start + 5, end + 5, 10)

print(melody2, '\n', melody1)
melody1, melody2 = reconcile_differences(melody1, melody2)
# rep_size = np.random.choice([3, 4, 5, 6])
# melody1, melody2 = add_contour_teehi(melody1, melody2, rep_size, 2)
print('\n', melody2, '\n', melody1)

# melody = add_repeat(melody, (1, 4))
# melody = add_turn(melody, 1, 2, -1)
# melody = add_jump(melody)


durs = rsm(len(melody1), 8) * 10
combined_melody = [list(i) for i in zip(melody1, melody2)] 
json.dump(combined_melody, open('pattern.JSON', 'w'), cls=h_tools.NpEncoder)
json.dump([durs], open('durs.JSON', 'w'), cls=h_tools.NpEncoder)
