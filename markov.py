import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rhythm_tools import nCVI, spread, start_cumsum
from rhythm_tools import rhythmic_sequence_maker as rsm
import json
rng = np.random.default_rng()
p_init = np.array([0.1, 0.8, 0.1])
p_transition = np.array(
    [[0.90, 0.05, 0.05],
     [0.01, 0.90, 0.09],
     [0.07, 0.03, 0.9]]
)


from scipy.stats import multinomial

def equilibrium_distribution(p_transition):
    n_states = p_transition.shape[0]
    A = np.append(
        arr=p_transition.T - np.eye(n_states),
        values=np.ones(n_states).reshape(1, -1),
        axis=0
    )
    b = np.transpose(np.array([0] * n_states + [1]))
    p_eq = np.linalg.solve(
        a=np.transpose(A).dot(A),
        b=np.transpose(A).dot(b)
    )
    return p_eq

def markov_sequence(p_init=None, p_transition=None, sequence_length=None):
    if p_init is None:
        p_init = equilibrium_distribution(p_transition)
    initial_state = list(multinomial.rvs(1, p_init)).index(1)
    states = [initial_state]
    for _ in range(sequence_length - 1):
        p_tr = p_transition[states[-1]]
        new_state = list(multinomial.rvs(1, p_tr)).index(1)
        states.append(new_state)
    return states

# states = markov_sequence(p_init, p_transition, sequence_length=1000)
# p_transition = rng.dirichlet(np.ones(5) * 10, (20))
# # print(p_transition)
# p_init = rng.dirichlet(np.ones(5) * 10, 1)[0]
# # print(p_init)
# states = markov_sequence(p_init, p_transition, 10)
# print(states)

from harmony_tools import utils as h_tools
from harmony_tools import plot as h_plot
import json
import itertools


def generate_transition_table(hsv):

    #TODO circle back to this, think about if it is possible to incorporate
    # distance of contour into this formulation. Closer in contour should be
    # higher probability, somehow. Would it work to make a second transition
    # table based entirely on contour distance, and then combine them somehow,
    # via multiplication and renormalization?
    p_transition = np.zeros((len(hsv), len(hsv)))
    for p, pitch in enumerate(hsv):
        containment_arr = np.array([h_tools.containment_relationship(pitch, vec) for vec in hsv])
        containment_arr[p] = 1.5 # this gives different status to 'same'
        unordered_tr_probs = rng.dirichlet(np.ones(len(hsv)))
        sort_arr_probs = np.argsort(unordered_tr_probs)[::-1]
        sort_arr_containment = np.argsort(containment_arr)
        ct = 0
        for i in range(4)[::-1]:
            idxs = [j for j, _ in enumerate(containment_arr) if _ == i]
            order = rng.choice(idxs, len(idxs), replace=False)
            for o in order:
                p_transition[p][o] = unordered_tr_probs[sort_arr_probs[ct]]
                ct += 1
    return p_transition


def registrate(note, low, high):
    low = np.ceil(np.log2((low/note).astype(float)))
    high = np.floor(np.log2((high/note).astype(float)))
    oct = np.where(low==high, low, np.random.randint(low, high+1))
    return note * (2 ** oct)

    # registrate where you care about previous so, so you don't jump around too much!!!

def enumerate_freqs(mode, fund, low, high, for_pivots=True):
    mode = np.array(mode)
    ordered_mode = mode[np.argsort(mode)]
    lowest_idx = np.argmin(np.ceil(np.log2((low/(ordered_mode*fund)).astype(float))))
    init_oct = np.ceil(np.log2((low/(ordered_mode[lowest_idx]*fund))))
    freqs = [ordered_mode[lowest_idx]*fund * (2 ** init_oct)]
    oct_ct = 0
    while freqs[-1] <= high:
        lowest_idx += 1
        oct = (lowest_idx // len(mode)) + init_oct
        idx = lowest_idx % len(mode)
        next_freq = ordered_mode[idx] * fund * (2 ** oct)
        freqs.append(next_freq)
    # if it is for pivots, you get one upper pitch above high.
    if for_pivots == False:
        freqs = freqs[:-1]
    return freqs

# enumerate_freqs(modes[0][2], 200, 300, 700)


def make_pluck_phrase(mode, fund, size, dur_tot, nCVI,
    freq_range=(250, 500), start_idx=None, end_idx=None, p_transition=None,
    pivot_ratio=0.25, hold_ratio=0.6, attack_ratio=0.66, pluck_amp=0.75, coef=0.45):
    """Makes a dictionary that can be interpretted by  supercollider SynthDef
    instrument `\moving_pluck`.

    mode: (np array floats) list of frequencies.
    fund: (float) fundamental frequency.
    size: (integer) number of notes in mode.
    dur_tot: (float) total duration of phrase.
    nCVI: (float) normalized combinatorial variability index.
    freq_range: (tuple of floats) minimum and maximum pitch of phrase.
    start_idx: (int) index of phrase's first note in mode array.
    end_idx: (int) index of phrase's last note in mode array.
    p_transition: (np array) Markov transition table.
    pivot_ratio: (float) proportion of items in pivot array that are freqs, as
        opposed to nils.
    hold_ratio: (float or np array of floats) proportion of note duration to
        stay on initial freq before beginning cosine interpolation to pivot or
        next note.
    attack_ratio: (float) proportion of notes that are rearticulated via pluck.
    pluck_amp: (float) avg amplitude of pluck artiuclations.
    """

    mode = np.array(mode)
    hsv = h_tools.gen_ratios_to_hsv(mode, [3, 5, 7, 11])
    p_transition = generate_transition_table(hsv)
    if start_idx is None:
        start_idx = np.random.randint(0, len(mode))
    if p_transition is None:
        p_transition = generate_transition_table(hsv)
    p_init = np.zeros(len(mode))
    p_init[start_idx] = 1
    note_seq = markov_sequence(p_init, p_transition, size)
    if end_idx != None:
        while note_seq[-1] != end_idx:
            note_seq = markov_sequence(p_init, p_transition, size)
    durs = rsm(size, nCVI) * dur_tot
    notes = registrate(mode[note_seq]*fund, freq_range[0], freq_range[1])

    num_of_pivots = np.round(pivot_ratio * (size-1))
    in_range_freqs = enumerate_freqs(mode, fund, freq_range[0], freq_range[1])
    pivot_locs = rng.choice(np.arange(size-1), num_of_pivots.astype(int), replace=False)
    pivots = []
    for i in range(len(notes)-1):
        if i in pivot_locs:
            high = np.max(notes[i:i+2])
            ir_freq_idx = in_range_freqs.index(high)
            pivot = in_range_freqs[ir_freq_idx+1]
            pivots.append(pivot)
        else:
            pivots.append('nil')

    if np.size(hold_ratio) == 1:
        hold_ratio = np.repeat(hold_ratio, size-1)

    num_of_plucks = np.round(attack_ratio*size).astype(int)
    if num_of_plucks == 0: num_of_plucks = 1
    pluck_amps = np.repeat(pluck_amp, num_of_plucks)
    pluck_amps = [spread(i, 4/3) for i in pluck_amps]
    pluck_locs = [0]
    num_of_plucks -= 1
    other_pluck_locs = rng.choice(np.arange(1, size), num_of_plucks, replace=False)
    pluck_locs = np.sort(np.concatenate((pluck_locs, other_pluck_locs)))
    starts = np.concatenate(([0], np.cumsum(durs)[:-1]))
    pluck_starts = starts[pluck_locs]
    wait_props = np.repeat(hold_ratio, size-1)
    wait_props = [spread(i, 5/4) for i in wait_props]
    moving_pluck_dict = {}
    moving_pluck_dict['notes'] = notes
    moving_pluck_dict['pivots'] = pivots
    moving_pluck_dict['durs'] = durs
    moving_pluck_dict['waitProps'] = wait_props
    moving_pluck_dict['pluckStarts'] = pluck_starts
    moving_pluck_dict['pluckAmps'] = pluck_amps
    moving_pluck_dict['durTot'] = dur_tot
    moving_pluck_dict['releaseDur'] = 3 * dur_tot
    moving_pluck_dict['symps'] = make_symps_from_mode(mode, fund)
    moving_pluck_dict['coef'] = coef
    return moving_pluck_dict


def closest_index(test, arr):
    logs = np.log2((test/arr).astype(float))
    dists = np.abs(np.round(logs) - logs)
    return np.argmin(dists)

def make_changing_pluck_phrase(mode_a, mode_b, fund, size, dur_tot, nCVI,
    p_transition_a, p_transition_b, proportion=0.5, freq_range=(250, 500),
    start_idx=None, end_idx=None, pivot_ratio=0.8, hold_ratio=0.75, attack_ratio=0.66,
    pluck_amp=0.75):
    """Makes a dictionary that can be interpretted by  supercollider SynthDef
    instrument `\moving_pluck`.

    changes from one mode to another, midway through. Any start_idx and end_idx
    is given for the first thing. When it gets taken over by second thing,
    continue from nearest note to current note, and nearest note to target note.

    mode: (np array floats) list of frequencies.
    fund: (float) fundamental frequency.
    size: (integer) number of notes in mode.
    dur_tot: (float) total duration of phrase.
    nCVI: (float) normalized combinatorial variability index.
    freq_range: (tuple of floats) minimum and maximum pitch of phrase.
    start_idx: (int) index of phrase's first note in mode array.
    end_idx: (int) index of phrase's last note in mode array.
    p_transition: (np array) Markov transition table.
    pivot_ratio: (float) proportion of items in pivot array that are freqs, as
        opposed to nils.
    hold_ratio: (float or np array of floats) proportion of note duration to
        stay on initial freq before beginning cosine interpolation to pivot or
        next note.
    attack_ratio: (float) proportion of notes that are rearticulated via pluck.
    pluck_amp: (float) avg amplitude of pluck artiuclations.
    """

    mode_a = np.array(mode_a)
    mode_b = np.array(mode_b)

    if start_idx is None:
        start_idx = np.random.randint(0, len(mode_a))
    p_init = np.zeros(len(mode_a))
    p_init[start_idx] = 1
    note_seq = markov_sequence(p_init, p_transition_a, size)
    if end_idx != None:
        while note_seq[-1] != end_idx:
            note_seq = markov_sequence(p_init, p_transition_a, size)
    durs = rsm(size, nCVI) * dur_tot
    midpoint = proportion * dur_tot
    starts = start_cumsum(durs)
    crx_idx = np.where(starts > midpoint)[0][0]

    crx_durs = durs[crx_idx:]
    last_note_idx = note_seq[crx_idx-1]
    last_note = mode_a[last_note_idx]
    crx_closest_st_idx = closest_index(last_note, mode_b)
    crx_closest_end_idx = closest_index(mode_a[note_seq[-1]], mode_b)
    crx_p_init = np.zeros(len(mode_b))
    crx_p_init[crx_closest_st_idx] = 1
    crx_note_seq = markov_sequence(crx_p_init, p_transition_b, size - crx_idx + 1)[1:]
    if end_idx != None:
        while crx_note_seq[-1] != crx_closest_end_idx:
            crx_note_seq = markov_sequence(crx_p_init, p_transition_b, size - crx_idx + 1)[1:]

    # make notes from first half note_seq, second half crx_note_seq.
    notes = np.concatenate((mode_a[note_seq][:crx_idx], mode_b[crx_note_seq]))

    notes = registrate(notes*fund, freq_range[0], freq_range[1])


    num_of_pivots = np.round(pivot_ratio * (size-1))
    in_range_freqs_a = enumerate_freqs(mode_a, fund, freq_range[0], freq_range[1])
    in_range_freqs_b = enumerate_freqs(mode_b, fund, freq_range[0], freq_range[1])


    pivot_locs = rng.choice(np.arange(size-1), num_of_pivots.astype(int), replace=False)
    pivots = []
    for i in range(len(notes)-1):
        if i in pivot_locs:
            if i+1 < crx_idx:
                high = np.max(notes[i:i+2])
                ir_freq_idx = in_range_freqs_a.index(high)
                pivot = in_range_freqs_a[(ir_freq_idx+1) % len(mode_a)]
            elif i+1 == crx_idx:
                high = notes[i+1]
                if high < notes[i]:
                    ir_freq_idx = in_range_freqs_b.index(high) + 1
                else:
                    ir_freq_idx = in_range_freqs_b.index(high)
                pivot = in_range_freqs_b[(ir_freq_idx+1) % len(mode_b)]
            else:
                high = np.max(notes[i:i+1])
                ir_freq_idx = in_range_freqs_b.index(high)
                pivot = in_range_freqs_b[(ir_freq_idx+1) % len(mode_b)]
            pivots.append(pivot)
        else:
            pivots.append('nil')

    # if np.size(hold_ratio) == 1:
    #     hold_ratio = np.repeat(hold_ratio, size-1)

    num_of_plucks = np.round(attack_ratio*size).astype(int)
    if num_of_plucks == 0: num_of_plucks = 1
    pluck_amps = np.repeat(pluck_amp, num_of_plucks)
    pluck_amps = [spread(i, 4/3) for i in pluck_amps]
    pluck_locs = [0]
    num_of_plucks -= 1
    other_pluck_locs = rng.choice(np.arange(1, size), num_of_plucks, replace=False)
    pluck_locs = np.sort(np.concatenate((pluck_locs, other_pluck_locs)))
    starts = np.concatenate(([0], np.cumsum(durs)[:-1]))
    pluck_starts = starts[pluck_locs]
    if np.size(hold_ratio) == 1:
        hold_ratio = np.repeat(hold_ratio, size-1)
        wait_props = [spread(i, 5/4) for i in hold_ratio]
    else: wait_props = hold_ratio
    moving_pluck_dict = {}
    moving_pluck_dict['notes'] = notes
    moving_pluck_dict['pivots'] = pivots
    moving_pluck_dict['durs'] = durs
    moving_pluck_dict['waitProps'] = wait_props
    moving_pluck_dict['pluckStarts'] = pluck_starts
    moving_pluck_dict['pluckAmps'] = pluck_amps
    moving_pluck_dict['durTot'] = dur_tot
    moving_pluck_dict['releaseDur'] = 3 * dur_tot

    return moving_pluck_dict


def make_symps_from_mode(mode, fund):
    min_freq = 200
    max_freq = min_freq * 2 ** 3
    freqs = mode * fund;
    symp_freqs = np.array([])
    for f in freqs:
        low_exp = np.ceil(np.log2(min_freq/f)).astype(int)
        high_exp = np.floor(np.log2(max_freq/f)).astype(int)
        exp = 2.0 ** np.arange(low_exp, high_exp+1)
        symp_freqs = np.concatenate((symp_freqs, f*exp))
    return symp_freqs

def make_multi_changing_pluck_phrase(modes, fund, size, dur_tot, nCVI,
    p_transitions, proportions, freq_range=(250, 500),
    start_idx=None, end_idx=None, pivot_ratio=0.25, hold_ratio=0.5, attack_ratio=0.66,
    pluck_amp=0.75, coef=0.4):
    """Makes a dictionary that can be interpretted by  supercollider SynthDef
    instrument `\moving_pluck`.

    can go between 2 or more modes now.

    `modes`, `p_transitions`, and `proportions` are now tuples w/ multiple items.

    changes from one mode to another, midway through. Any start_idx and end_idx
    is given for the first thing. When it gets taken over by second thing,
    continue from nearest note to current note, and nearest note to target note.

    mode: (np array floats) list of frequencies.
    fund: (float) fundamental frequency.
    size: (integer) number of notes in mode.
    dur_tot: (float) total duration of phrase.
    nCVI: (float) normalized combinatorial variability index.
    freq_range: (tuple of floats) minimum and maximum pitch of phrase.
    start_idx: (int) index of phrase's first note in mode array.
    end_idx: (int) index of phrase's last note in mode array.
    p_transition: (np array) Markov transition table.
    pivot_ratio: (float) proportion of items in pivot array that are freqs, as
        opposed to nils.
    hold_ratio: (float or np array of floats) proportion of note duration to
        stay on initial freq before beginning cosine interpolation to pivot or
        next note.
    attack_ratio: (float) proportion of notes that are rearticulated via pluck.
    pluck_amp: (float) avg amplitude of pluck artiuclations.
    """

    modes = [np.array(i) for i in modes]

    if start_idx is None:
        start_idx = np.random.randint(0, len(modes[0]))
    p_init = np.zeros(len(modes[0]))
    p_init[start_idx] = 1
    # note_seq = markov_sequence(p_init, p_transitions[0], size)
    # if end_idx != None:
    #     while note_seq[-1] != end_idx:
    #         note_seq = markov_sequence(p_init, p_transitions[0], size)
    durs = rsm(size, nCVI) * dur_tot
    midpoints = np.array(proportions) * dur_tot
    starts = start_cumsum(durs)

    while len(midpoints) > 1 and np.all(starts < midpoints[-1]):
        midpoints = midpoints[:-1]

    if len(midpoints) == 1 and np.all(starts < midpoints[0]):
        print('this one actually does happen sometimes')
        return make_pluck_phrase(modes[0], fund, size, dur_tot, nCVI, freq_range,
                start_idx, end_idx, p_transitions[0], pivot_ratio, hold_ratio,
                attack_ratio, pluck_amp, coef)
    else:

        crx_idxs = [np.where(starts > mp)[0][0] for mp in midpoints]

        # crx_durs = durs[crx_idx:]
        all_durs = []
        all_durs.append(durs[:crx_idxs[0]])
        for i, crx_idx in enumerate(crx_idxs):
            if i != len(crx_idxs) - 1:
                crx_durs = durs[crx_idx: crx_idxs[i+1]]
            else:
                crx_durs = durs[crx_idx:]
            all_durs.append(crx_durs)

        note_seqs = []
        for i, ad in enumerate(all_durs):
            note_seq = markov_sequence(p_init, p_transitions[i], len(ad))
            note_seqs.append(note_seq)
            if i < len(all_durs) - 1:
                p_init = np.zeros(len(modes[i+1]))
                p_init[closest_index(modes[i][note_seq[-1]], modes[i+1])] = 1
        # breakpoint()
        if not end_idx is None:
            ideal_last_idx = closest_index(modes[0][note_seqs[0][0]], modes[-1])
            while note_seqs[-1][-1] != ideal_last_idx:
                p_init = np.zeros(len(modes[0]))
                p_init[start_idx] = 1
                note_seqs = []

                for i, ad in enumerate(all_durs):
                    note_seq = markov_sequence(p_init, p_transitions[i], len(ad))
                    note_seqs.append(note_seq)
                    if i < len(all_durs) - 1:
                        p_init = np.zeros(len(modes[i+1]))
                        p_init[closest_index(modes[i][note_seq[-1]], modes[i+1])] = 1

        all_notes = [modes[i][note_seqs[i]] for i in range(len(all_durs))]
        all_notes = np.concatenate(all_notes)
        all_notes = registrate(all_notes*fund, freq_range[0], freq_range[1])
        
        if len(all_notes) != len(durs):
            sys.exit("Will break Supercollider, no good, try again!")



        in_range_freqs = [enumerate_freqs(modes[i], fund, freq_range[0], freq_range[1]) for i in range(len(all_durs))]
        num_of_pivots = np.round(pivot_ratio * (size-1))
        pivot_locs = rng.choice(np.arange(size-1), num_of_pivots.astype(int), replace=False)
        pivot_locs = np.sort(pivot_locs)
        segmented_pivot_locs = [[] for _ in range(len(all_durs))]
        for pl in pivot_locs:
            for i, ad in enumerate(all_durs):
                lo = sum([len(j) for j in all_durs[:i]])
                hi = sum([len(j) for j in all_durs[:i+1]])

                if pl < hi-1 and pl >= lo-1:
                    segmented_pivot_locs[i].append(pl)

        pivots = []
        for i in range(len(all_notes)-1):
            if i in pivot_locs:

                for j in range(len(segmented_pivot_locs)):
                    if i in segmented_pivot_locs[j]:
                        hi = np.max(all_notes[i:i+2])
                        higher_idx = np.where(np.array(in_range_freqs[j]) > hi)[0][0]
                        pivot = np.array(in_range_freqs[j])[higher_idx]
                        pivots.append(pivot)
            else:
                pivots.append('nil')


        # if np.size(hold_ratio) == 1:
        #     hold_ratio = np.repeat(hold_ratio, size-1)


        num_of_plucks = np.round(attack_ratio*size).astype(int)
        if num_of_plucks == 0: num_of_plucks = 1
        pluck_amps = np.repeat(pluck_amp, num_of_plucks)
        pluck_amps = [spread(i, 4/3) for i in pluck_amps]
        pluck_locs = [0]
        num_of_plucks -= 1
        other_pluck_locs = rng.choice(np.arange(1, size), num_of_plucks, replace=False)
        pluck_locs = np.sort(np.concatenate((pluck_locs, other_pluck_locs)))
        starts = np.concatenate(([0], np.cumsum(durs)[:-1]))
        pluck_starts = starts[pluck_locs]
        if np.size(hold_ratio) == 1:
            hold_ratio = np.repeat(hold_ratio, size-1)
            wait_props = [spread(i, 5/4) for i in hold_ratio]
        else: wait_props = hold_ratio
        moving_pluck_dict = {}
        moving_pluck_dict['notes'] = all_notes
        moving_pluck_dict['pivots'] = pivots
        moving_pluck_dict['durs'] = durs
        moving_pluck_dict['waitProps'] = wait_props
        moving_pluck_dict['pluckStarts'] = pluck_starts
        moving_pluck_dict['pluckAmps'] = pluck_amps
        moving_pluck_dict['durTot'] = dur_tot
        moving_pluck_dict['releaseDur'] = 3 * dur_tot
        moving_pluck_dict['symps'] = make_symps_from_mode(modes[0], fund)
        moving_pluck_dict['coef'] = coef

        # here's where I will do the converting to 'real time' stuff


        return moving_pluck_dict


def make_pitch_pairs(mode, transition_table, start_idx=None, end_idx=None):
    if not start_idx is None:
        p_init = np.zeros(len(mode))
        p_init[start_idx] = 1
    else:
        p_init = None
    seq = markov_sequence(p_init, transition_table, 2)
    if not end_idx is None:
        while seq[-1] != end_idx:
            seq = markov_sequence(p_init, transition_table, 2)
    return seq


# modes = json.load(open('JSON/modes_and_variations.json', 'rb'))
# # #
# # mode_a = modes[0][1]
# # mode_b = modes[0][2]
# modes_ = modes[0][:3]
# # #
# tt_a = generate_transition_table(h_tools.gen_ratios_to_hsv(modes_[0], [3, 5, 7, 11]))
# tt_b = generate_transition_table(h_tools.gen_ratios_to_hsv(modes_[1], [3, 5, 7, 11]))
# tt_c = generate_transition_table(h_tools.gen_ratios_to_hsv(modes_[2], [3, 5, 7, 11]))
# p_transitions = [tt_a, tt_b, tt_c]
# # #
# phrase = make_multi_changing_pluck_phrase(modes_, 200, 10, 7, 40, p_transitions,
#     [0.33, 0.67], end_idx=3)
# print(phrase)
#
#
# json.dump(phrase, open('JSON/moving_pluck_phrase.JSON', 'w'), cls=h_tools.NpEncoder)


# hsv = h_tools.gen_ratios_to_hsv(modes[0][4], [3, 5, 7, 11])
# ord_hsv = h_tools.cast_to_ordinal(hsv)
# p_tr = generate_transition_table(ord_hsv)
# seq = markov_sequence(p_transition=p_tr, sequence_length=30)
# phrase = make_pluck_phrase(modes[0][7], 200, 8, 3, 20, start_idx=0, end_idx=4)
# json.dump(phrase, open('JSON/moving_pluck_phrase.JSON', 'w'), cls=h_tools.NpEncoder)
