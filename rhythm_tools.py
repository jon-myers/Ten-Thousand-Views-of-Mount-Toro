import numpy as np
import itertools, math
import pretty_midi

def nPVI(d):
    m = len(d)
    return 100 / (m - 1) * sum([abs((d[i] - d[(i + 1)]) / (d[i] + d[(i + 1)]) / 2) for i in range(m - 1)])


def nCVI(d):
    matrix = [list(i) for i in itertools.combinations(d, 2)]
    matrix = [nPVI(i) for i in matrix]
    return sum(matrix) / len(matrix)



def rhythmic_sequence_maker(num_of_events, nCVI_average, factor=2.0, start_times=False):
    """aka 'rsm'. """
    if nCVI_average == 0:
        section_durs = np.ones(num_of_events) / num_of_events
        starts = np.linspace(0, 1, num_of_events, endpoint=False)
    else:
        section_durs = factor ** np.random.normal(size=2)
        while abs(nCVI(section_durs) - nCVI_average) > 1.0:
            section_durs = factor ** np.random.normal(size=2)
        for i in range(num_of_events - 2):
            next_section_durs = np.append(section_durs,[factor ** np.random.normal()])
            ct=0
            while abs(nCVI(next_section_durs) - nCVI_average) > 1.0:
                ct+=1
                next_section_durs = np.append(section_durs, [factor ** np.random.normal()])
            section_durs = next_section_durs
        section_durs /= np.sum(section_durs)
        cumsum = np.cumsum(section_durs)[:-1]
        starts = np.insert(cumsum, 0, 0)
    if start_times == 'both':
        return section_durs, starts
    elif start_times:
        return starts
    else:
        return section_durs

# def patterned_rsm(num_of_events, nCVI_avg, reps, factor=2.0, start_times=False):
#     """rsm, but split such that the same thing happens repeatedly, thoughts style.
#     need to somehow carry over the info about where the pattern repeats. """
# 


def easy_midi_generator(notes, file_name, midi_inst_name):
    notes = sorted(notes, key=(lambda x: x[1]))
    score = pretty_midi.PrettyMIDI()
    instrument_program = pretty_midi.instrument_name_to_program(midi_inst_name)
    instrument = pretty_midi.Instrument(program=0)
    for n, note in enumerate(notes):
        if type(note[3]) == np.float64:
            vel = np.int(np.round(127 * note[3]))
        elif type(note[3]) == float:
            vel = np.int(np.round(127 * note[3]))
        elif type(note[3]) == int:
            vel = note[3]
        else: print(note[3])
        note = pretty_midi.Note(velocity=vel, pitch=(note[0]), start=(note[1]), end=(note[1] + note[2]))
        instrument.notes.append(note)
    score.instruments.append(instrument)
    score.write(file_name)

def teehi_specifier(dur_tot, sequence, size, start_time=0, nCVI=10, repeats=3,
                    endnote=2, emphasis=1.5, seed=False, order=1, last=False):
    '''size is number of notes in phrase repeated to form teehi'''

    endnote_index = [i for i in range(len(sequence)) if sequence[i] == endnote][-1]
    sequence = sequence[:endnote_index+1]
    durs = rhythmic_sequence_maker(len(sequence), nCVI)
    # durs[-1] *= emphasis

    max_dur_of_phrase = np.max(durs[-size:])
    if durs[-1] < max_dur_of_phrase:
        durs[-1] = emphasis * max_dur_of_phrase
    dur_phrase = durs[-size:]
    durs = np.concatenate((durs, dur_phrase, dur_phrase))[:-1]
    durs = [dur_tot * i / sum(durs) for i in durs] + [dur_tot]
    starts = [sum(durs[:i]) for i in range(len(durs))]
    phrase = sequence[-size:]
    sequence = sequence + phrase + phrase
    starts += [dur_tot]
    notes = [[sequence[i], starts[i]+start_time, 0.1, 60] for i in range(len(sequence))]
    if last == False:
        notes = notes[:-1]
    if seed == False:
        return notes
    else:
        return notes, sequence[-order:]






def sequence_from_sample(samples, bounds):

    """ example usage:
        sample_a = np.random.uniform(size=150)
        sample_a = np.append(sample_a, np.random.uniform(0, 0.01, size=100))
        sample_b = np.random.uniform(size=150)
        sample_c = np.random.uniform(size=150)
        samples = (sample_a, sample_b, sample_c)
        bounds = [(0, 0.5), (0.5, 1)]
        seq = sequence_from_sample(samples, bounds)
    """
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


def spread(init, max_ratio):
    exponent = np.clip(np.random.normal() / 3, -1, 1)
    return init * (max_ratio ** exponent)

def normalize(array):
    array = np.array(array)
    return array / sum(array)

def jiggle_sequence(sequence, spd):
    return normalize(np.array([spread(i, spd) for i in sequence]))

# test = [0.2, 0.2, 0.2, 0.2, 0.2]
# out = jiggle_sequence(test, 1.5)
# print(out)
