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



def rhythmic_sequence_maker(num_of_thoughts,nCVI_average,factor=2.0):
    section_durs = factor ** np.random.normal(size=2)
    while abs(nCVI(section_durs) - nCVI_average) > 1.0:
        section_durs = factor ** np.random.normal(size=2)
    for i in range(num_of_thoughts - 2):
        next_section_durs = np.append(section_durs,[factor ** np.random.normal()])
        ct=0
        while abs(nCVI(next_section_durs) - nCVI_average) > 1.0:
            ct+=1
            next_section_durs = np.append(section_durs, [factor ** np.random.normal()])
        section_durs = next_section_durs
        # print(ct)
    section_durs /= np.sum(section_durs)
    return section_durs

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
    print(sequence)
    print()
    starts += [dur_tot]
    notes = [[sequence[i], starts[i]+start_time, 0.1, 60] for i in range(len(sequence))]
    if last == False:
        notes = notes[:-1]
    if seed == False:
        return notes
    else:
        return notes, sequence[-order:]






#
# dur = 60 * 2
# num_of_sections = 13
# avg_td = 2
# avg_ncvi = 8
# durs = dur * rhythmic_sequence_maker(num_of_sections, 10)
# starts = [sum(durs[:i]) for i in range(len(durs))]
# tds = rhythmic_sequence_maker(num_of_sections, 10) * avg_td * num_of_sections
# ncvis = rhythmic_sequence_maker(num_of_sections, 10) * num_of_sections * avg_ncvi
#
# all_events = []
# for s in range(num_of_sections):
#     num_of_attacks = int(tds[s] * durs[s] // 1)
#     event_durs = rhythmic_sequence_maker(num_of_attacks, ncvis[s]) * durs[s]
#     event_starts = [sum(event_durs[:i]) + starts[s] for i in range(len(event_durs))]
#     all_events += list(event_starts)
# all_durs = [all_events[i+1] - all_events[i] for i in range(len(all_events) - 1)]
# all_durs.append(dur - all_events[-1])
# notes = [[60, all_events[i], all_durs[i], 85] for i in range(len(all_durs))]
# for note in notes:
#     print(note)
# easy_midi_generator(notes, 'test_midi.MIDI', 'Acoustic Grand Piano')


# for one cycle, single drum, 13 modes long, each has a different temporal density
# and different
