import itertools
import numpy as np
from rhythm_tools import rhythmic_sequence_maker, easy_midi_generator, teehi_specifier
t, p, k, b, o = 0, 1, 2, 3, 4


#no duration, just events for transition table
#buka and first ompak
buka = [t, t, p, b, b, p, b, p, b, p, b, p, b, p, b, p, b, p, b, p, p, b, k, t, \
p, b, k, t, p, b]

ompak_l2 = [p, b, k, o, p, k, t, p, b, k, t, p, p, p, p, b, p, k, t, p, \
b, p, k, t, p, b, p, b, p, b, p, b, p, b, k, o, k, o, k, o, p, k, t, p, b, p, b]

ompak_l1 = [k, o, k, o, k, o, k, o, k, o, k, o, k, p, p, k, t, p, b, k, o, \
k, o, k, o, k, k, o, p, k, o, p, k, t, p, b, b, p, k, o, k, o, k, o, p, b, k, t, \
p, b]

#line 1 partial
ngelik_l1p = [p, b, p, b, p, k, t, p, b, p, b, p, b, p, b, p, b, k, o, k, o, \
b, p, k, t, p, b, p, b, p, p, p, b, p]

#line 1 continue
ngelik_l1c = [p, k, o, k, o, k, o, p, b, k, t, p, b]

ngelik_l1_to_suwuk = [p, p, b, k, t, p, b, k, t, p, b, p, b, k, o, p, k, t, \
p, b, k, t, p, p, p, p, b, p, t, t, b, p, t, t, b, p, t, t, b, p, t, t, b, p, t,\
t, b, k, o, k, o, k, o, p, k, k, k]


structure = buka + ompak_l2 + \
ompak_l1 + ompak_l2 + \
ngelik_l1p + ngelik_l1c + ompak_l2 + \
ompak_l1 + ompak_l2 + \
ompak_l1 + ompak_l2 + \
ngelik_l1p + ngelik_l1c + ompak_l2 + \
ompak_l1 + ompak_l2 + \
ompak_l1 + ompak_l2 + \
ngelik_l1p + ngelik_l1_to_suwuk

counts = [structure.count(i) for i in range(5)]

from markov import Markov

def make_sequence(mark, size, seed=None):
    if seed == None:
        states = [i[0] for i in mark.transitions.keys()]
        mark.state = states[np.random.randint(len(states))]
        sequence = mark.choice(k=size - len(mark.state))
    else:
        mark.state = tuple(seed)
        sequence = mark.choice(k=size + 1)[len(mark.state)-1:]
    return sequence

order = 4
mark = Markov()
mark.train(structure, order=order)

states = [i[0] for i in mark.transitions.keys()]
state = states[np.random.randint(len(states))]

seq = make_sequence(mark, 45)
notes, seed = teehi_specifier(20, seq, 6, seed=True, order=order)
seq2 = make_sequence(mark, 35, seed)
notes2, seed = teehi_specifier(20, seq2, 5, 20, seed=True, order=order)
seq3 = make_sequence(mark, 55, seed)
notes3 = teehi_specifier(20, seq3, 7, 40, order=order, last=True)
notes = notes + notes2 + notes3

easy_midi_generator(notes, 'test_midi.MIDI', 'Acoustic Grand Piano')


# print(out)
# t = Transition(structure, 3)
# print(t.table)

# seq = t.make_sequence(45)
# notes, seed = teehi_specifier(20, seq, 6, seed=True)
# seq2 = t.make_sequence(30, seed)
# notes_2, seed = teehi_specifier(20, seq2, 4, 21, seed=True)
# seq3 = t.make_sequence(36, seed)
# notes_3 = teehi_specifier(20, seq3, 5, 42)
#
# notes += notes_2
# notes += notes_3


# easy_midi_generator(notes, 'test_midi.MIDI', 'Acoustic Grand Piano')
