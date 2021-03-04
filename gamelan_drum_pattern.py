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

class Transition:
    def __init__(self, sequence):
        self.sequence = sequence
        self.keys = list(set(self.sequence))
        self.counts = [self.sequence.count(i) for i in self.keys]
        self.make_table()
    
    def make_table(self, degree=2):
        combs = list(itertools.product(self.keys, repeat=degree))
        split_combs = [[i for i in combs if i[0] == key] for key in self.keys]
        self.table = {}
        for k, key in enumerate(self.keys):
            self.table[key] = {}
            for key2 in split_combs[k]:
                self.table[key][key2[1]] = 0
        for i in range(len(self.sequence) - (degree-1)):
            self.table[self.sequence[i]][self.sequence[i+1]] += 1
    
    def make_sequence(self, size=20, seed=None):
        if seed == None:
            sequence = [np.random.choice(self.keys)] 
            for step in range(size-1):
                table = self.table[sequence[-1]]
                counts = [table[i] for i in self.keys]
                weights = [i/sum(counts) for i in counts]
                item = np.random.choice(self.keys, p=weights)
                sequence.append(item)
        else: 
            sequence = [seed]
            for step in range(size-1):
                table = self.table[sequence[-1]]
                counts = [table[i] for i in self.keys]
                weights = [i/sum(counts) for i in counts]
                item = np.random.choice(self.keys, p=weights)
                sequence.append(item)
            sequence = sequence[1:]
        return sequence
        
t = Transition(structure)

seq = t.make_sequence(45)
notes, seed = teehi_specifier(20, seq, 6, seed=True)
seq2 = t.make_sequence(30, seed)
notes_2, seed = teehi_specifier(20, seq2, 4, 21, seed=True)
seq3 = t.make_sequence(36, seed)
notes_3 = teehi_specifier(20, seq3, 5, 42)

notes += notes_2
notes += notes_3


easy_midi_generator(notes, 'test_midi.MIDI', 'Acoustic Grand Piano')
