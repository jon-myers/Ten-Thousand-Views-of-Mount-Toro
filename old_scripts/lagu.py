import json
import numpy as np
modes = json.load(open('modes.json', 'r'))


degree_melody = [('5_', 2), ('6_', 1), ('1', 1), ('2', 2), ('2', 2), ('2', 2), 
('1', 1), ('2', 1), ('1', 2), ('3', 2), ('2', 2), ('1', 1), ('6_', 1), ('5_', 2),

('3_', 1), ('5_', 1), ('2_', 2), ('3_', 1), ('5_', 1), ('6_', 1), 
('2', 1), ('2', 3), ('2', 1), ('3', 1), ('2', 1), ('1', 3), 

('3', 1), ('2', 2), ('1', 1), ('2', 1), ('6_', 1),
('2', 1), ('2', 3), ('2', 1), ('3', 1), ('2', 1), ('1', 3),

('3', 1), ('2', 2), ('1', 1), ('6', 1), ('5', 3),
('5', 1), ('6', 1), ('1_', 1), ('6', 1), ('5', 1), ('3', 1),

('2', 1), ('2', 2), ('3', 1), ('5', 1), ('3', 1), ('2', 1), ('1', 1),
('3', 1), ('5', 1), ('3', 1), ('2', 2), ('1', 1), ('6_', 1), ('5_', 4) 
]

def dm_to_freq(dm, mode, key='5_', fund=230):
    """
    Converts degree melody to frequency melody, given mode and fundamental key.
    """
    degrees = ['1_', '2_', '3_', '5_', '6_', '1', '2', '3', '5', '6', '1^', '2^']
    pitches = [degrees.index(i[0]) - degrees.index(key) for i in dm]
    octs = [i // 5 for i in pitches]
    print(pitches)
    print(octs)
    pitch_class = [i%5 for i in pitches]
    print(pitch_class)
    mode = [np.sort(mode)[i] for i in [0, 1, 2, 4, 5]]
    print(mode)
    freqs = [fund * mode[pitch_class[i]] * (2 ** octs[i]) for i in range(len(dm))] 
    return freqs   
    
freqs = dm_to_freq(degree_melody, modes[0])
durs = [i[1] for i in degree_melody]
json.dump(freqs, open('melody.JSON', 'w'))
json.dump(durs, open('durs.JSON', 'w'))
