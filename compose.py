from tempo_curve import Time
from rhythm_tools import rhythmic_sequence_maker as rsm
import numpy as np
from mode_generation import make_mode_sequence, make_melody
import json
from harmony_tools import utils as h_tools
import numpy_indexed as npi
from textures import Thoughts_Texture




noc = 7
dur_tot = 60*60
fund = 150
modes, variations = make_mode_sequence((10, 20))
melody = make_melody(modes, variations)
events_per_cycle = len(modes)
t = Time(dur_tot=dur_tot, f=0.3, noc=noc)
t.set_cycle(len(modes))


# adds thoughts textures to event_dur_dict
for i in range(len(modes)):
    section = t.event_dur_dict[i]
    for subdiv in range(1, 6):
        seq = section[subdiv]['sequence']
        section[subdiv]['texture'] = {}
        for v, var in enumerate(seq):
            if var == melody[i][0]:
                target = melody[i][1]
            else:
                target = np.random.choice(np.arange(len(modes[i])))
            reg = (80, 6)
            nol = np.random.choice(np.arange(2, 8))
            rep_r = (3, 7)
            size_r = (3, 12)
            tex = Thoughts_Texture(modes[i], fund, reg, target, nol, rep_r, size_r)
            section[subdiv]['texture'][v] = tex


# print(t.event_dur_dict[0][1]['texture'][0].phrases)

# print(t.cycle_starts)
# print(t.cycle_ends)
# print(t.event_map)
# print(t.subdivs)
# print()
# print(t.event_dur_dict)




for i in range(noc):
    cycle = t.event_map[i]
    for cycle_time in cycle.keys():
        obj = cycle[cycle_time]
        mode = obj['mode']
        var = obj['variation']
        subdivs = t.subdivs[mode][i]
        obj['texture'] = t.event_dur_dict[mode][subdivs]['texture']

# print(t.event_map)

cycle_event_map = {}
for c in range(noc):
    cycle_event_map[c] = {}
    for m in range(len(modes)):
        cycle_event_map[c][m] = {}
        for s in range(int(t.subdivs[m][c])):
            tex = t.event_map[c]


# print(t.event_map)
# text = t.event_map[[0][1]]
# print(t.event_dur_dict[0][1]['texture'].phrases)
        # tex = Thoughts_Texture()
        # t.event_dur_dict[section][subdiv] =


variations_0 = np.array([i[0] for i in variations])
variations_1 = np.array([i[1] for i in variations])
json.dump([modes, variations_0, variations_1], open('JSON/modes_and_variations.JSON', 'w'), cls=h_tools.NpEncoder)
#


        # melody_index = npi.indices(A, melody_note)
        # print(melody_index)

    # print(abc)
    #
    # print(ab)
    # print(bc)
    # print(ac)
    # print('\n\n\n')

#
# h_tools.plot_basic_hsl(A, 'test_plots/A')
# h_tools.plot_basic_hsl(B, 'test_plots/B')
# h_tools.plot_basic_hsl(C, 'test_plots/C')
# print(A, '\n\n', B, '\n\n', C, '\n\n')
#
# a = h_tools.fix_collection(A)
# print(a)
