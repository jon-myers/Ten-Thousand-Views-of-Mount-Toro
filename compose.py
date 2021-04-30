from tempo_curve import Time
from rhythm_tools import rhythmic_sequence_maker as rsm
import numpy as np
from mode_generation import make_mode_sequence
import json
from harmony_tools import utils as h_tools
import numpy_indexed as npi

noc = 8
dur_tot = 29*60
modes, variations = make_mode_sequence((20, 30))

events_per_cycle = len(modes)
t = Time(dur_tot=dur_tot, f=0.3, noc=noc)
t.set_cycle(len(modes))
# print(len(modes))
# print(t.event_map[0])
# print('\n\n')
# print(t.event_map[noc-1])
# for i in t.event_map.keys():
#     print(i)

for key, value in t.real_time_event_map.items():
    print(key, ' : ', value)
# print(t.real_time_event_map)
# )
#
# cycle_events = rsm(events_per_cycle, 15, start_times=True)
# cycle_events = np.tile(cycle_events, noc) + np.repeat(np.arange(noc), events_per_cycle)
# time_events = []
# for i in cycle_events:
#     time_events.append(t.real_time_from_cycles(i))
# time_events = np.array(time_events)
# event_durations = np.ediff1d(time_events)
#
# json.dump(event_durations, open('JSON/event_durs.JSON', 'w'), cls=h_tools.NpEncoder)
# json.dump(modes, open('JSON/modes.JSON', 'w'), cls=h_tools.NpEncoder)
#

variations_0 = np.array([i[0] for i in variations])
variations_1 = np.array([i[1] for i in variations])
json.dump([modes, variations_0, variations_1], open('JSON/modes_and_variations.JSON', 'w'), cls=h_tools.NpEncoder)

melody = []
for i in range(len(modes)):
    A = h_tools.gen_ratios_to_hsv(modes[i]/modes[i][0], [3, 5, 7])
    B = h_tools.gen_ratios_to_hsv(variations[i][0]/modes[i][0], [3, 5, 7])
    C = h_tools.gen_ratios_to_hsv(variations[i][1]/modes[i][0], [3, 5, 7])
    
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
print(melody)
        
        
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
