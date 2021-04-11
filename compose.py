from tempo_curve import Time
from rhythm_tools import rhythmic_sequence_maker as rsm
import numpy as np
from mode_generation import make_mode_sequence
import json
from harmony_tools import utils as h_tools

noc = 8
dur_tot = 180
modes = make_mode_sequence((20, 30))

events_per_cycle = len(modes)
t = Time(dur_tot=dur_tot, f=0.3, noc=noc)
cycle_events = rsm(events_per_cycle, 15, start_times=True)
cycle_events = np.tile(cycle_events, noc) + np.repeat(np.arange(noc), events_per_cycle)
time_events = []
for i in cycle_events:
    time_events.append(t.real_time_from_cycles(i))
time_events = np.array(time_events)
event_durations = np.ediff1d(time_events)

json.dump(event_durations, open('JSON/event_durs.JSON', 'w'), cls=h_tools.NpEncoder)
json.dump(modes, open('JSON/modes.JSON', 'w'), cls=h_tools.NpEncoder)
