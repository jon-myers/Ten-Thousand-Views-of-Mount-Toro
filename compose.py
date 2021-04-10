from tempo_curve import Time
from rhythm_tools import rhythmic_sequence_maker as rsm
import numpy as np
from mode_generation import make_mode_sequence

noc = 12
modes = make_mode_sequence()

events_per_cycle = len(modes)
t = Time(f=0.3, noc=noc)
cycle_events = rsm(events_per_cycle, 10, start_times=True)
cycle_events = np.tile(cycle_events, noc) + np.repeat(np.arange(noc), events_per_cycle)
time_events = []
for i in cycle_events:
    time_events.append(t.real_time_from_cycles(i))
