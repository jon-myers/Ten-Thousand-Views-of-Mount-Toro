from scipy.integrate import quad as integrate
from scipy.misc import derivative
from scipy.special import gammainc as gamma
from scipy.optimize import fsolve
import math
from rhythm_tools import rhythmic_sequence_maker as rsm
from rhythm_tools import sequence_from_sample
import numpy as np


# the time variable refers to the initial time frame. To refer to the actual
# time in the final piece, as stretched to end at dur_tot, I will use `real_time`
class Time:
    """Creates a time object, for a looping cycle (or buffer) with a
    variable speed, fixed accelration, which can be queried to, for example,
    get the phase, tempo-level, and irama at a particular moment in real time.
    """

    def __init__(self, irama_levels=4, dur_tot=1740, z=0.5, f=0.5, noc=10):
        self.dur_tot = dur_tot
        self.z = z
        self.f = f
        self.end_time = self.time_from_tempo(2 ** -irama_levels)
        self.end_beats = self.b(self.end_time)
        self.norm_factor = self.dur_tot / self.end_time
        self.noc = noc #number of cycles

    def mm(self, time):
        """Instantaneous tempo at a given time measured from begining of the
        piece."""
        return self.z ** (time ** self.f)

    # def b_alt(self, time):
    #     """Number of beats passed since beginning of piece. Anayltic, rather
    #     than numeric. (in order to get this to match b, I had to remove a
    #     negative sign from the answer provided by integral-calculator.com !?)"""
    #     part = -math.log(self.z) * (time ** self.f)
    #     return gamma(1/self.f, part) * time / (self.f * (part ** (1 / self.f)))

    def b(self, time):
        """Number of beats passed since beginning of piece."""
        return integrate(self.mm, 0, time)[0]

    def acc(self, time):
        """Instantaneous acceleration at a given time, measured from the
        beginning of the piece."""
        if time == 0:
            return -math.inf
        else:
            return self.mm(time) * math.log(self.z) * self.f * (time ** (self.f - 1))

    def time_from_tempo(self, tempo):
        """Returns the time when a given instantaneous tempo occurs."""
        return math.e ** (math.log(math.log(tempo, self.z)) / self.f)

    def time_from_beat(self, beat):
        """Returns the time when a given number of beats have elapsed."""
        if beat == 0:
            return 0
        else:
            return fsolve(lambda x: self.b(x) - beat, 0.01)[0]

    def cycles_from_beats(self, beat):
        """Converts a given location in beats to the location in cycles."""
        return beat * self.noc / self.end_beats

    def beats_from_cycles(self, cycles):
        """Converts a given location in cycles to the location in beats."""
        return cycles * self.end_beats / self.noc

    def cycles_from_time(self, time):
        """Converts a given location in time to the location in cycles."""
        return self.cycles_from_beats(self.b(time))

    def time_from_cycles(self, cycles):
        """Converts a given location in cycles to the location in time."""
        beats = self.beats_from_cycles(cycles)
        time = self.time_from_beat(beats)
        return time

    def real_time_from_time(self, time):
        """Converts from time to literal time of the piece as realized."""
        return time * self.norm_factor

    def time_from_real_time(self, real_time):
        """Converts from literal time of the piece as realized to abstract time.
        """
        return real_time / self.norm_factor

    def real_time_from_beats(self, beats):
        """Converts from beats to literal time of the piece as realized."""
        time = self.time_from_beat(beats)
        return self.real_time_from_time(time)

    def real_time_from_cycles(self, cycles):
        """Converts a given location in cycles to literal time of the piece as
        realized"""
        beats = self.beats_from_cycles(cycles)
        return self.real_time_from_beats(beats)

    def mm_from_cycles(self, cycles):
        """Converts a given location in cycles to tempo at that moment."""
        time = self.time_from_cycles(cycles)
        mm = self.mm(time)
        return mm

    def cycles_from_real_time(self, real_time):
        """Given a real moment in the time of the realized piece, returns the
        number of cycles that have elapsed."""
        time = self.time_from_real_time(real_time)
        return self.cycles_from_time(time)

    def set_cycle(self, nos, nCVI=7):
        """Allows you to assign a set of numbers between 0 and 1, the start
        times of the sections within a cycle. Probably generated from `rhythmic
        sequence maker`, with `start_times` set to True."""
        self.nos = nos
        self.cycle_durs, self.cycle_starts = rsm(nos, nCVI, start_times='both')
        self.cycle_ends = np.append(self.cycle_starts[1:], [1])
        min_dur = 7
        max_subdivs = 5
        self.event_dur_dict = {}
        self.event_map = {}
        self.subdivs = {}
        for c in range(self.noc):
            self.event_map[c] = {}
        for i in range(nos):
            # Set the subdivisional pattern for various different subdivs

            sample_a = np.random.uniform(size=150)
            sample_a = np.append(sample_a, np.random.uniform(0, 0.01, size=100))
            sample_b = np.random.uniform(size=150)
            sample_c = np.random.uniform(size=150)
            samples = (sample_a, sample_b, sample_c)

            self.event_dur_dict[i] = {}
            self.subdivs[i] = {}
            for j in range(1, max_subdivs + 1):
                if j == 1:
                    starts = [0]
                    bounds = [(0, 1)]
                else:
                    starts = rsm(j, nCVI, start_times=True)
                    ends = np.append(starts[1:], 1)
                    bounds = [(starts[i], ends[i]) for i in range(len(starts))]

                seq = sequence_from_sample(samples, bounds)
                dict = {'starts': starts, 'sequence': seq}
                self.event_dur_dict[i][j] = dict


            for j in range(self.noc):
                dur = self.real_time_dur_from_cycle_event(j, i)
                subdivs = np.floor(dur / min_dur)
                if subdivs == 0:
                    subdivs = 1
                if subdivs > max_subdivs:
                    subdivs = max_subdivs
                self.subdivs[i][j] = subdivs
                starts = self.event_dur_dict[i][subdivs]['starts']
                seq = self.event_dur_dict[i][subdivs]['sequence']
                for k in range(len(starts)):
                    ct_start = self.cycle_starts[i] + (self.cycle_durs[i] * starts[k])
                    self.event_map[j][ct_start] = {'mode': i, 'variation': seq[k]}
        self.real_time_event_map = {}
        for cycle in self.event_map.keys():
            for event in self.event_map[cycle].keys():
                real_time = self.real_time_from_cycles(cycle+event)
                self.real_time_event_map[real_time] = self.event_map[cycle][event]


            #



            # for cycle in range(self.noc):
            #     dur = self.real_time_dur_from_cycle_event(cycle, i)
            #     subdivs = np.floor(dur / min_dur)
            #     if subdivs == 0:
            #         suvdivs = 1
            #     if subdivs > max_subdivs:
            #         subiivs = max_subdivs
            #         starts = rsm(subdivs)
            # event_dur_dict[i] =

    def real_time_dur_from_cycle_event(self, cycle_num, event_num):
        start = self.cycle_starts[event_num] + cycle_num
        if event_num+1 == len(self.cycle_starts):
            end = 1 + cycle_num
        else:
            end = self.cycle_starts[event_num+1] + cycle_num
        rt_start = self.real_time_from_cycles(start)
        rt_end = self.real_time_from_cycles(end)
        dur = rt_end - rt_start
        return dur
# test_cycle = rsm(10, 10, start_times=True)
# print(test_cycle)

# t = Time(f=0.3, noc=9)
# t.set_cycle(7)
# print(t.event_map[8])
# for i in range(10):
#     print(t.real_time_dur_from_cycle_event(i, 2))
# print()
# for i in range(6):
#     print(t.real_time_dur_from_cycle_event(8, i))

# real_times = [t.real_time_from_cycles(i) for i in test_cycle]
# print(real_times)
# out = t.time_from_beat(0.041)
# # print(t.end)
# # print(t.end_beats)
# # print(t.real_time_from_cycles(1))
# cycle_durs = rsm(20, 10)
# cumsum = np.cumsum(cycle_durs)[:-1]
# cycle_events = np.insert(cumsum, 0, 0)
#
# real_times = [t.real_time_from_cycles(i) for i in cycle_events]
# print(real_times)
# print(cycle_events)
