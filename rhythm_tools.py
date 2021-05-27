import numpy as np
import itertools, math
import pretty_midi
from scipy.integrate import quad as integrate
from scipy.misc import derivative
from scipy.special import gammainc as gamma
from scipy.optimize import fsolve
import math
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
            return fsolve(lambda x: self.b(x) - beat, 0.00001)[0]

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
        self.cycle_durs, self.cycle_starts = rhythmic_sequence_maker(nos, nCVI,
                                                             start_times='both')
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
                    starts = rhythmic_sequence_maker(j, nCVI, start_times=True)
                    ends = np.append(starts[1:], 1)
                    bounds = [(starts[i], ends[i]) for i in range(len(starts))]

                seq = sequence_from_sample(samples, bounds)
                dict = {'starts': starts, 'sequence': seq}
                self.event_dur_dict[i][j] = dict


            for j in range(self.noc):
                dur = self.real_time_dur_from_cycle_event(j, i)
                upper_lim = np.floor(dur / min_dur)
                if upper_lim > 1:
                    subdivs = np.random.choice(np.arange(1, np.floor(dur / min_dur)))
                else:
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

def nPVI(d):
    m = len(d)
    return 100 / (m - 1) * sum([abs((d[i] - d[(i + 1)]) / ((d[i] + d[(i + 1)]) / 2)) for i in range(m - 1)])


def nCVI(d):
    matrix = [list(i) for i in itertools.combinations(d, 2)]
    matrix = [nPVI(i) for i in matrix]
    return sum(matrix) / len(matrix)

def faster_nCVI(d):
    """Turns out that this is actually much slower, by a factor of ~ 3 or 4. Damn."""
    m = len(d)
    n = np.math.factorial(m) / (np.math.factorial(m-2) * 2)
    ij_pairs = [[(i, j) for j in range(i+1, m)] for i in range(0, m-1)]
    ij_pairs = np.concatenate(ij_pairs)
    diff = (lambda x: np.abs(2 * (d[x[0]] - d[x[1]]) / (d[x[0]] + d[x[1]])))
    deltas = map(diff, ij_pairs)
    sums = list(map(diff, ij_pairs))
    return 100*np.sum(list(map(diff, ij_pairs))) / n


def rhythmic_sequence_maker(num_of_events, nCVI_average, factor=2.0, start_times=False):
    """aka 'rsm'. """
    num_of_events = int(num_of_events)
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

# def phrase_compiler(notes_per_seg, dur_per_seg, nCVI):
#     full_seq = np.array([])
#     for seg in range(len(notes_per_seg)):
#         noe = notes_per_seg[seg]
#         sequence = rhythmic_sequence_maker(noe, nCVI) * dur_per_seg[seg]
#         full_seq = np.concatenate((full_seq, sequence))
#     return full_seq

class Density_Curve:

    def __init__(self, durs, edges):

        """durs should add to one"""
        self.durs = durs
        self.edges = edges
        self.starts = np.concatenate(([0], np.cumsum(self.durs)[:-1]))

    def value(self, x):
        seg = np.where(x >= self.starts)[0][-1]
        prop = (x - self.starts[seg]) / self.durs[seg]
        mu2 = (1 - np.cos(prop * np.pi)) / 2
        out = self.edges[seg] * (1 - mu2) + self.edges[seg+1] * mu2
        return out

    def area(self, x0, x1):
        """area under curve"""
        return integrate(self.value, x0, x1)[0]

def phrase_compiler(dc_durs, dc_edges, nos, frame_nCVI, nCVI):
    dur_tot = np.sum(dc_durs)
    dc = Density_Curve(dc_durs, dc_edges)
    durs = rhythmic_sequence_maker(nos, frame_nCVI) * dur_tot
    starts = np.concatenate(([0], np.cumsum(durs)[:-1]))
    sizes = [round(dc.area(starts[i], starts[i]+durs[i])) for i in range(nos)]
    segs = np.array([])
    for s in range(nos):
        rs = rhythmic_sequence_maker(sizes[s], nCVI) * durs[s]
        segs = np.concatenate((segs, rs))
    return segs




# dc = Density_Curve([4, 8], [2, 13, 5])
# vals = np.linspace(0, 1, 100)
# vals = [dc.value(i) for i in vals]
# area = dc.area_under_curve(0, 3)[0]
# print(area)
# phrase = phrase_compiler([5, 5, 5], [2, 13, 2, 6], 20, 30, 50)
#
# from matplotlib import pyplot as plt
# real_cumsum = np.concatenate(([0], np.cumsum(phrase)[:-1]))
# plt.bar(real_cumsum, phrase)
#
# plt.show()


# fs = phrase_compiler([10, 10, 10, 10, 10], [4, 3, 2, 3, 4], 50)
# plt.bar(np.arange(len(fs)), fs)
# plt.show()
# print(fs)


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


def spread(init, max_ratio, scale='log', func=None):
    exponent = np.clip(np.random.normal() / 3, -1, 1)
    if scale == 'log':
        return init * (max_ratio ** exponent)
    elif scale == 'linear':
        out = init + exponent * max_ratio
        if func != None and np.abs(init-out) > max_ratio:
            
            func()
        return out



# print(spread(0.5, 2, scale='linear'))
def normalize(array):
    array = np.array(array)
    return array / sum(array)

def jiggle_sequence(sequence, spd):
    return normalize(np.array([spread(i, spd) for i in sequence]))
