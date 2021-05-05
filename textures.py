from harmony_tools import utils as h_tools
from rhythm_tools import rhythmic_sequence_maker as rsm
import rhythm_tools as r_tools
import numpy as np

class Contour:

    def nearest(sequence, lim=7):
        diff = np.diff(sequence)
        condition = np.abs(diff) > lim/2
        def flip(x): return -1 * np.sign(x) * (lim - np.abs(x))
        diff = np.where(condition, flip(diff), diff)
        return diff


class Thoughts_Texture:


    """Makes textures akin to my percussion piece, Thoughts, in which a
    number of different phrases are made, which each land on a common target
    pitch. These phrases are made to repeat a number of times, landing all
    together on the target.

    mode: array [float, float, ...], ratios of pitches.
    fund: float, fundamental for entire piece.
    register: tuple (float, int), min frequency and num of octaves above that min.
    target_index: integer, index of mode pitch that is the target of the
        texture
    num_of_lines: integer, total number of phrases
    repetition_range: tuple (int, int), min and max possible reps per phrase

    """
    def __init__(self, mode, fund, register, target_index, num_of_lines, 
            repetition_range, size_range):
        self.mode = np.sort(mode)
        self.fund = fund
        self.reg_min = register[0]
        self.reg_octs = register[1]
        self.reg_max = self.reg_min * (2 ** self.reg_octs)
        self.target_index = target_index
        self.nol = num_of_lines
        self.rep_range = repetition_range
        self.size_range = size_range
        self.target_time = 0.95
        self.max_start_offset = 0.2
        self.zero_freq = self.fund * self.mode[0]
        while self.zero_freq < self.reg_min:
            self.zero_freq *= 2
        while self.zero_freq >= 2 * self.reg_min:
            self.zero_freq /= 2
        if self.min_pc_index() != 0:
            self.zero_freq /= 2
        self.make_phrases()

    def min_pc_index(self):
        freqs = self.fund * self.mode
        while np.any(freqs >= 2 * self.reg_min):
            freqs = np.where(freqs >= 2*self.reg_min, freqs / 2, freqs)
        return np.argmin(freqs)

    def max_pc_index(self):
        freqs = self.fund * self.mode
        while np.any(freqs > 2 * fund):
            freqs = np.where(freqs > 2 * fund, freqs / 2, freqs)
        return np.argmax(freqs)
    
    def pitch_to_freq(self, pitch_arr):
        oct_shift = 2 ** np.floor(pitch_arr/len(self.mode))
        out = self.mode[pitch_arr%len(self.mode)] * oct_shift * self.zero_freq
        return out

    def make_phrase(self, size, reps, nCVI=7, alpha=2):

        # rhythm
        durs = rsm(size, nCVI)
        div = self.target_time / (reps - 1 + sum(durs[:-1]))
        aggregated_durs = []
        for i in range(reps-1):
            rep = r_tools.jiggle_sequence(durs, 1.3) * div
            aggregated_durs.extend(rep)
        last_rep = r_tools.jiggle_sequence(durs[:-1], 1.3) * div * sum(durs[:-1])
        aggregated_durs.extend(last_rep)
        aggregated_durs = np.array(aggregated_durs)
        # this normalization in the next line is not strictly necessary, fixes
        # (likly insignificant) sum errors, in ~ the 10th significant digit.
        aggregated_durs *= self.target_time / sum(aggregated_durs)
        offset = np.random.uniform(0, self.max_start_offset)
        aggregated_durs *= (self.target_time - offset) / self.target_time
        starts = np.append(np.array((0)), np.cumsum(aggregated_durs)[:-1]) + offset
        starts = np.append(starts, self.target_time)
        starts = np.append((0), starts)

        
        # melody
        # run a dc alg backwards, starting from target, then flip
        counts = np.zeros(len(self.mode), dtype=int) + 2
        counts[self.target_index] = 1
        dc_seq = h_tools.dc_alg(len(self.mode), size-1, counts, alpha)
        seq = np.append([self.target_index], dc_seq)[::-1]
        # print(seq)
        contour = Contour.nearest(seq)
        register = np.sum(contour)
        line = [0]
        for d in contour:
            line.append(line[-1] + d)
        line = np.array(line)
        # print(contour)
        # print(line)
        line -= line[-1]
        line += self.target_index
        while np.min(line) < self.min_pc_index():
            line += len(self.mode)
            
        lowest_freqs = self.pitch_to_freq(line)
        possible_octs_above = 0
        while np.max((2 ** (possible_octs_above)) * lowest_freqs) <= self.reg_max:
            possible_octs_above += 1
        octs_above = np.random.choice(np.arange(possible_octs_above))
        freqs = lowest_freqs * (2 ** octs_above)
        freqs = np.tile(freqs, reps)
        freqs = np.append((0), freqs)
        
        # dynamics
        dynamics = np.random.uniform(0.5, 1.0, size)
        dynamics = np.tile(dynamics, reps)  
        dynamics = np.append((0), dynamics)
            
        phrase = {'freqs': freqs, 'starts': starts, 'dynamics': dynamics}
        return phrase
        
    def make_phrases(self):
        all_sizes = np.random.randint(*self.size_range, self.nol)
        all_reps = np.random.randint(*self.rep_range, self.nol)
        all_nCVIs = np.random.uniform(2, 15, self.nol)
        self.phrases = []
        for i in range(self.nol):
            phrase = self.make_phrase(all_sizes[i], all_reps[i], all_nCVIs[i])
            self.phrases.append(phrase)
        
        

# mode = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
# fund = 200
# register = (60, 6)
# target_index = 1
# num_of_lines = 4
# repetition_range = (3, 8)
# size_range = (5, 15)
# t = Thoughts_Texture(mode, fund, register, target_index, num_of_lines,
#     repetition_range, size_range)
# print(t.phrases)
