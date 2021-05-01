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
    def __init__(self, mode, fund, register, target_index, num_of_lines, repetition_range):
        self.mode = np.sort(mode)
        self.fund = fund
        self.reg_min = register[0]
        self.reg_octs = register[1]
        self.target_index = target_index
        self.nol = num_of_lines
        self.rep_range = repetition_range
        self.target_time = 0.9


    def min_pc_index(self):
        freqs = self.fund * self.mode
        while np.any(freqs > 2 * fund):
            freqs = np.where(freqs > 2*fund, freqs / 2, freqs)
        return np.argmin(freqs)

    def max_pc_index(self):
        freqs = self.fund * self.mode
        while np.any(freqs > 2 * fund):
            freqs = np.where(freqs > 2 * fund, freqs / 2, freqs)
        return np.argmax(freqs)

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

        # melody
        # run a dc alg backwards, starting from target, then flip
        counts = np.zeros(len(self.mode), dtype=int) + 2
        counts[self.target_index] = 1
        dc_seq = h_tools.dc_alg(len(self.mode), size, counts, alpha)
        seq = np.append([self.target_index], dc_seq)[::-1]
        print(seq)
        contour = Contour.nearest(seq)
        register = np.sum(contour)
        line = [0]
        for d in contour:
            line.append(line[-1] + d)
        line = np.array(line)
        print(contour)
        print(line)
        line -= line[-1]
        line += (self.target_index - self.min_pc_index()) % len(self.mode)
        while np.any(line) < 0:
            line += len(mode)
        line_max = np.max(line)
        min_oct = np.ceil(line_max / len(self.mode))
        choice_oct = np.random.choice(np.arange(self.reg_octs-min_oct))
        line = choice_oct * len(mode) + line
        


        print(line)
        print(register)

        print(-register + (self.target_index - self.min_pc_index()) % len(self.mode))



        return

mode = [1.25, 1.35, 1.45, 2.1, 1.6, 1.5, 1.4]
fund = 200
register = (150, 6)
target_index = 1
num_of_lines = 4
repetition_range = (3, 8)
t = Thoughts_Texture(mode, fund, register, target_index, num_of_lines,repetition_range)
out = t.make_phrase(10, 3)
# print(t.max_pc_index())
# print(out)
# print(sum(out))
# arr = np.array([4, 6, 1, 5, 1])
# Contour.nearest(arr)
