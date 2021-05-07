import json
from harmony_tools import utils as h_tools
import numpy as np
import numpy_indexed as npi
modes = json.load(open('JSON/modes_and_variations.JSON', 'rb'))[0]


def make_triads(mode, num_of_triads, fund=100, min=150, alpha=3, min_ratio=1.5):
    unq_lens = 0
    mode = np.array(mode)
    while np.any(unq_lens != 3):
        seq = h_tools.dc_alg(len(mode), 3 * num_of_triads, alpha=alpha)
        triads = np.array(np.split(seq, num_of_triads))
        unq_lens = np.array([len(set(i)) for i in triads])
    freqs = mode[triads] * fund
    freqs = np.sort(freqs)
    freqs = np.where(freqs < min,
        freqs * (2 ** np.ceil(np.log2(min/freqs))), freqs)
    freqs = np.where(freqs >= 2 * min,
        freqs / (2 ** np.floor(np.log2(freqs/min))), freqs)
    freqs = np.sort(freqs)

    def condition(freq_triad):
        out = np.logical_or(freq_triad[1] / freq_triad[0] < min_ratio,
            freq_triad[2]/ freq_triad[1] < 1.3333)
        return out

    for i in range(len(freqs)):
        init_freqs_i = freqs[i]
        while condition(freqs[i]):
            freqs[i][1] *= 2
            freqs[i] = np.sort(freqs[i])
            if np.any(freqs[i] == np.inf):
                breakpoint()
    return [[i] for i in freqs]
mins = np.linspace(75, 300, 25)
mins = np.append(mins, np.linspace(300, 75, 25))
freqs = [make_triads(i, 50, min=150) for index, i in enumerate(modes)]
json.dump(np.array(freqs), open('JSON/triads.JSON', 'w'), cls=h_tools.NpEncoder)
