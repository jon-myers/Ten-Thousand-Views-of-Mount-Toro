import json
from harmony_tools import utils as h_tools
from rhythm_tools import rhythmic_sequence_maker as rsm
from rhythm_tools import spread
import numpy as np
import numpy_indexed as npi
modes = json.load(open('JSON/modes_and_variations.JSON', 'rb'))[0]
from numpy.random import default_rng
rng = default_rng()


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
mins = np.expand_dims(mins, axis=1)
freqs = [make_triads(i, 50, min=mins) for index, i in enumerate(modes)]
freqs = np.concatenate(freqs)
json.dump(np.array(freqs), open('JSON/triads.JSON', 'w'), cls=h_tools.NpEncoder)


Golden = (1 + 5 ** 0.5) / 2
class Pluck:
    """
    irama 0: basic triad (mode 0, 1, 2), low register, repeated with very low
             ncvi (< 3), all three notes in unison (no delays), consistent coef. No
             patterning. avg real time tempo = ~20 bpm. no delays. fixed register.
             soft dynamic. long decay (~8 seconds avg?)
    irama 1: low to middle register, nCVI < 5, triads must share two notes with
             basic triad. No patterning.  avg real time tempo = ~ 32. fixed register.
    irama 2: triads must share one note with basic triad. nCVI < 10. One note
             can be delayed / reverse delayed. Patterning. avg real time tempo =
             ~ 51. moving register. Possibly 1 delayed from other two (or two
             delayed from other one).
    irama 3: all triads are possible. nCVI < 20. Patterning. avg real time tempo
             ~ 81. moving register.
    """
    def __init__(self, irama, real_dur, offsets, mode, fund, rt_since_last):
        self.irama = irama
        mode = np.array(mode)
        self.real_dur = real_dur
        self.offsets = offsets
        self.mode = mode
        self.fund = fund
        self.rt_since_last = rt_since_last # real time since last previous attack
        if np.size(irama) == 2:
            self.transition_point = np.random.uniform(0.25, 0.75)
        self.base_tempo = 20
        self.base_min_freq = 90
        self.base_decay_dur = 8
        self.base_coef = 0.65
        self.floor_nCVI = 0
        self.base_vol = 0.5

    def render(self):
        if np.size(self.irama) == 2:
            print('gotta figure this part out')
        elif self.irama == 0:
            return self.irama_0()
        elif self.irama == 1:
            return self.irama_1()
        elif self.irama == 2:
            return self.irama_2()
        elif self.irama == 3:
            return self.irama_3()


    def irama_0(self):
        """basic triad (mode 0, 1, 2), low register, repeated with very low
           ncvi (< 3), all three notes in unison (no delays), consistent coef. No
           patterning. avg real time tempo = ~20 bpm. no delays. fixed register.
           soft dynamic. long decay (~8 seconds avg?)"""
        tempo_offset = self.offsets[0]
        nCVI_offset = self.offsets[1]
        min_offset = self.offsets[2]
        decay_offset = self.offsets[3]
        coef_offset = self.offsets[4]

        base_tempo = self.base_tempo
        base_min_freq = self.base_min_freq
        base_decay_dur = self.base_decay_dur
        base_coef = self.base_coef
        floor_nCVI = self.floor_nCVI
        base_vol = self.base_vol

        tempo = self.do_offset(base_tempo, tempo_offset)
        avg_dur = 60 / tempo
        num_of_events = np.round(self.real_dur / avg_dur).astype(int)
        if num_of_events == 0:
            print('num_of_events would have been none!')
            num_of_events = 1
        nCVI = floor_nCVI + 3 * nCVI_offset
        durs, event_locs = rsm(num_of_events, nCVI, start_times='both')
        start_offset = (1 / num_of_events) - self.rt_since_last / self.real_dur
        event_locs += start_offset
        event_locs = np.append((0), event_locs)
        event_locs = event_locs[np.nonzero(event_locs < 1)]
        durs = np.append((start_offset), durs)[:event_locs.size]
        triad = self.mode[:3] * self.fund
        min_freq = base_min_freq * 2 ** ((min_offset - 0.5))
        freqs = self.registrate(triad, min_freq)
        freqs = [freqs for i in range(len(event_locs) - 1)]
        freqs.insert(0, 'Rest()')
        delays = [0, 0, 0]
        decay = base_decay_dur * (2 ** (decay_offset - 1))
        coef = base_coef * (2 ** (coef_offset - 1))
        if coef >= 1: coef = 0.999 # prob don't need this, but just in case
        self.packets = []
        for i in range(len(event_locs)):
            packet = {}
            packet['freqs'] = freqs[i]
            packet['dur'] = durs[i]
            packet['coef'] = coef
            packet['decay'] = decay
            packet['delays'] = delays
            packet['vol'] = base_vol
            self.packets.append(packet)
        return self.packets

    def irama_1(self):
        """irama 1: low to middle register, nCVI < 5, triads must share two
        notes with basic triad. No patterning.  avg real time tempo = ~32. fixed
        register. No delays."""
        tempo_offset = self.offsets[0]
        nCVI_offset = self.offsets[1]
        min_offset = self.offsets[2]
        decay_offset = self.offsets[3]
        coef_offset = self.offsets[4]
        vol_offset = self.offsets[5]

        base_tempo = self.base_tempo * Golden
        base_min_freq = self.base_min_freq * Golden
        base_decay_dur = self.base_decay_dur / Golden
        base_coef = self.base_coef / Golden
        bass_nCVI = Golden ** 2 # then 4rd, 6th,
        base_vol = self.base_vol

        tempo = self.do_offset(base_tempo, tempo_offset)
        avg_dur = 60 / tempo
        num_of_events = np.round(self.real_dur / avg_dur).astype(int)
        if num_of_events == 0:
            print('num_of_events would have been none!')
            num_of_events = 1
        nCVI = self.do_offset(bass_nCVI, nCVI_offset)
        durs, event_locs = rsm(num_of_events, nCVI, start_times='both')
        start_offset = (1 / num_of_events) - self.rt_since_last / self.real_dur
        # breakpoint()
        event_locs += start_offset
        event_locs = np.append((0), event_locs)
        event_locs = event_locs[np.nonzero(event_locs < 1)]
        durs = np.append((start_offset), durs)[:event_locs.size]

        triads = self.get_triads(len(event_locs), shared=2)
        # triad = self.get_triad(shared=2) * self.fund
        min_freq = base_min_freq * 2 ** ((min_offset - 0.5))
        freqs = [self.registrate(triad, min_freq) for triad in triads]
        freqs.insert(0, 'Rest()')
        delays = [0, 0, 0]
        decay = base_decay_dur * (2 ** (decay_offset - 1))
        coef = base_coef * (2 ** (coef_offset - 1))
        if coef >= 1: coef = 0.999 # prob don't need this, but just in case
        vol = base_vol * 2 ** ((vol_offset - 0.5) / 2)
        self.packets = []
        for i in range(len(event_locs)):
            packet = {}
            packet['freqs'] = freqs[i]
            packet['dur'] = durs[i]
            packet['coef'] = coef
            packet['decay'] = decay
            packet['delays'] = delays
            packet['vol'] = vol
            self.packets.append(packet)
        return self.packets


    def irama_2(self):
        """triads must share one note with basic triad. nCVI < 10. One note can 
        be delayed / reverse delayed. Patterning. avg real time tempo = ~51. 
        moving register. Possibly 1 delayed from other two (or two delayed from 
        other one)."""
        
        tempo_offset = self.offsets[0]
        nCVI_offset = self.offsets[1]
        min_offset = self.offsets[2]
        decay_offset = self.offsets[3]
        coef_offset = self.offsets[4]
        vol_offset = self.offsets[5]
        
        base_tempo = self.base_tempo * (Golden ** 2)
        base_min_freq = self.base_min_freq * (Golden ** 2)
        base_decay_dur = self.base_decay_dur / (Golden **2)
        base_coef = self.base_coef / (Golden ** 2)
        bass_nCVI = Golden ** 4 # then 6th,
        base_vol = self.base_vol
        
        tempo = self.do_offset(base_tempo, tempo_offset)
        avg_dur = 60 / tempo
        nCVI = self.do_offset(bass_nCVI, nCVI_offset)
        
        patterning_event_min = 3
        num_of_events = np.round(self.real_dur / avg_dur).astype(int)
        if num_of_events == 0: 
            print('num_of_events would have been none!')
            num_of_events = 1
            
        max_num_of_reps = np.floor(num_of_events / patterning_event_min)
        if max_num_of_reps < 2:
            num_of_reps = 1
        else:
            num_of_reps = rng.choice(np.arange(1, max_num_of_reps))
        num_per_rep = np.floor(num_of_events / num_of_reps)
        
        rep_durs = rsm(num_per_rep, nCVI)
        rep_durs = [spread(rep_durs) for i in range(num_of_reps)]
        
         
        
        
        
        
        
        
        
        
    
    def get_delays(self, groups=2, max_del=0.2):
        """Returns array with delay times that specify how much after event time
        the three notes of the triad will be struck. Always at least one at zero
        time. If groups == 2, (at least) two of three must occur simultaenously.
        If groups == 3, all can occur seperate, (a 'strum')."""
        max_del = float(max_del)
        delays = np.zeros(3)
        if groups == 2:
            del_times = np.random.random() * max_del
            num_dels = np.random.randint(2) + 1
            idxs = rng.choice(np.arange(3), size=num_dels, replace=False)            
        elif groups == 3:
            del_times = np.random.random(2) * max_del
            idxs = rng.choice(np.arange(3), size=2, replace=False)
        delays[idxs] = del_times
        return delays
            
        


    def registrate(self, chord, min):
        freqs = np.sort(chord)
        freqs = np.where(freqs < min,
            freqs * (2 ** np.ceil(np.log2(min/freqs))), freqs)
        freqs = np.where(freqs >= 2 * min,
            freqs / (2 ** np.floor(np.log2(freqs/min))), freqs)
        freqs = np.sort(freqs)

        def condition(freq_triad):
            out = np.logical_or(freq_triad[1] / freq_triad[0] < 1.5,
                freq_triad[2]/ freq_triad[1] < 1.3333)
            return out

        while condition(freqs):
            freqs[1] *= 2
            freqs = np.sort(freqs)

        return freqs

    def get_triad(self, shared=3):
        """Grab a single particular triad"""
        base_triad = self.mode[:3]
        out = base_triad[rng.choice(np.arange(3), size=shared, replace=False)]
        while np.size(out) < 3:
            ad = rng.choice(np.arange(3, np.size(self.mode)), size=3-np.size(out))
            out = np.append(out, self.mode[ad])
        return np.sort(out)

    def get_triads(self, num_of_triads, shared=3):
        """Generate a list of triads. """
        triads = np.array([self.get_triad(shared) for i in range(num_of_triads)])
        return triads
        
    def do_offset(self, base_val, offset):
        """Spreads value to limit of (x ** -0.5, x ** 0.5) according to offset. 
        offset must be betwen 0 and 1. """
        return base_val * (2 ** (offset - 0.5))


p = Pluck(1, 10, np.random.uniform(0, 1, size=6), modes[0], 150, 1.0)
print(p.render())
