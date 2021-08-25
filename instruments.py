import json
from harmony_tools import utils as h_tools
from rhythm_tools import rhythmic_sequence_maker as rsm
from rhythm_tools import jiggle_sequence, spread, phrase_compiler, nCVI
import numpy as np
import numpy_indexed as npi
modes = json.load(open('JSON/modes_and_variations.JSON', 'rb'))[0]
from numpy.random import default_rng
import itertools
from mode_generation import Note_Stream, get_sub_mode
from markov import make_pluck_phrase, generate_transition_table, make_multi_changing_pluck_phrase, closest_index, make_pitch_pairs
rng = default_rng()
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
    def __init__(self, irama, real_dur, offsets, mode, fund, rt_since_last,
                 next_offsets=None):
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
            if self.irama[0] == 0:
                A = self.irama_0()
                B = self.irama_1()
            elif self.irama[0] == 1:
                A = self.irama_1()
                B = self.irama_2()
            elif self.irama[0] == 2:
                A = self.irama_2()
                B = self.irama_3()
            elif self.irama[0] == 3:
                return self.irama_3()
            A = [i for i in A if i['start'] < self.transition_point]
            B = [i for i in B if i['start'] >= self.transition_point]
            return A + B

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
        freq_offset = self.offsets[2]
        decay_offset = self.offsets[3]
        coef_offset = self.offsets[4]

        base_tempo = self.base_tempo
        base_min_freq = self.base_min_freq
        base_decay_dur = self.base_decay_dur / (Golden ** 2.5)
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
        if start_offset < 0: start_offset = 0
        event_locs += start_offset
        event_locs = np.append((0), event_locs)
        event_locs = event_locs[np.nonzero(event_locs < 1)]
        durs = np.append((start_offset), durs)[:event_locs.size]
        triad = self.mode[:3] * self.fund
        min_freq = base_min_freq * 2 ** ((freq_offset - 0.5))
        freqs = self.registrate(triad, min_freq)
        freqs = [freqs for i in range(len(event_locs) - 1)]
        freqs.insert(0, 'Rest()')
        delays = [0, 0, 0]
        decay = base_decay_dur * (2 ** (decay_offset - 1))
        coef = base_coef * (2 ** (coef_offset - 1))
        if coef >= 1: coef = 0.999 # prob don't need this, but just in case
        self.packets = []
        for i in range(len(event_locs)):
            vol_ = [spread(base_vol, 1.5, scale='linear') for i in range(3)]
            packet = {}
            packet['freqs'] = freqs[i]
            packet['dur'] = durs[i]
            packet['coef'] = coef
            packet['decay'] = decay
            packet['delays'] = delays
            packet['vol'] = vol_
            packet['start'] = event_locs[i]
            if i == len(event_locs) - 1:
                packet['end'] = 1.0
            else:
                packet['end'] = event_locs[i+1]
            self.packets.append(packet)
        return self.packets

    def irama_1(self):
        """irama 1: low to middle register, nCVI < 5, triads must share two
        notes with basic triad. No patterning.  avg real time tempo = ~32. fixed
        register. No delays."""
        tempo_offset = self.offsets[0]
        nCVI_offset = self.offsets[1]
        freq_offset = self.offsets[2]
        decay_offset = self.offsets[3]
        coef_offset = self.offsets[4]
        vol_offset = self.offsets[5]

        base_tempo = self.base_tempo * Golden
        base_min_freq = self.base_min_freq * Golden
        base_decay_dur = self.base_decay_dur / (Golden ** 2)
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
        if start_offset < 0: start_offset = 0
        event_locs += start_offset
        event_locs = np.append((0), event_locs)
        event_locs = event_locs[np.nonzero(event_locs < 1)]
        durs = np.append((start_offset), durs)[:event_locs.size]
        triads = self.get_triads(len(event_locs), shared=2)
        min_freq = base_min_freq * 2 ** ((freq_offset - 0.5))
        freqs = [self.registrate(triad, min_freq) for triad in triads]
        freqs.insert(0, 'Rest()')
        delays = [0, 0, 0]
        decay = base_decay_dur * (2 ** (decay_offset - 1))
        coef = base_coef * (2 ** (coef_offset - 1))
        if coef >= 1: coef = 0.999 # prob don't need this, but just in case
        vol = base_vol + ((vol_offset - 0.5) / 2)
        self.packets = []
        for i in range(len(event_locs)):
            vol_ = [spread(vol, 2, scale='linear') for i in range(3)]
            packet = {}
            packet['freqs'] = freqs[i]
            packet['dur'] = durs[i]
            packet['coef'] = coef
            packet['decay'] = decay
            packet['delays'] = delays
            packet['vol'] = vol_
            packet['start'] = event_locs[i]
            if i == len(event_locs) - 1:
                packet['end'] = 1.0
            else:
                packet['end'] = event_locs[i+1]
            self.packets.append(packet)
        return self.packets


    def irama_2(self):
        """triads must share one note with basic triad. nCVI < 10. One note can
        be delayed / reverse delayed. Patterning. avg real time tempo = ~51.
        moving register. Possibly 1 delayed from other two (or two delayed from
        other one)."""

        tempo_offset = self.offsets[0]
        nCVI_offset = self.offsets[1]
        freq_offset = self.offsets[2]
        decay_offset = self.offsets[3]
        coef_offset = self.offsets[4]
        vol_offset = self.offsets[5]

        base_tempo = self.base_tempo * (Golden ** 2)
        base_min_freq = self.base_min_freq * (Golden ** 2)
        base_decay_dur = self.base_decay_dur / (Golden ** 1.5)
        base_coef = self.base_coef / (Golden ** 2)
        bass_nCVI = Golden ** 4 # then 6th,
        base_vol = self.base_vol

        tempo = self.do_offset(base_tempo, tempo_offset)
        avg_dur = 60 / tempo
        nCVI = self.do_offset(bass_nCVI, nCVI_offset)
        max_gap_size = 1/3 # as compared with a given rep.

        patterning_event_min = 3
        num_of_events = np.round(self.real_dur / avg_dur).astype(int)
        if num_of_events == 0:
            print('num_of_events would have been none!')
            num_of_events = 1

        max_num_of_reps = np.floor(num_of_events / patterning_event_min)
        if max_num_of_reps < 2:
            num_of_reps = 1
        else:
            num_of_reps = rng.choice(np.arange(1, max_num_of_reps)).astype(int)
        num_per_rep = np.floor(num_of_events / num_of_reps).astype(int)
        durs = rsm(num_per_rep, nCVI)

        triads = self.get_triads(len(durs), shared=1)
        min_freq = self.do_offset(base_min_freq, freq_offset)
        freqs = [self.registrate(triad, min_freq) for triad in triads]
        delays = [self.get_delays(np.random.randint(2)+1) for i in range(num_per_rep)]
        coef_center = self.do_offset(base_coef, coef_offset, subtract=1)
        coefs = [spread(coef_center, 2) for i in range(num_per_rep)]
        decay_center = self.do_offset(base_decay_dur, decay_offset, subtract=1)
        decays = [spread(decay_center, 2) for i in range(num_per_rep)]
        vol_center = base_vol + ((vol_offset - 0.5) / 2)
        vols = [spread(vol_center, 3, scale='linear') for i in range(num_per_rep)]
        vols = [[np.clip(spread(i, 2, scale='linear'), 0, 1) for j in range(3)] for i in vols]
        rep_delays = [delays for i in range(num_of_reps)]
        rep_freqs = [freqs for i in range(num_of_reps)]
        rep_coefs = [coefs for i in range(num_of_reps)]
        rep_decays = [decays for i in range(num_of_reps)]
        rep_vols = [vols for i in range(num_of_reps)]
        rep_durs = [jiggle_sequence(durs, 1.2) for i in range(num_of_reps)]
        for gap_index in range(1, num_of_reps)[::-1]:
            gap = np.random.randint(2, dtype=bool)
            if gap:
                gap_size = np.random.rand() * max_gap_size
                rep_durs.insert(gap_index, np.array([gap_size]))
                rep_freqs.insert(gap_index, ['Rest()'])
                rep_delays.insert(gap_index, [[0, 0, 0]])
                rep_coefs.insert(gap_index, [0.1])
                rep_decays.insert(gap_index, [3])
                rep_vols.insert(gap_index, [[0.5, 0.5, 0.5]])

        full_durs = np.concatenate(rep_durs)
        full_durs /= np.sum(full_durs)
        full_freqs = list(itertools.chain.from_iterable(rep_freqs))
        full_delays = np.concatenate(rep_delays)
        full_coefs = np.concatenate(rep_coefs)
        full_decays = np.concatenate(rep_decays)
        full_vols = np.concatenate(rep_vols)

        full_event_locs = np.append((0), np.cumsum(full_durs)[:-1])
        start_offset = (1 / len(full_event_locs)) - self.rt_since_last / self.real_dur
        if start_offset < 0: start_offset = 0
        full_event_locs += start_offset
        full_event_locs = full_event_locs[np.nonzero(full_event_locs < 1)]

        size = np.size(full_event_locs)
        full_freqs.insert(0, 'Rest()')
        full_freqs = full_freqs[:size]
        fd_shape = (len(full_delays) + 1, 3)
        full_delays = np.append([0, 0, 0], full_delays).reshape(fd_shape)[:size]
        full_coefs = np.append(0.1, full_coefs)[:size]
        full_decays = np.append(3, full_decays)[:size]
        v_shape = (len(full_vols) + 1, 3)
        full_vols = np.append([0.5, 0.5, 0.5], full_vols).reshape(v_shape)[:size]

        self.packets = []
        for i in range(len(full_event_locs)):
            packet = {}
            packet['freqs'] = full_freqs[i]
            packet['dur'] = full_durs[i]
            packet['coef'] = full_coefs[i]
            packet['decay'] = full_decays[i]
            packet['delays'] = full_delays[i]
            packet['vol'] = full_vols[i]
            packet['start'] = full_event_locs[i]
            if i == len(full_event_locs) - 1:
                packet['end'] = 1.0
            else:
                packet['end'] = full_event_locs[i+1]
            self.packets.append(packet)
        return self.packets



    def irama_3(self):
        """all triads are possible. nCVI < 20. Patterning. avg real time tempo
                 ~ 81. moving register."""

        tempo_offset = self.offsets[0]
        nCVI_offset = self.offsets[1]
        freq_offset = self.offsets[2]
        decay_offset = self.offsets[3]
        coef_offset = self.offsets[4]
        vol_offset = self.offsets[5]

        base_tempo = self.base_tempo * (Golden ** 3)
        base_min_freq = self.base_min_freq * (Golden ** 3)
        base_decay_dur = self.base_decay_dur / Golden
        base_coef = self.base_coef / (Golden ** 3)
        bass_nCVI = Golden ** 6 # then 6th,
        base_vol = self.base_vol

        tempo = self.do_offset(base_tempo, tempo_offset)
        avg_dur = 60 / tempo
        nCVI = self.do_offset(bass_nCVI, nCVI_offset)
        max_gap_size = 2/5 # as compared with a given rep.

        patterning_event_min = 6
        num_of_events = np.round(self.real_dur / avg_dur).astype(int)
        if num_of_events == 0:
            print('num_of_events would have been none!')
            num_of_events = 1

        max_num_of_reps = np.floor(num_of_events / patterning_event_min)
        if max_num_of_reps < 2:
            num_of_reps = 1
        else:
            num_of_reps = rng.choice(np.arange(1, max_num_of_reps)).astype(int)
        num_per_rep = np.floor(num_of_events / num_of_reps).astype(int)
        durs = rsm(num_per_rep, nCVI)

        triads = self.get_triads(len(durs), shared=0)
        middle_freq = self.do_offset(base_min_freq, freq_offset)
        up = np.random.randint(3)
        if up == 0:
            start_freq = middle_freq * (2 ** -0.5)
            end_freq = middle_freq * (2 ** 0.5)
            min_freq = np.linspace(start_freq, end_freq, len(triads))
        elif up == 1:
            start_freq = middle_freq * (2 ** 0.5)
            end_freq = middle_freq * (2 ** -0.5)
            min_freq = np.linspace(start_freq, end_freq, len(triads))
        elif up == 2:
            min_freq = np.repeat(middle_freq, len(triads))
        freqs = [self.registrate(triad, min_freq[i]) for i, triad in enumerate(triads)]
        delays = [self.get_delays(np.random.randint(3)+1) for i in range(num_per_rep)]
        coef_center = self.do_offset(base_coef, coef_offset, subtract=1)
        coefs = [spread(coef_center, 2) for i in range(num_per_rep)]
        decay_center = self.do_offset(base_decay_dur, decay_offset, subtract=1)
        decays = [spread(decay_center, 2) for i in range(num_per_rep)]
        vol_center = base_vol + ((vol_offset - 0.5) / 2)
        vols = [spread(vol_center, 4, scale='linear') for i in range(num_per_rep)]
        vols = [[np.clip(spread(i, 2, scale='linear'), 0, 1) for j in range(3)] for i in vols]


        rep_delays = [delays for i in range(num_of_reps)]
        rep_freqs = [freqs for i in range(num_of_reps)]
        rep_coefs = [coefs for i in range(num_of_reps)]
        rep_decays = [decays for i in range(num_of_reps)]
        rep_vols = [vols for i in range(num_of_reps)]
        rep_durs = [jiggle_sequence(durs, 1.2) for i in range(num_of_reps)]

        for gap_index in range(1, num_of_reps)[::-1]:
            gap = np.random.randint(2, dtype=bool)
            if gap:
                gap_size = np.random.rand() * max_gap_size
                rep_durs.insert(gap_index, np.array([gap_size]))
                rep_freqs.insert(gap_index, ['Rest()'])
                rep_delays.insert(gap_index, [[0, 0, 0]])
                rep_coefs.insert(gap_index, [0.1])
                rep_decays.insert(gap_index, [3])
                rep_vols.insert(gap_index, [[0.5, 0.5, 0.5]])

        full_durs = np.concatenate(rep_durs)
        full_durs /= np.sum(full_durs)
        full_freqs = list(itertools.chain.from_iterable(rep_freqs))
        full_delays = np.concatenate(rep_delays)
        full_coefs = np.concatenate(rep_coefs)
        full_decays = np.concatenate(rep_decays)
        full_vols = np.concatenate(rep_vols)

        full_event_locs = np.append((0), np.cumsum(full_durs)[:-1])
        start_offset = (1 / len(full_event_locs)) - self.rt_since_last / self.real_dur
        if start_offset < 0: start_offset = 0
        full_event_locs += start_offset
        full_event_locs = full_event_locs[np.nonzero(full_event_locs < 1)]

        size = np.size(full_event_locs)
        full_freqs.insert(0, 'Rest()')
        full_freqs = full_freqs[:size]
        fd_shape = (len(full_delays) + 1, 3)
        full_delays = np.append([0, 0, 0], full_delays).reshape(fd_shape)[:size]
        full_coefs = np.append(0.1, full_coefs)[:size]
        full_decays = np.append(3, full_decays)[:size]
        v_shape = (len(full_vols) + 1, 3)
        full_vols = np.append([0.5, 0.5, 0.5], full_vols).reshape(v_shape)[:size]
        self.packets = []
        for i in range(len(full_event_locs)):
            packet = {}
            packet['freqs'] = full_freqs[i]
            packet['dur'] = full_durs[i]
            packet['coef'] = full_coefs[i]
            packet['decay'] = full_decays[i]
            packet['delays'] = full_delays[i]
            packet['vol'] = full_vols[i]
            packet['start'] = full_event_locs[i]
            if i == len(full_event_locs) - 1:
                packet['end'] = 1.0
            else:
                packet['end'] = full_event_locs[i+1]

            self.packets.append(packet)
        return self.packets

    def get_delays(self, groups=2, max_del=0.2):
        """Returns array with delay times that specify how much after event time
        the three notes of the triad will be struck. Always at least one at zero
        time. If groups == 2, (at least) two of three must occur simultaenously.
        If groups == 3, all can occur seperate, (a 'strum')."""
        max_del = float(max_del)
        delays = np.zeros(3)
        if groups == 1:
            return np.array([0, 0, 0])
        elif groups == 2:
            del_times = np.random.random() * max_del
            num_dels = np.random.randint(2) + 1
            idxs = rng.choice(np.arange(3), size=num_dels, replace=False)
        elif groups == 3:
            del_times = np.random.random(2) * max_del
            idxs = rng.choice(np.arange(3), size=2, replace=False)
        delays[idxs] = del_times
        return delays

    def registrate(self, chord, min_):
        freqs = np.sort(chord).astype(float)
        freqs = np.where(freqs < min_,
            freqs * (2 ** np.ceil(np.log2(min_/freqs))), freqs)
        freqs = np.where(freqs >= 2 * min_,
            freqs / (2 ** np.floor(np.log2(freqs/min_))), freqs)
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

    def do_offset(self, base_val, offset, subtract = 0.5):
        """Spreads value to limit of (x ** -0.5, x ** 0.5) according to offset.
        offset must be betwen 0 and 1. """
        return base_val * (2 ** (offset - subtract))


# class Klank:
# 
#     def __init__(self, piece, irama, fine_tuning):
#         self.piece = piece
#         self.irama = irama
#         self.fine_tuning = fine_tuning
#         self.assign_frame_timings()
#         self.current_mode = None
#         self.levels = np.arange(1, 5) / 5
#         self.pan_pos = np.arange(-3, 4) / 4
# 
# 
#         if self.irama == 0:
#             self.rt_td = 4
#             self.tot_events = self.rt_dur * self.rt_td
# 
#         self.repeatable_make_packets(0.25)
#         self.alt_add_notes()
#         self.add_real_times()
# 
#         # self.make_packets()
#         # self.add_notes()
#         # self.add_spec()
#         # self.add_real_times()
# 
# 
# 
#     def assign_frame_timings(self):
#         trans = self.piece.get_irama_transitions()
#         if self.irama == 0:
#             start = (0, 0, 0)
#         else:
#             start = trans[self.irama-1]
#             start = (start[0], start[1], self.fine_tuning[self.irama-1])
#         if self.irama == 3:
#             end = (self.piece.noc+1, 0, 0)
#         else:
#             end = trans[self.irama]
#             end = (end[0], end[1], self.fine_tuning[self.irama])
# 
#         start_section = self.piece.sections[start[1]]
#         ss_beginning = start_section.cy_start
#         ss_ending = start_section.cy_end
#         ss_dur = ss_ending - ss_beginning
# 
#         end_section = self.piece.sections[end[1]]
#         es_beginning = end_section.cy_start
#         es_ending = end_section.cy_end
#         es_dur = es_ending - es_beginning
# 
#         self.cy_start = start[0] + ss_beginning + ss_dur * start[2]
#         self.cy_end = end[0] + es_beginning + es_dur * end[2]
#         self.cy_dur = self.cy_end - self.cy_start
# 
#         self.rt_start = self.piece.time.real_time_from_cycles(self.cy_start)
#         self.rt_end = self.piece.time.real_time_from_cycles(self.cy_end)
#         self.rt_dur = self.rt_end - self.rt_start
# 
# 
# 
#     def repeatable_make_packets(self, repeat_chance=0.25, rest_ratio=0.3):
#         dur_min = 1
#         dur_octs = 4
#         dur_tot = self.rt_dur * (1 - rest_ratio)
#         avg_dur = dur_min * (2 ** (dur_octs / 2))
#         num_events = np.round(dur_tot / avg_dur)
#         dur_pool = rsm(num_events, 60)
#         np.random.shuffle(dur_pool)
# 
# 
#         events = []
#         ct = 0
#         for i in range(len(dur_pool)):
#             if i == 0 or np.random.rand() > repeat_chance:
#                 obj = {}
#                 obj['dur'] = dur_pool[ct]
#                 obj['is_copy'] = False
#                 obj['rep_choice_weight'] = np.random.rand()
#                 obj['unique_id'] = ct
#                 obj['loc_id'] = i
#                 obj['original_loc_id'] = i
#                 events.append(obj)
#                 ct += 1
#             else:
#                 iterable = (k['rep_choice_weight'] for k in events)
#                 weights = np.fromiter(iterable, float)
#                 weights /= sum(weights)
#                 index = np.random.choice(np.arange(len(events)), p=weights)
#                 obj = events[index].copy()
#                 obj['is_copy'] = True
#                 obj['loc_id'] = i
# 
#                 events.append(obj)
#         durs = np.fromiter((i['dur'] for i in events), float)
#         durs *= dur_tot * np.sum(durs)
#         for i, obj in enumerate(events):
#             obj['dur'] = durs[i]
# 
#         max_num_rests = len(durs) + 1#inclusive
#         min_num_rests = np.round(len(durs)/4)
#         if min_num_rests == 0: min_num_rests = 1
#         num_rest_octs = np.log2(max_num_rests / min_num_rests)
#         num_rests = min_num_rests * 2 ** np.random.uniform(num_rest_octs)
#         num_rests = np.round(num_rests).astype(int)
#         cy_rests = rsm(num_rests, 60) * rest_ratio * self.cy_dur
#         rest_locs = rng.choice(np.arange(len(events)+1), num_rests, False)
#         cy_time = 0
#         # self.packets = []
#         self.events = []
#         r_ct = 0
#         for i, event in enumerate(events):
#             event['packets'] = []
#             if event['is_copy'] == True:
#                 original_event = events[event['original_loc_id']]
#                 event['phrase'] = original_event['phrase']
#             else:
#                 dur = event['dur']
#                 dc_durs, dc_edges = self.make_pc_edges(dur)
#                 subdivs = np.random.choice(np.arange(len(dc_durs) + 1, len(dc_durs) * 2 + 1))
#                 nCVI_low = 20
#                 nCVI_high = 60
#                 nCVI_tup = (nCVI_low, nCVI_high)
#                 phrase = phrase_compiler(dc_durs, dc_edges, subdivs, 24, nCVI_tup)
#                 event['phrase'] = phrase
#             cy_phrase_durs = event['phrase'] * self.cy_dur / self.rt_dur
#             cy_phrase_starts = cy_time + np.concatenate(([0], np.cumsum(cy_phrase_durs)[:-1]))
#             cy_time += dur * self.cy_dur / self.rt_dur
# 
# 
#             for j in range(len(event['phrase'])):
#                 packet = {}
#                 packet['cy_start'] = cy_phrase_starts[j]
#                 packet['cy_dur'] = cy_phrase_durs[j]
#                 packet['type'] = 'note'
#                 packet['phrase_num'] = i
#                 packet['original_loc_id'] = event['original_loc_id']
#                 packet['loc_id'] = event['loc_id']
#                 packet['is_copy'] = event['is_copy']
#                 packet['irama'] = self.irama
#                 event['packets'].append(packet)
#             if i in rest_locs:
#                 packet = {}
#                 packet['cy_start'] = cy_time
#                 packet['cy_dur'] = cy_rests[r_ct]
#                 packet['type'] = 'rest'
#                 packet['phrase_num'] = i
#                 packet['original_loc_id'] = event['original_loc_id']
#                 packet['loc_id'] = event['loc_id']
#                 packet['is_copy'] = event['is_copy']
#                 packet['irama'] = self.irama
#                 event['packets'].append(packet)
#                 cy_time += cy_rests[r_ct]
#                 r_ct += 1
#             self.events.append(event)
#         breakpoint()
# 
#     def alt_add_notes(self):
#         # first, put into grouped_packets, according to when mode changes
#         self.packets = []
#         for e in self.events:
#             for packet in e['packets']:
#                 self.packets.append(packet)
# 
#         out_ct, in_ct, current_mode = 0, 0, None
#         self.g_packets = [] # grouped packets
#         for packet in self.packets:
#             em_event = self.get_em_event(packet)
#             if current_mode != (em_event['mode'], em_event['variation']):
#                 current_mode = (em_event['mode'], em_event['variation'])
# 
#                 if out_ct != 0:
#                     self.g_packets.append(self.packets[out_ct-in_ct:out_ct])
#                     in_ct = 0
#             out_ct += 1
#             in_ct += 1
# 
#         self.g_packets.append(self.packets[out_ct-in_ct:out_ct])
# 
#         decay_prop = 4
# 
#         levels_0 = h_tools.dc_alg(len(self.levels), len(self.g_packets))
#         levels_0 = self.levels[levels_0]
#         levels_1 = h_tools.dc_alg(len(self.levels), len(self.g_packets))
#         levels_1 = self.levels[levels_1]
#         levels = [(levels_0[i], levels_1[i]) for i in range(len(self.g_packets))]
# 
#         for i, gp in enumerate(self.g_packets):
#             # breakpoint()
#             em_event = self.get_em_event(gp[0])
#             mode = self.piece.modes[em_event['variation'], em_event['mode']]
#             print(em_event['variation'], em_event['mode'])
#             mode_size = np.random.choice([4, 5, 6, 7])
#             sub_mode = get_sub_mode(mode, mode_size)
#             cs = np.arange(2, len(sub_mode))
#             cs_min = 2
#             cs_max = np.round((len(sub_mode) - cs_min) + cs_min)
#             if cs_max == cs_min: cs_max = cs_min + 1
#             cs = np.arange(cs_min, cs_max).astype(int)
#             ns = Note_Stream(sub_mode, self.piece.fund, None, cs)
#             reg_mins = 75 * 2 ** np.random.uniform(0, 1.5, size=2)
#             reg_maxs = 566 * 2 ** np.random.uniform(0, 1.5, size=2)
#             reg_min = np.linspace(reg_mins[0], reg_mins[1], len(gp))
#             reg_max = np.linspace(reg_maxs[0], reg_maxs[1], len(gp))
# 
#             this_prop = spread(decay_prop, 2)
#             pan_gamut_size = np.random.choice(np.arange(4, 10))
#             pan_gamut = np.random.random(size=pan_gamut_size) * 2 - 1
#             pans = pan_gamut[h_tools.dc_alg(len(pan_gamut), len(gp))]
#             transient_dur = spread(0.005, 6)
#             transient_curve = spread(0, 4, 'linear')
# 
# 
#             loc_id = gp[0]['loc_id']
#             p_ct = 0
#             for p, packet in enumerate(gp):
#                 packet['mode'] = mode
#                 if packet['loc_id'] != loc_id:
#                     p_ct = 0
#                     loc_id = packet['loc_id']
#                 if packet['is_copy'] == True:
#                     if packet['type'] == 'rest':
#                         packet['freqs'] = [100]
#                         packet['amps'] = [0.5]
#                         packet['pan'] = 0.0
#                         packet['cy_decays'] = [1.0]
#                         packet['transient_dur'] = 0.005
#                         packet['transient_curve'] = 0.0
# 
#                     else:
#                         orig_event = self.events[packet['original_loc_id']]
#                         if np.all(orig_event['packets'][p_ct]['mode'] == mode):
#                             source = orig_event['packets'][p_ct]
#                             packet['freqs'] = source['freqs']
#                             packet['amps'] = source['amps']
#                             packet['pan'] = source['pan']
#                             packet['transient_dur'] = source['transient_dur']
#                             packet['transient_curve'] = source['transient_curve']
#                             packet['cy_decays'] = source['cy_decays']
#                         else:
#                             reg_tup = (reg_min[p], reg_max[p])
#                             packet['freqs'] = ns.next_chord(reg_tup)
#                             dur = packet['cy_dur']
#                             size = np.size(packet['freqs'])
#                             decays = np.ones(size) * dur * spread(this_prop, 2)
#                             decays = np.array([spread(i, 2.0) for i in decays])
#                             packet['cy_decays'] = decays
#                             this_levels = (np.clip(spread(levels[i][k], 3), 0, 1) for k in range(2))
#                             this_levels = tuple(this_levels)
#                             amps = np.linspace(this_levels[0], this_levels[1], size)
#                             amps = np.array([np.clip(spread(i, 2.0), 0, 1) for i in amps])
# 
#                             packet['amps'] = amps
#                             packet['pan'] = pans[p]
#                             packet['transient_dur'] = spread(transient_dur, 4)
#                             packet['transient_curve'] = spread(transient_curve, 2, 'linear')
# 
#                 else:
#                     if packet['type'] == 'note':
#                         reg_tup = (reg_min[p], reg_max[p])
#                         packet['freqs'] = ns.next_chord(reg_tup)
#                     else:
#                         packet['freqs'] = 100
# 
#                     dur = packet['cy_dur']
#                     size = np.size(packet['freqs'])
#                     decays = np.ones(size) * dur * spread(this_prop, 2)
#                     decays = np.array([spread(i, 2.0) for i in decays])
#                     packet['cy_decays'] = decays
#                     this_levels = (np.clip(spread(levels[i][k], 3), 0, 1) for k in range(2))
#                     this_levels = tuple(this_levels)
#                     amps = np.linspace(this_levels[0], this_levels[1], size)
#                     amps = np.array([np.clip(spread(i, 2.0), 0, 1) for i in amps])
# 
#                     packet['amps'] = amps
#                     packet['pan'] = pans[p]
#                     packet['transient_dur'] = spread(transient_dur, 4)
#                     packet['transient_curve'] = spread(transient_curve, 2, 'linear')
#                 p_ct += 1
# 
#     def make_packets(self):
#         """Just the temporality. Add notes and other stats later."""
#         # this works for irama 1. For later iramas, where repetition is wanted,
#         # this has to happen hierarchically rather than sequentially.
# 
#         rest_ratio = 0.15 # proportion of klank that consists of rests
#         # dur_range = (3, 20)
#         dur_min = 2.5
#         dur_octs = 3
#         rt_durs = []
#         remaining = self.rt_dur
#         while sum(rt_durs) < self.rt_dur * (1 - rest_ratio):
#             remaining = self.rt_dur * (1 - rest_ratio) - sum(rt_durs)
#             if remaining >= dur_min and remaining < dur_min * 2 ** dur_octs:
#                 next = remaining
#             elif remaining < dur_min:
#                 rt_durs[-1] += remaining
#                 break
#             else:
#                 next = dur_min * 2 ** np.random.uniform(dur_octs)
#             rt_durs.append(next)
# 
#         max_num_rests = len(rt_durs) + 1#inclusive
#         min_num_rests = np.round(len(rt_durs)/3)
#         num_rests = np.random.randint(min_num_rests, max_num_rests)
#         cy_rests = rsm(num_rests, 60) * rest_ratio * self.cy_dur
#         rest_locs = rng.choice(np.arange(len(rt_durs)+1), size=num_rests, replace=False)
#         cy_time = 0
#         self.packets = []
#         r_ct = 0
#         for i, rt_dur in enumerate(rt_durs):
#             dc_durs, dc_edges = self.make_pc_edges(rt_dur)
#             subdivs = np.random.choice([3, 4, 5, 6])
#             nCVI_low = 20
#             nCVI_high = 60
#             phrase = phrase_compiler(dc_durs, dc_edges, subdivs, 24, (nCVI_low, nCVI_high))
#             cy_phrase_durs = phrase * self.cy_dur / self.rt_dur
#             cy_phrase_starts = cy_time + np.concatenate(([0], np.cumsum(cy_phrase_durs)[:-1]))
#             cy_time += rt_dur * self.cy_dur / self.rt_dur
#             if cy_time > 8: breakpoint()
#             # phrase_durs.append(cy_phrase_durs)
#             # phrase_starts.append(cy_phrase_starts)
#             for j in range(len(phrase)):
#                 packet = {}
#                 packet['cy_start'] = cy_phrase_starts[j]
#                 packet['cy_dur'] = cy_phrase_durs[j]
#                 packet['type'] = 'note'
#                 packet['phrase_num'] = i
#                 self.packets.append(packet)
#             if i in rest_locs:
#                 packet = {}
#                 packet['cy_start'] = cy_time
#                 packet['cy_dur'] = cy_rests[r_ct]
#                 packet['type'] = 'rest'
#                 packet['phrase_num'] = i
#                 self.packets.append(packet)
#                 cy_time += cy_rests[r_ct]
#                 r_ct += 1
# 
# 
#     def add_notes(self):
#         outer_ct = 0
#         inner_ct = 0
#         current_mode = None
#         self.grouped_packets = []
#         for packet in self.packets:
# 
#             event = self.get_em_event(packet)
#             # mode = self.piece.modes[var, event['mode']]
#             if current_mode != (event['mode'], event['variation']):
#                 # sub_mode = get_sub_mode(mode, 5)
#                 # packet['submode'] = sub_mode
#                 current_mode = (event['mode'], event['variation'])
#                 if outer_ct != 0:
#                     self.grouped_packets.append(self.packets[outer_ct-inner_ct:outer_ct])
#                     inner_ct = 0
#             outer_ct += 1
#             inner_ct +=1
# 
#         self.grouped_packets.append(self.packets[outer_ct-inner_ct:outer_ct])
# 
#         for i, gp in enumerate(self.grouped_packets):
#             event = self.get_em_event(gp[0])
#             mode = self.piece.modes[event['variation'], event['mode']]
#             mode_size = np.random.choice([4, 5, 6, 7])
#             sub_mode = get_sub_mode(mode, mode_size)
#             cs = np.arange(2, len(sub_mode))
#             cs_min = 2
#             cs_max = np.round((len(sub_mode) - cs_min) + cs_min)
#             if cs_max == cs_min: cs_max = cs_min + 1
#             cs = np.arange(cs_min, cs_max).astype(int)
#             gamut_size = np.random.choice(np.arange(6, 18))
#             ns = Note_Stream(sub_mode, self.piece.fund, chord_sizes=cs)
#             register = (100, 500) # just for now
#             reg_mins = 75 * 2 ** np.random.uniform(0, 1.5, size=2)
#             reg_maxs = 566 * 2 ** np.random.uniform(0, 1.5, size=2)
#             reg_min = np.linspace(reg_mins[0], reg_mins[1], len(gp))
#             reg_max = np.linspace(reg_maxs[0], reg_maxs[1], len(gp))
#             for p, packet in enumerate(gp):
#                 if packet['type'] == 'note':
#                     packet['freqs'] = ns.next_chord((reg_min[p], reg_max[p]))
#                 else:
#                     packet['freqs'] = [100]
# 
#         # now assign the sub mode to each
# 
# 
#     def add_spec(self, prop=5, amp=0.5):
#         """adds decay times and amp levels for Klank supercollider Ugen. """
#         #for a start, just have random delays, less than total dur.
#         levels_0 = h_tools.dc_alg(len(self.levels), len(self.grouped_packets))
#         levels_0 = self.levels[levels_0]
#         levels_1 = h_tools.dc_alg(len(self.levels), len(self.grouped_packets))
#         levels_1 = self.levels[levels_1]
#         levels = [(levels_0[i], levels_1[i]) for i in range(len(self.grouped_packets))]
# 
#         pan_pos = h_tools.dc_alg(len(self.pan_pos), len(self.grouped_packets))
#         pan_pos = self.pan_pos[pan_pos]
# 
#         for i, gp in enumerate(self.grouped_packets):
#             this_prop = spread(prop, 2)
#             pan_gamut_size = np.random.choice(np.arange(4, 10))
#             pan_gamut = np.random.random(size=pan_gamut_size) * 2 - 1
#             pans = pan_gamut[h_tools.dc_alg(len(pan_gamut), len(gp))]
#             transient_dur = spread(0.005, 6)
#             transient_curve = spread(0, 4, 'linear')
#             for p, packet in enumerate(gp):
#                 dur = packet['cy_dur']
#                 decays = np.ones(np.size(packet['freqs'])) * dur * spread(this_prop, 2)
#                 decays = np.array([spread(i, 2.0) for i in decays])
#                 packet['cy_decays'] = decays
# 
#                 this_levels = (np.clip(spread(levels[i][0], 3), 0, 1), np.clip(spread(levels[i][1], 3), 0, 1))
#                 amps = np.linspace(this_levels[0], this_levels[1], np.size(packet['freqs']))
#                 amps = np.array([np.clip(spread(i, 2.0), 0, 1) for i in amps])
# 
#                 packet['amps'] = amps
#                 packet['pan'] = pans[p]
#                 packet['transient_dur'] = spread(transient_dur, 4)
#                 packet['transient_curve'] = spread(transient_curve, 2, 'linear')
# 
# 
#     def get_em_event(self, packet):
#         """Started calling it `em_event` instead of just `event` to signify
#         that they are different things."""
#         start = packet['cy_start']
#         cycle = (start // 1).astype(int)
#         event_map = self.piece.cycles[cycle].event_map
#         em_starts = np.array(list(event_map.keys()))
#         event_idx = np.nonzero(start%1 >= em_starts)[0][-1]
#         key = list(event_map.keys())[event_idx]
#         event = event_map[key]
#         return event
# 
#     def add_real_times(self):
#         time = self.piece.time
#         for packet in self.packets:
#             cy_start = packet['cy_start']
#             cy_end = cy_start + packet['cy_dur']
#             rt_start = time.real_time_from_cycles(cy_start)
#             rt_end = time.real_time_from_cycles(cy_end)
#             rt_dur = rt_end - rt_start
#             packet['rt_start'] = rt_start
#             packet['rt_dur'] = rt_dur
#             cy_dec_ends = packet['cy_decays'] + cy_start
#             rt_dec_ends = [time.real_time_from_cycles(i) for i in cy_dec_ends]
#             rt_dec_ends = np.array(rt_dec_ends)
#             rt_dec_durs = rt_dec_ends - rt_start
#             packet['rt_decays'] = rt_dec_durs
#             if packet['type'] == 'rest':
#                 packet['rt_dur'] = "Rest(" + str(packet['rt_dur']) + ")"
# 
# 
# 
#     def make_pc_edges(self, rt_dur, irama=0):
#         # for irama 1, shape supplied to phrase compiler should be either
#         # down-up-down, down-up-middle, middle-up-down
#         if irama == 0:
#             midpoint = np.random.uniform(0.25, 0.75)
#             dc_durs = np.array([midpoint, 1-midpoint]) * rt_dur
#             down = np.random.uniform(1.5, 3) / 3
#             middle = np.random.uniform(3, 4.5) / 3
#             up = np.random.uniform(4.5, 6) / 3
#             profiles = [[down, up, down], [down, up, middle], [middle, up, down]]
#             dc_edges = profiles[np.random.choice(np.arange(3))]
#             return dc_durs, dc_edges
# 
#     def make_generalized_pc_edges(self, rt_dur, irama=0):
#         min_seg_dur = 2
#         max_seg_dur = 6
#         seg_dur_octs = np.log2(max_seg_dur / min_seg_dur)
#         avg_seg_dur = min_seg_dur * (2 ** np.random.uniform(seg_dur_octs))
#         num_segs = np.round(rt_dur / avg_seg_dur).astype(int)
#         if num_segs == 0: num_segs = 1
#         dc_durs = rsm(num_segs, 15) * rt_dur
# 
#         min_td = 0.25
#         max_td = 6
#         td_octs = np.log2(max_td / min_td)
#         dc_edges = [min_td * 2 ** np.random.uniform(td_octs) for i in range(num_segs + 1)]
#         # should I sculpt these such that the mins end up at the beginning and
#         # end, so that phrases ease in and out?
#         # Or, should I accept what the fates ordain, and go about my merry way?
#         # for now, stick with the fates.
# 
#         return dc_durs, dc_edges
# 
#     def save_as_json(self, path):
#         json.dump(self.packets, open(path, 'w'), cls=h_tools.NpEncoder)


class MovingPluck:

    def __init__(self, piece):
        self.piece = piece
        self.irama_levels = self.piece.time.irama_levels
        self.time = self.piece.time
        self.make_transition_tables()
        self.assign_frame_timings()
        self.assign_phrase_timings()
        self.get_mode_regions()
        self.make_phrases()
        self.save_phrases()



    def save_phrases(self):
        path = 'JSON/moving_pluck_phrases.JSON'
        json.dump(self.phrases, open(path, 'w'), cls=h_tools.NpEncoder)




    def make_phrases(self):
        init_td = 0.5
        init_nCVI = 8
        self.phrases = []
        for il in range(self.irama_levels):
            freq_min = 100 
            freq_min_mult_gamut = 2 ** np.linspace(0, il * 2/3, 10)
            freq_min_mult = h_tools.dc_alg(10, len(self.il_phrase_timings[il]), alpha=2)
            freq_min_mult = freq_min_mult_gamut[freq_min_mult]
            freq_min = freq_min * freq_min_mult
            
            
            attack_ratio = 0.1 + (0.2 * il)
            avg_td = init_td * (Golden ** (il/3))
            avg_nCVI = init_nCVI * (2 ** il)
            phrase_timings = self.il_phrase_timings[il]
            
            td_mult_gamut = 2 ** np.linspace(-1 * Golden ** (il/2), Golden ** (il/2), 10)
            td_mult = h_tools.dc_alg(10, len(self.il_phrase_timings[il]), alpha=2)
            td_mult = td_mult_gamut[td_mult] * avg_td
            
            
            # breakpoint()
            for idx, pt in enumerate(phrase_timings):
                p_td = td_mult[idx]
                p_nCVI = spread(avg_nCVI, 1.5)
                size = np.round(pt['rt_dur'] * p_td).astype(int)
                if size == 0: size = 1
                dur_tot = pt['cy_dur'] # keep it in cy, then convert later
                modes = []
                tts = []
                for idx in range(len(pt['modes'])):
                    var_idx = pt['variations'][idx]
                    mode_idx = pt['modes'][idx]
                    mode = self.piece.modes[var_idx][mode_idx]
                    modes.append(mode)
                    tt = self.tts[mode_idx][var_idx]
                    tts.append(tt)
                modes = np.array(modes)
                tts = np.array(tts)
                midpoints = pt['midpoints']
                if len(np.shape(midpoints)) == 0: midpoints = [midpoints]
                if len(modes) == 1:
                    pp = make_pluck_phrase(modes[0], self.piece.fund, size,
                        dur_tot, p_nCVI, (freq_min[idx], 2*freq_min[idx] * (Golden ** (il/2))),
                        p_transition=tts[0], attack_ratio=attack_ratio)
                else:
                    pp = make_multi_changing_pluck_phrase(modes, self.piece.fund,
                        size, dur_tot, p_nCVI, tts, midpoints,
                        (freq_min[idx], 2*freq_min[idx] * (Golden ** (il/2))), attack_ratio=attack_ratio)
                    # next thing to do is convert this timing stuff,
                # which is in cycles, to real time. will require including the
                # cy start time in tphhe function itself? or going through some
                # conversion function?
                pp = self.add_real_time_to_pluck_phrase(pp, pt)
                self.phrases.append(pp)


    def add_real_time_to_pluck_phrase(self, pp, pt):
        cy_start = pt['cy_start_time']
        rt_start_of_phrase = self.piece.time.real_time_from_cycles(cy_start)
        cy_durs = pp['durs']
        rt_durs = []
        cy_pluck_starts = pp['pluckStarts']
        rt_pluck_starts = []
        accumulator = 0
        for cy_dur in cy_durs:
            start = cy_start + accumulator
            accumulator += cy_dur
            end = cy_start + accumulator
            rt_start = self.piece.time.real_time_from_cycles(start)
            rt_end = self.piece.time.real_time_from_cycles(end)
            rt_dur = rt_end - rt_start
            rt_durs.append(rt_dur)

        for pluck_start in cy_pluck_starts:
            cy_pluck_start = pluck_start + cy_start
            rt_pluck_start = self.piece.time.real_time_from_cycles(cy_pluck_start)
            rt_pluck_starts.append(rt_pluck_start - rt_start_of_phrase)
        rt_dur_tot = sum(rt_durs)
        pp['rt_durs'] = rt_durs
        pp['rt_pluckStarts'] = rt_pluck_starts
        pp['rt_durTot'] = rt_dur_tot
        pp['rt_releaseDur'] = 3 * rt_dur_tot
        pp['rt_start'] = rt_start_of_phrase
        return pp



    def make_transition_tables(self):
        self.tts = []
        for s in range(self.piece.nos):
            section_tts = []
            for i in range(3):
                mode = self.piece.modes[i, s]
                p_transition = generate_transition_table(mode)
                section_tts.append(p_transition)
            self.tts.append(section_tts)


    def assign_phrase_timings(self):
        self.il_phrase_timings = [] # holds phrase timings for each irama level, in
        # terms of rt durTot of irama level (as if not slowing down).


        for i in range(self.irama_levels):
            st_avg_dur = 4
            end_avg_dur = 12
            abs_min = 2


            avg_dur = st_avg_dur * 2 ** (i * np.log(end_avg_dur / st_avg_dur) / (self.irama_levels-1))
            spread_ratio = avg_dur / abs_min
            avg_nCVI = nCVI([spread(1, spread_ratio) for _ in range(1000)])
            rest_prop = 0.3
            rp_lo = 0.4 - (i * 0.1)
            num_of_rest_prop = np.random.uniform(rp_lo, rp_lo + 0.3)
            phrase_durtot = self.rt_durs[i] * (1 - rest_prop)
            rest_durtot = self.rt_durs[i] * (rest_prop)
            num_phrases = np.round(phrase_durtot / avg_dur).astype(int)
            num_rests = np.round(num_of_rest_prop * num_phrases).astype(int)
            if num_rests == 0 or num_phrases == 0: breakpoint()
            rest_locs = rng.choice(np.arange(num_phrases), num_rests, replace=False)
            phrase_durs = rsm(num_phrases, avg_nCVI) * phrase_durtot
            rest_durs = rsm(num_rests, avg_nCVI) * rest_durtot
            phrases = []
            time_ct = self.rt_starts[i]
            # breakpoint()

            rest_idx_ct = 0
            for p in range(num_phrases):
                start_prop = (time_ct - self.rt_starts[i]) / self.rt_durs[i] # with regard to whole irama level duration
                end_prop = (time_ct + phrase_durs[p] - self.rt_starts[i]) / self.rt_durs[i]
                cy_start_time = start_prop * self.cy_durs[i] + self.cy_starts[i]
                cy_end_time = end_prop * self.cy_durs[i] + self.cy_starts[i]
                cy_dur = cy_end_time - cy_start_time
                rt_start_time = self.time.real_time_from_cycles(cy_start_time)
                rt_end_time = self.time.real_time_from_cycles(cy_end_time)
                rt_dur = rt_end_time - rt_start_time
                # cy_mode_transitions = [i['cy_start'] for i in self.piece.consolidated_em]
                start_mt_idx = np.where(cy_start_time < self.piece.cy_mode_transitions)[0][0] - 1
                end_mt_idx = np.where(cy_end_time < self.piece.cy_mode_transitions)[0]
                if np.size(end_mt_idx) == 0:
                    end_mt_idx = len(self.piece.cy_mode_transitions)-1
                else:
                    end_mt_idx = end_mt_idx[0] - 1
                # breakpoint()
                if start_mt_idx == end_mt_idx:
                    modes = [self.piece.consolidated_em[start_mt_idx]['mode']]
                    vars = [self.piece.consolidated_em[start_mt_idx]['variation']]
                    midpoints = 0
                # elif start_mt_idx == end_mt_idx - 1:
                #     modes = []
                #     modes.append(self.piece.consolidated_em[start_mt_idx]['mode'])
                #     modes.append(self.piece.consolidated_em[end_mt_idx]['mode'])
                #     vars = []
                #     vars.append(self.piece.consolidated_em[start_mt_idx]['variation'])
                #     vars.append(self.piece.consolidated_em[end_mt_idx]['variation'])
                #     tr_point = self.piece.cy_mode_transitions[end_mt_idx]
                #     midpoint = (tr_point - cy_start_time) / cy_dur
                else:
                    modes = []
                    vars = []
                    midpoints = []
                    for j in range(end_mt_idx - start_mt_idx):
                        modes.append(self.piece.consolidated_em[j+start_mt_idx]['mode'])
                        vars.append(self.piece.consolidated_em[j+start_mt_idx]['variation'])

                        tr_point = self.piece.cy_mode_transitions[j+start_mt_idx+1]
                        midpoint = (tr_point - cy_start_time) / cy_dur
                        midpoints.append(midpoint)
                    if end_mt_idx != len(self.piece.consolidated_em):
                        modes.append(self.piece.consolidated_em[end_mt_idx]['mode'])
                        vars.append(self.piece.consolidated_em[end_mt_idx]['variation'])
                    # either make the `make_changing_pluck_phrase` accept
                    # unlimted  mode shifts; or break so that it never lands across more than two mode shifts
                current_mode = self.piece.modes[vars[0]][modes[0]]
                if len(phrases) == 0 and len(self.il_phrase_timings) == 0:
                    pitch_pairs = [None, None]
                else:
                    if len(phrases) != 0:
                        # this is all just getting info from previous phrase
                        last_mode_idx = phrases[-1]['modes'][-1]
                        last_var_idx = phrases[-1]['variations'][-1]
                        last_pitch_idx = phrases[-1]['pitch_pairs'][-1]
                        first_mode_idx = phrases[-1]['modes'][0]
                        first_var_idx = phrases[-1]['variations'][0]
                        first_pitch_idx = phrases[-1]['pitch_pairs'][0]
                    elif len(self.il_phrase_timings) != 0:
                        last_mode_idx = self.il_phrase_timings[-1][-1]['modes'][-1]
                        last_var_idx = self.il_phrase_timings[-1][-1]['variations'][-1]
                        last_pitch_idx = self.il_phrase_timings[-1][-1]['pitch_pairs'][-1]
                        first_mode_idx = self.il_phrase_timings[-1][-1]['modes'][0]
                        first_var_idx = self.il_phrase_timings[-1][-1]['variations'][0]
                        first_pitch_idx = self.il_phrase_timings[-1][-1]['pitch_pairs'][0]

                    last_mode = self.piece.modes[last_var_idx][last_mode_idx]
                    last_pitch = last_mode[last_pitch_idx]
                    first_mode = self.piece.modes[first_var_idx][first_mode_idx]
                    first_pitch = first_mode[first_pitch_idx]

                    closest_first_idx = closest_index(first_pitch, current_mode)
                    closest_last_idx = closest_index(last_pitch, current_mode)
                    # possible situations:
                    # 0. same first pitch,
                    # 1. same last pitch,
                    # 2. both same,
                    # 3. previous last pitch is new first pitch
                    situ = np.random.randint(4)
                    if situ == 0:
                        pitch_pairs = [closest_first_idx, None]
                    elif situ == 1:
                        pitch_pairs = [None, closest_last_idx]
                    elif situ == 2:
                        pitch_pairs = [closest_first_idx, closest_last_idx]
                    elif situ == 3:
                        pitch_pairs = [closest_last_idx, None]
                tt = self.tts[modes[0]][vars[0]]
                pitch_pairs = make_pitch_pairs(current_mode, tt, pitch_pairs[0], pitch_pairs[1])
                phrase_spec = {
                    'cy_start_time': cy_start_time,
                    'cy_dur': cy_dur,
                    'cy_end_time': cy_end_time,
                    'rt_start_time': rt_start_time,
                    'rt_end_time': rt_end_time,
                    'rt_dur': rt_dur,
                    'start_time': time_ct,
                    'dur': phrase_durs[p],
                    'end_time': time_ct + phrase_durs[p],
                    'irama': i,
                    'modes': modes,
                    'variations': vars,
                    'midpoints': midpoints,
                    'pitch_pairs': pitch_pairs
                    }
                phrases.append(phrase_spec)
                time_ct += phrase_durs[p]
                if p in rest_locs:
                    time_ct += rest_durs[rest_idx_ct]
                    rest_idx_ct += 1
            self.il_phrase_timings.append(phrases)
         # CHECK IF? (i think I fixed it) for some reason, things are off by like 0.5 seconds in real time ... calculation error?
        # TODO connect up to phrase maker to do it all in series, will take some work to make the SC work as well ...














    def assign_frame_timings(self):
        trans = self.piece.get_irama_transitions()
        self.trans = trans
        fine_tuning = np.random.random(size=3)
        self.cy_starts = []
        self.cy_ends = []
        self.cy_durs = []
        self.rt_starts = []
        self.rt_ends = []
        self.rt_durs = []

        for il in range(self.piece.time.irama_levels):
            if il == 0:
                start = (0, 0, 0)
            else:
                start = trans[il-1]
                start = (start[0], start[1], fine_tuning[il-1])
            if il == self.piece.time.irama_levels - 1:
                end = (self.piece.noc+1, 0, 0)
            else:
                end = trans[il]
                end = (end[0], end[1], fine_tuning[il])

            start_section = self.piece.sections[start[1]]
            ss_beginning = start_section.cy_start
            ss_ending = start_section.cy_end
            ss_dur = ss_ending - ss_beginning

            end_section = self.piece.sections[end[1]]
            es_beginning = end_section.cy_start
            es_ending = end_section.cy_end
            es_dur = es_ending - es_beginning

            cy_start = start[0] + ss_beginning + ss_dur * start[2]
            if il == self.piece.time.irama_levels - 1:
                cy_end = self.piece.noc
            else:
                cy_end = end[0] + es_beginning + es_dur * end[2]
            cy_dur = cy_end - cy_start

            rt_start = self.piece.time.real_time_from_cycles(cy_start)
            rt_end = self.piece.time.real_time_from_cycles(cy_end)
            rt_dur = rt_end - rt_start

            self.cy_starts.append(cy_start)
            self.cy_ends.append(cy_end)
            self.cy_durs.append(cy_dur)
            self.rt_starts.append(rt_start)
            self.rt_ends.append(rt_end)
            self.rt_durs.append(rt_dur)

    def get_mode_regions(self):
        cy_end = 0
        cy_ct = 0
        em_ct = 0
        self.mode_regions = []
        for il in range(self.piece.time.irama_levels):
            while cy_end < self.cy_ends[il]:
                em = self.piece.cycles[cy_ct].event_map
                key = list(em.keys())[em_ct]
                ev = em[key]
                cy_start = cy_ct + ev['start']
                cy_end = cy_ct + ev['end']

                mode = self.piece.modes[ev['variation'], ev['mode']]
                dur = ev['dur']
                obj = {'cy_start': cy_start, 'cy_end': cy_end, 'mode': mode, 'dur': dur}
                self.mode_regions.append(obj)
                em_ct += 1
                if em_ct >= len(em):
                    cy_ct += 1
                    em_ct = 0
                # em_ct = em_ct % self.piece.nos
