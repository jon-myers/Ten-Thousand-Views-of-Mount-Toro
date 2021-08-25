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
rng = default_rng()
Golden = (1 + (5 ** 0.5)) / 2

class Klank_alt:
    
    def __init__(self, piece):
        self.piece = piece
        self.fine_tuning = np.random.random(size=3)
        self.current_mode = None
        self.levels = np.arange(3, 5) / 5
        self.pan_pos = np.arange(-3, 4) / 4
        self.repeat_chance = np.linspace(0.2, 0.65, 4)
        self.rest_proportion = [0.1, 0.2, 0.3, 0.4]
        self.phrase_dur_nCVIs = np.linspace(20, 50, 4)
        self.avg_tds = [(Golden ** (i/2)) for i in range(4)]
        self.assign_frame_timings()
        self.assign_phrase_timings()
        self.assign_note_timings()
        self.group_by_mode()
        self.assign_notes()
        # breakpoint()
        self.make_packets()
        # breakpoint()
        self.save_packets()
        # breakpoint()
        
    
    def assign_frame_timings(self):
        trans = self.piece.get_irama_transitions()
        # something like [(1, 0), (4, 3), 6, 7)]
        self.cy_starts = []
        self.cy_ends = []
        self.cy_durs = []
        self.rt_starts = []
        self.rt_ends = []
        self.rt_durs = []
        for i in range(4):
            if i == 0:
                cy_start = 0
                rt_start = 0
            else:
                sec_idx = trans[i-1][1]
                sec_start = self.piece.sections[sec_idx].cy_start
                sec_end = self.piece.sections[sec_idx].cy_end
                sec_dur = sec_end - sec_start
                
                cy_start = trans[i-1][0] + sec_start + sec_dur * self.fine_tuning[i-1]
                rt_start = self.piece.time.real_time_from_cycles(cy_start)
            
            if i == 3:
                cy_end = self.piece.noc
                rt_end = self.piece.time.dur_tot
            else:
                sec_idx = trans[i][1]
                sec_start = self.piece.sections[sec_idx].cy_start
                sec_end = self.piece.sections[sec_idx].cy_end
                sec_dur = sec_end - sec_start
                
                cy_end = trans[i][0] + sec_start + sec_dur * self.fine_tuning[i]
                rt_end = self.piece.time.real_time_from_cycles(cy_end)
                
            cy_dur = cy_end - cy_start
            rt_dur = rt_end - rt_start
            self.cy_starts.append(cy_start)
            self.cy_ends.append(cy_end)
            self.cy_durs.append(cy_dur)
            self.rt_starts.append(rt_start)
            self.rt_ends.append(rt_end)
            self.rt_durs.append(rt_dur)
    
    def assign_phrase_timings(self):
        base_avg_phrase_dur = 2
        self.phrases = []
        for i in range(4):
            phrases = []
            avg_phrase_dur = base_avg_phrase_dur * (2 ** (i/4))
            rt_rest_dur_tot = self.rt_durs[i] * self.rest_proportion[i]
            num_of_rests = np.round(rt_rest_dur_tot / avg_phrase_dur).astype(int)
            if num_of_rests == 0: num_of_rests = 1
            rt_phrase_dur_tot = self.rt_durs[i] * (1 - self.rest_proportion[i])
            num_of_phrases = np.round(rt_phrase_dur_tot / avg_phrase_dur).astype(int)
            num_of_rep_phrases = np.round(num_of_phrases * self.repeat_chance[i]).astype(int)
            num_of_orig_phrases = num_of_phrases - num_of_rep_phrases
            orig_seq = rsm(num_of_orig_phrases, self.phrase_dur_nCVIs[i])
            max_reps = [2, 3, 4, 5]
            rep_sizes = split_into_groups(num_of_phrases, num_of_orig_phrases, max_reps[i])
            copy_status = []
            phrase_durs = np.array([])
            j = 0
            for rs in rep_sizes:
                for rsi in range(rs):
                    if rsi == 0:
                        copy_status.append('no')
                        phrase_durs = np.concatenate([phrase_durs, [orig_seq[j]]])
                        # phrase_durs.append(orig_seq[j])
                        j += 1
                        trig = 0
                    else:
                        copy_status.append(len(copy_status) - 1 - trig)
                        phrase_durs = np.concatenate([phrase_durs, [phrase_durs[-1]]])
                        trig += 1
                        # phrase_durs.append(phrase_durs[-1])
            phrase_durs = phrase_durs * rt_phrase_dur_tot / np.sum(phrase_durs)
            rest_durs = rsm(num_of_rests, self.phrase_dur_nCVIs[i]) * rt_rest_dur_tot
            rest_locs = rng.choice(np.arange(len(phrase_durs)), size=num_of_rests, replace=False)
            r_ct = 0
            o_ct = 0
            rt_dur_ct = self.rt_starts[i]
            cy_dur_ct = self.cy_starts[i]
            # assign the ncvi and td for next level down as well
            nCVIs = rsm(num_of_orig_phrases, 20 + 10 * i) * num_of_orig_phrases * 30
            tds = rsm(num_of_orig_phrases, 30 + 5 * i) * num_of_orig_phrases * self.avg_tds[i]
            trig = 0
            for k in range(num_of_phrases):
                if copy_status[k] == 'no':
                    cy_dur_tot = phrase_durs[k] * self.cy_durs[i] / self.rt_durs[i]
                    p_obj = {
                      'rt_dur_tot': phrase_durs[k],
                      'cy_dur_tot': cy_dur_tot,
                      'rt_start': rt_dur_ct,
                      'cy_start': cy_dur_ct,
                      'nCVI': nCVIs[o_ct],
                      'td': tds[o_ct],
                      'copy': 'no', 
                      'type': 'phrase',
                      'mode_info': self.get_modes(cy_dur_ct, cy_dur_tot),
                      'irama': i
                    }
                    o_ct += 1
                    trig = 0
                else:
                    cy_dur_tot = phrase_durs[k] * self.cy_durs[i] / self.rt_durs[i]
                    p_obj = {
                      'rt_dur_tot': phrase_durs[k], 
                      'cy_dur_tot': cy_dur_tot,
                      'rt_start': rt_dur_ct,
                      'cy_start': cy_dur_ct,
                      'copy': copy_status[k] + r_ct - trig,
                      'copy_target': phrases[copy_status[k] + r_ct - trig],
                      'type': 'phrase',
                      'mode_info': self.get_modes(cy_dur_ct, cy_dur_tot),
                      'irama': i
                    }
                    # breakpoint()
                phrases.append(p_obj)
                rt_dur_ct += phrase_durs[k]
                cy_dur_ct += phrase_durs[k] * self.cy_durs[i] / self.rt_durs[i]
                if np.isin(k, rest_locs):
                    r_obj = {
                      'rt_dur_tot': rest_durs[r_ct],
                      'cy_dur_tot': rest_durs[r_ct] * self.cy_durs[i] / self.rt_durs[i], 
                      'rt_start': rt_dur_ct,
                      'cy_start': cy_dur_ct,
                      'copy': 'no',
                      'type': 'rest',
                      'irama': i,
                      'mode_info': self.get_modes(cy_dur_ct, cy_dur_tot)
                    }
                    
                    phrases.append(r_obj)
                    rt_dur_ct += rest_durs[r_ct]
                    cy_dur_ct += rest_durs[r_ct] * self.cy_durs[i] / self.rt_durs[i]
                    r_ct += 1
                    trig += 1
            # breakpoint()
            self.phrases.append(phrases)
            
    def assign_note_timings(self):
        for irama in range(4):
            phrases = self.phrases[irama]
            for p, phrase in enumerate(phrases):
                if phrase['type'] == 'phrase':
                    if phrase['copy'] == 'no':
                        num_of_notes = np.round(phrase['td'] * phrase['rt_dur_tot']).astype(int)
                        cy_note_durs = rsm(num_of_notes, phrase['nCVI']) * phrase['cy_dur_tot']
                        cy_note_starts = np.concatenate([[0], np.cumsum(cy_note_durs)[:-1]]) + phrase['cy_start']
                        phrase['cy_note_durs'] = cy_note_durs
                        phrase['cy_note_starts'] = cy_note_starts
                    else:
                        target_phrase = phrases[phrase['copy']]
                        cy_note_durs = jiggle_sequence(target_phrase['cy_note_durs'], 1.2, True)
                        phrase['cy_note_durs'] = cy_note_durs
                        cy_note_starts = np.concatenate([[0], np.cumsum(cy_note_durs)[:-1]]) + phrase['cy_start']
                        phrase['cy_note_starts'] = cy_note_starts
                    get_real = lambda x: self.piece.time.real_time_from_cycles(x)
                    get_real = np.vectorize(get_real)    
                    rt_note_starts = get_real(cy_note_starts)
                    cy_end = phrase['cy_dur_tot'] + phrase['cy_start']
                    rt_end = get_real(cy_end)
                    rt_note_ends = np.concatenate([rt_note_starts[1:], [rt_end]])
                    rt_durs = rt_note_ends - rt_note_starts
                    phrase['output_rt_durs'] = rt_durs
                    phrase['output_rt_starts'] = rt_note_starts
                        
    def group_by_mode(self):
        self.phrase_groups = []
        self.all_phrases = list(itertools.chain.from_iterable(self.phrases))
        for phrase in self.all_phrases:
            mode_info = phrase['mode_info']
            if len(mode_info) == 1:
                mode = list(mode_info.values())[0]
                if np.all(mode == self.current_mode):
                    self.phrase_groups[-1].append(phrase)
                else:
                    self.phrase_groups.append([phrase])
                    self.current_mode = mode
            if len(mode_info) > 1:
                first_mode = list(mode_info.values())[0]
                last_mode = list(mode_info.values())[-1]
                if np.all(first_mode == self.current_mode):
                    self.phrase_groups[-1].append(phrase)
                else:
                    self.phrase_groups.append([phrase])
                    self.current_mode = last_mode
                
    def assign_notes(self):
        levels_0 = h_tools.dc_alg(len(self.levels), len(self.phrase_groups))
        levels_0 = self.levels[levels_0]
        levels_1 = h_tools.dc_alg(len(self.levels), len(self.phrase_groups))
        levels_1 = self.levels[levels_1]
        levels = [(levels_0[i], levels_1[i]) for i in range(len(self.phrase_groups))]
        
        pan_gamut = np.linspace(-1, 1, 10)
        pan_centers = h_tools.dc_alg(10, len(self.phrase_groups), alpha=2)
        pan_centers = pan_gamut[pan_centers]
        pan_bw_gamut = np.linspace(0, 0.5, 10)
        pan_bws = h_tools.dc_alg(10, len(self.phrase_groups), alpha=2)
        pan_bws = pan_bw_gamut[pan_bws] 
        
        transient_dur_gamut = 0.01 * (2 ** np.linspace(0, 3, 10))
        transient_durs = h_tools.dc_alg(10, len(self.phrase_groups), alpha=2)
        transient_durs = transient_dur_gamut[transient_durs]
        transient_dur_bw_gamut = np.linspace(0, 3, 10)
        transient_dur_bws = h_tools.dc_alg(10, len(self.phrase_groups), alpha=2)
        transient_dur_bws = transient_dur_bw_gamut[transient_dur_bws]
        
        transient_curve_gamut = np.linspace(-4, 4, 10)
        transient_curves = h_tools.dc_alg(10, len(self.phrase_groups), alpha=2)
        transient_curves = transient_curve_gamut[transient_curves]
        
        decay_ctr_gamut = 2 ** np.linspace(1, 3, 10)
        decay_ctrs = h_tools.dc_alg(10, len(self.phrase_groups), alpha=2)
        decay_ctrs = decay_ctr_gamut[decay_ctrs]
        decay_ctr_bw_gamut = np.linspace(0, 2, 10)
        decay_ctr_bws = h_tools.dc_alg(10, len(self.phrase_groups), alpha=2)
        decay_ctr_bws = decay_ctr_bw_gamut[decay_ctr_bws]
        
        cs_ctr_gamut = np.linspace(0, 1, 10)
        cs_ctrs = h_tools.dc_alg(10, len(self.phrase_groups), alpha=2)
        cs_ctrs = cs_ctr_gamut[cs_ctrs]
        cs_bw_gamut = np.linspace(0, 0.5, 10)
        cs_bws = h_tools.dc_alg(10, len(self.phrase_groups), alpha=2)
        cs_bws = cs_bw_gamut[cs_bws]
        
        
        
        
        
        
        cur_mode = None
        for g, group in enumerate(self.phrase_groups):
            print(g)
            mode = list(group[0]['mode_info'].values())[0]
            if not np.all(cur_mode == mode):
                mode_size = np.random.choice([4, 5, 6, 7])
                sub_mode = get_sub_mode(mode, mode_size)
                
                cs_min = 2
                cs_max = np.round((len(sub_mode) - cs_min) + cs_min)
                if cs_max == cs_min: cs_max = cs_min + 1
                
                low_cs = np.clip(cs_ctrs[g] - cs_bws[g], 0, 1)
                high_cs = np.clip(cs_ctrs[g] + cs_bws[g], 0, 1)
                low_cs = np.floor(cs_min + low_cs * (cs_max - cs_min)).astype(int)
                high_cs = np.ceil(cs_min + high_cs * (cs_max - cs_min)).astype(int)
                if high_cs == low_cs:
                    if high_cs == cs_max:
                        low_cs -= 1
                    else:
                        high_cs += 1
                
                
                
                cs = np.arange(low_cs, high_cs).astype(int)
                ns = Note_Stream(sub_mode, self.piece.fund, chord_sizes=cs)
            for phrase in group:
                pass_on = False
                if phrase['type'] == 'rest':
                    continue
                elif phrase['copy'] != 'no' and len(phrase['mode_info']) == 1:
                    first_mode = list(phrase['mode_info'].values())[0]
                    target_phrase = phrase['copy_target']
                    target_first_mode = list(target_phrase['mode_info'].values())[0]
                    if np.all(first_mode == target_first_mode):
                        phrase['freqs'] = target_phrase['freqs']
                        phrase['amps'] = target_phrase['amps']
                        phrase['decays'] = target_phrase['decays']
                        phrase['pan'] = target_phrase['pan']
                        phrase['transient_dur'] = target_phrase['transient_dur']
                        phrase['transient_curve'] = target_phrase['transient_curve']
                        
                    else:
                        pass_on = True
                else:
                    pass_on = True
                if pass_on:
                    reg_mins = 75 * (2 ** np.random.uniform(0, 1, size=2))
                    reg_maxs = 300 * (2 ** np.random.uniform(0, 1, size=2)) * (Golden ** phrase['irama'])
                    num_of_notes = len(phrase['cy_note_durs'])
                    reg_min = np.linspace(reg_mins[0], reg_mins[1], num_of_notes)
                    reg_max = np.linspace(reg_maxs[0], reg_maxs[1], num_of_notes) 
                    
                    # set frequencies
                    freqs = []
                    if len(phrase['mode_info']) == 1:
                        for i in range(num_of_notes):
                            freqs.append(ns.next_chord((reg_min[i], reg_max[i])))
                    else:
                        modes = list(phrase['mode_info'].values())
                        m_start_times = np.array(list(phrase['mode_info'].keys()))
                        cur_mode_idx = 0
                        for i in range(num_of_notes):
                            mode_idx = np.nonzero(phrase['cy_note_starts'][i] >= m_start_times)[0][-1]
                            if mode_idx == cur_mode_idx:
                                freqs.append(ns.next_chord((reg_min[i], reg_max[i])))
                            else:
                                cur_mode_idx = mode_idx
                                mode = modes[mode_idx]
                                mode_size = np.random.choice([4, 5, 6, 7])
                                sub_mode = get_sub_mode(mode, mode_size)
                                cs_min = 2
                                cs_max = np.round((len(sub_mode) - cs_min) + cs_min)
                                if cs_max == cs_min: cs_max = cs_min + 1
                                cs = np.arange(cs_min, cs_max).astype(int)
                                ns = Note_Stream(sub_mode, self.piece.fund, chord_sizes=cs)
                                freqs.append(ns.next_chord((reg_min[i], reg_max[i])))
                    phrase['freqs'] = freqs
                    
                    uni = np.random.uniform(-1, 1, size=num_of_notes)
                    decay = decay_ctrs[g] * (2 ** (uni * decay_ctr_bws[g]))
                    inner_bws = np.random.uniform(0, 1, size=num_of_notes)
                    output_decays = []
                    for i in range(num_of_notes):
                        cs = len(phrase['freqs'][i])
                        mult = 2 ** np.random.uniform(0, 2, size=cs) * inner_bws[i]
                        inner_decays = decay[i] * mult * phrase['output_rt_durs'][i]
                        output_decays.append(inner_decays)
                    
                    phrase['decays'] = output_decays
                    
                    amps = []
                    for i in range(num_of_notes):
                        cs = len(phrase['freqs'][i])
                        this_levels = (np.clip(spread(levels[g][k], 2), 0, 1) for k in range(2))
                        this_levels = tuple(this_levels)
                        klank_amps = np.linspace(this_levels[0], this_levels[1], cs)
                        np.random.shuffle(klank_amps)
                        amps.append(klank_amps)
                        
                    phrase['amps'] = amps
                    
                    uni = np.random.uniform(-1, 1, size=num_of_notes)
                    pan = np.clip(uni * pan_bws[g] + pan_centers[g], -1, 1)
                    phrase['pan'] = pan
                    
                    
                    uni = np.random.uniform(-1, 1, size=num_of_notes)
                    transient_dur = transient_durs[g] * 2 ** (transient_dur_bws[g] * uni)
                    phrase['transient_dur'] = transient_dur
                    
                    phrase['transient_curve'] = transient_curves[g]
                    
                    
                    
                    
                    
                cur_mode = mode    
                
    def make_packets(self):
        self.packets = []
        running_total = 0
        for phrase in self.all_phrases:
            if phrase['type'] == 'phrase':
                for i in range(len(phrase['freqs'])):
                    packet = {
                        'freqs': phrase['freqs'][i],
                        'amps': phrase['amps'][i],
                        'pan': phrase['pan'][i],
                        'rt_dur': phrase['output_rt_durs'][i],
                        'rt_decays': phrase['decays'][i],
                        'transient_dur': phrase['transient_dur'][i],
                        'transient_curve': phrase['transient_curve'],    
                    }
                    running_total += packet['rt_dur']
                    self.packets.append(packet)
            elif phrase['type'] == 'rest':
                get_time = lambda x: self.piece.time.real_time_from_cycles(x)
                rt_start = get_time(phrase['cy_start'])
                rt_end = get_time(phrase['cy_start']+phrase['cy_dur_tot'])
                rt_dur = rt_end - rt_start
                packet = {
                'freqs': 100,
                'amps': 0.5,
                'pan': 0,
                'amps': 0.5,
                'transient_dur': 0.1,
                'transient_curve': 0,
                'rt_decays': [1],
                'rt_dur': 'Rest('+str(rt_dur)+')',
                'rt_start': rt_start
                
                }
                running_total += rt_dur
                self.packets.append(packet)
                
    def save_packets(self):
        json.dump(self.packets, open('JSON/klank_packets_alt.JSON', 'w'), cls=h_tools.NpEncoder)
        
        
                    
            
            
        
            
        
    def get_modes(self, cy_start, cy_dur_tot):
        cy_end = cy_start + cy_dur_tot
        ct_event_map = self.piece.time.ct_event_map
        keys = list(ct_event_map.keys())
        cy_em_starts = np.array(keys)
        # breakpoint()
        start_idx = np.nonzero(cy_start >= cy_em_starts)[0][-1]
        end_idx = np.nonzero(cy_end >= cy_em_starts)[0][-1]
        mode_timings = {}
        for idx in range(start_idx, end_idx+1):
            ev = ct_event_map[keys[idx]]
            mode = self.piece.modes[ev['variation']][ev['mode']]
            mode_timings[keys[idx]] = mode
        return mode_timings
            
        
        
            
        
        
        
        
    


def split_into_groups(num_of_items, num_of_groups, max_group_size=2):
    groups = np.zeros(num_of_groups, dtype=int) + 1
    for i in range(num_of_items - num_of_groups):
        group_choice = np.random.randint(num_of_groups)
        while groups[group_choice] >= max_group_size:
            group_choice = np.random.randint(num_of_groups)
        groups[group_choice] += 1
    return groups
    
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
                
                
