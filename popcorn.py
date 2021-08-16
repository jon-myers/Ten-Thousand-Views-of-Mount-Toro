import numpy as np
from dataclasses import dataclass
from rhythm_tools import rhythmic_sequence_maker as rsm
from harmony_tools import utils as h_tools
import json

avg_center_freq = np.log2(800)
attack_avg = np.log2(0.04)
# sample_control_spec = {
# 'dur_tot': 20,
# 'num_of_kernals': 20,
# 'volume_tot_offset': 5, # I think, but may have to experiment with these values
# 'onsets_nCVI': 15,
# 'volume_distribution_nCVI': 25,
# 'avg_center_freq': avg_center_freq,
# 'center_freq_max_bw': 1.5,
# 'nCVI_vol_factor_weights': 40,
# 'nCVI_amp': 30,
# 'nCVI_dur': 20,
# 'nCVI_bw': 40,
# 'attack_avg': attack_avg,
# 'attack_avg_max_bw': 1.5,
# }


def get_clipped_normal(num, lo=-1, hi=1):
    return np.clip(np.random.normal(0, 1/3, num), lo, hi)

@dataclass
class Timespan:

    dur_tot: float = 20
    num_of_kernals: int = 20
    volume_tot_offset: float = 1.2
    onsets_nCVI: float = 70
    volume_distribution_nCVI: float = 60
    avg_center_freq: float = avg_center_freq
    center_freq_max_bw: float = 1.5
    max_freq_oct_bw: float = 1.5
    nCVI_vol_factor_weights: float = 40
    nCVI_amp: float = 30
    nCVI_dur: float = 30
    nCVI_bw: float = 30
    attack_avg: float = 0.01
    attack_avg_max_bw: float = 1.8

    # def __init__(
    #     self, dur_tot, num_of_kernals, volume_tot, onsets_nCVI, volume_distribution_nCVI,
    #     avg_center_freq, center_freq_max_bw, nCVI_vol_factor_weights, nCVI_amp,
    #     nCVI_dur, nCVI_bw, attack_avg, attack_avg_max_bw
    #     ):



    """
    dur_tot: float, in cy_time,
    start_time: float, in cy_time,
    num_of_kernals: int,
    volume_tot_offset: float,
    onsets_nCVI: float,
    volume_distribution_nCVI: float,
    avg_center_freq: float, log scale,
    center_freq_max_bw: float, log scale, _max_ max is 1, meaning an octave,
        for -1 to 1 clipped normal dist.
    nCVI_vol_factor_weights: float, (not really sequential, but using nCVI
        because it sums to one, and won't have outliers),
    nCVI_amp: float,
    nCVI_dur: float,
    nCVI_bw: float,
    attack_avg: float, reg time scale, this doesn't get real_time-ified; doesn't
        exist in twisty time.
    attack_avg_max_bw: float, log2 scale, for -1 to 1 clipped normal dist.    """

    def build(self):
        onset_times = rsm(self.num_of_kernals, self.onsets_nCVI, start_times=True) * self.dur_tot
        # max_amp = 1
        # max_dur = 8
        # max_vol_tot = self.dur_tot * self.max_freq_bw
        # # have to figure out standard vals for these thigns, dur should be somehow
        # # assigned externally; otherwise, not sure how to
        
        vol_dist_vals = rsm(3, self.volume_distribution_nCVI)
        vol_dist_vals /= np.prod(vol_dist_vals)** (1/3)
        (avg_amp, avg_dur, avg_bw) = vol_dist_vals * self.volume_tot_offset
        if avg_amp > 2:
            avg_dur *= (avg_amp / 2)
            avg_amp = 2
        if avg_bw > 2:
            avg_dur *= (avg_bw / 2)
            avg_bw = 2
        
        base_avg_amp = 0.5
        base_avg_dur = 0.8 * self.dur_tot / self.num_of_kernals
        base_avg_bw = self.max_freq_oct_bw / 2
        
        amps = rsm(self.num_of_kernals, self.nCVI_amp) * base_avg_amp * avg_amp * self.num_of_kernals
        durs = rsm(self.num_of_kernals, self.nCVI_dur) * base_avg_dur * avg_dur * self.num_of_kernals
        bws = rsm(self.num_of_kernals, self.nCVI_bw) * base_avg_bw * avg_bw * self.num_of_kernals
        
        attack_mults = get_clipped_normal(self.num_of_kernals) * self.attack_avg_max_bw
        attacks = self.attack_avg * (2 ** attack_mults)
        
        freq_ctr_mults = get_clipped_normal(self.num_of_kernals) * self.center_freq_max_bw
        freq_ctrs = avg_center_freq + freq_ctr_mults
        
        highs = 2 ** (freq_ctrs + bws)
        lows = 2 ** (freq_ctrs - bws)
        
        kernals = []
        for i in range(self.num_of_kernals):
            kernal = {}
            kernal['hp_freq'] = lows[i]
            kernal['lp_freq'] = highs[i]
            kernal['attack'] = attacks[i]
            kernal['dur'] = durs[i]
            kernal['amp'] = amps[i]
            kernal['onset'] = onset_times[i]   
            kernals.append(kernal)
        
        self.kernals = kernals
        
    def save_kernals(self, path):
        json.dump(self.kernals, open(path, 'w'), cls=h_tools.NpEncoder)
        

        


ts = Timespan()
ts.build()
ts.save_kernals('JSON/kernals.JSON')
