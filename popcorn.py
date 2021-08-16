import numpy as np
from dataclasses import dataclass, field
from rhythm_tools import rhythmic_sequence_maker as rsm
from harmony_tools import utils as h_tools
import json
from typing import List

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

    cy_dur_tot: float = 60
    cy_kernal_density: float = 2.5
    volume_tot_offset: float = 1.3
    onsets_nCVI: float = 70
    vol_dist_vals: List[float] = field(default_factory=lambda: [1, 1, 1])
    avg_center_freq: float = avg_center_freq
    center_freq_max_bw: float = 1.5
    max_freq_oct_bw: float = 1.5
    nCVI_amp: float = 30
    nCVI_dur: float = 30
    nCVI_bw: float = 30
    attack_avg: float = 0.01
    attack_avg_max_bw: float = 1.8
    irama: int = 0

    def build(self):
        # irama transformations
        self.avg_center_freq += self.irama
        self.cy_kernal_density *= 2 ** self.irama

        self.vol_dist_vals = np.array(self.vol_dist_vals)



        self.num_of_kernals = int(self.cy_dur_tot * self.cy_kernal_density)
        if self.num_of_kernals == 0:
            self.num_of_kernals = 1
        onset_times = rsm(self.num_of_kernals, self.onsets_nCVI, start_times=True) * self.cy_dur_tot
        # max_amp = 1
        # max_dur = 8
        # max_vol_tot = self.dur_tot * self.max_freq_bw
        # # have to figure out standard vals for these thigns, dur should be somehow
        # # assigned externally; otherwise, not sure how to

        # vol_dist_vals = rsm(3, self.volume_distribution_nCVI)
        # vol_dist_vals /= np.prod(vol_dist_vals) ** (1/3)
        # breakpoint()
        (avg_amp, avg_dur, avg_bw) = self.vol_dist_vals * self.volume_tot_offset
        if avg_amp > 2:
            avg_dur *= (avg_amp / 2)
            avg_amp = 2
        if avg_bw > 2:
            avg_dur *= (avg_bw / 2)
            avg_bw = 2

        base_avg_amp = 0.5
        base_avg_dur = 0.8 * self.cy_dur_tot / self.num_of_kernals
        base_avg_bw = self.max_freq_oct_bw / 2

        amps = rsm(self.num_of_kernals, self.nCVI_amp) * base_avg_amp * avg_amp * self.num_of_kernals
        # breakpoint()
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
