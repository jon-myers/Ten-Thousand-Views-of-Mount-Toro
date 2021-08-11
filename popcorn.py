import numpy as np
from dataclasses import dataclass
from rhythm_tools import rhythmic_sequence_maker as rsm

avg_center_freq = np.log2(400)
attack_avg = np.log2(0.04)
sample_control_spec = {
'dur_tot': 20,
'num_of_kernals': 20,
'volume_tot': 5, # I think, but may have to experiment with these values
'onsets_nCVI': 15,
'volume_distribution_nCVI': 25,
'avg_center_freq': avg_center_freq,
'center_freq_max_bw': 2/3,
'nCVI_vol_factor_weights': 40,
'nCVI_amp': 30,
'nCVI_dur': 20,
'nCVI_bw': 40,
'attack_avg': attack_avg,
'attack_avg_max_bw': 1.5,


}



@dataclass
class Timespan:

    dur_tot: float = 20
    num_of_kernals: int = 20
    volume_tot: float = 5
    onsets_nCVI: float = 15
    volume_distribution_nCVI: float = 25
    avg_center_freq: float = avg_center_freq
    center_freq_max_bw: float = 2/3
    max_freq_oct_bw: float = 2.5
    nCVI_vol_factor_weights: float = 40
    nCVI_amp: float = 30
    nCVI_dur: float = 20
    nCVI_bw: float = 40
    attack_avg: float = attack_avg
    attack_avg_max_bw: float = 1.5

    # def __init__(
    #     self, dur_tot, num_of_kernals, volume_tot, onsets_nCVI, volume_distribution_nCVI,
    #     avg_center_freq, center_freq_max_bw, nCVI_vol_factor_weights, nCVI_amp,
    #     nCVI_dur, nCVI_bw, attack_avg, attack_avg_max_bw
    #     ):



    """
    dur_tot: float, in cy_time,
    start_time: float, in cy_time,
    num_of_kernals: int,
    volume_tot: float,
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
    attack_avg: float, log2 scale, this doesn't get real_time-ified; doesn't
        exist in twisty time.
    attack_avg_max_bw: float, log2 scale, for -1 to 1 clipped normal dist.    """

    def build(self):
        onset_times = rsm(self.num_of_kernals, self.onsets_nCVI, start_times=True) * self.dur_tot
        max_amp = 1
        max_dur = 8
        max_vol_tot = self.dur_tot * self.max_freq_bw
        # have to figure out standard vals for these thigns, dur should be somehow
        # assigned externally; otherwise, not sure how to

        (avg_amp, avg_dur, avg_bw) = rsm(3, sself.volume_distribution_nCVI)



    def return_dur_tot(self):
        return self.dur_tot

ts = Timespan()
dt = ts.return_dur_tot()
print(ts)
