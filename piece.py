# from tempo_curve import Time
from rhythm_tools import Time
from rhythm_tools import rhythmic_sequence_maker as rsm
from mode_generation import make_mode_sequence, make_melody
import numpy as np
from instruments import Pluck, MovingPluck
import json
from harmony_tools import utils as h_tools
import pickle
from popcorn import Timespan
from klank import Klank_alt

class Piece:

    def __init__(self, time, modes, fund):
        self.time = time
        self.modes = modes
        self.fund = fund
        self.noc = time.noc
        self.nos = time.nos
        self.consolidate_em()
        self.melody = make_melody(modes[0], modes[1:])
        self.sections = [Section(i, self) for i in range(self.nos)]
        for i, section in enumerate(self.sections):
            section.melody_note = self.fund * modes[self.melody[i][0], i, self.melody[i][1]]
        self.cycles = []
        for i in range(self.noc):
            cy = Cycle(i, self.time, self.sections, self)
            self.cycles.append(cy)
        for section in self.sections:
            section.cycles = self.cycles
        self.assign_params()
        for cycle in self.cycles:
            cycle.assign_instances()
        self.offsets = np.random.random(size=(self.nos, 6))
        self.get_irama_transitions()
        for cycle in self.cycles:
            for instance in cycle.instances:
                instance.make_plucks()
        self.assess_chord_substitutions()
        # self.compile_plucks()
        # self.format_plucks_JSON()
        self.klanks = Klank_alt(self)
        self.make_moving_plucks()
        self.make_popcorn()

    def make_sympathetics(self):
        regions = self.consolidated_em
        min_freq = 200
        max_freq = 800
        self.symp_packets = []
        for i, r in enumerate(regions):
            var_idx = r['variation']
            mode_idx = r['mode']
            mode = self.modes[var_idx][mode_idx]
            freqs = mode * self.fund
            symp_freqs = np.array([])
            for f in freqs:
                low_exp = np.ceil(np.log2(min_freq/f)).astype(int)
                high_exp = np.floor(np.log2(max_freq/f)).astype(int)
                exp = 2.0 ** np.arange(low_exp, high_exp+1)
                symp_freqs = np.concatenate((symp_freqs, f*exp))
            cy_start = r['cy_start']
            rt_start = self.time.real_time_from_cycles(cy_start)
            if i == len(regions)-1:
                rt_end = self.time.dur_tot
            else:
                cy_end = regions[i+1]['cy_start']
                rt_end = self.time.real_time_from_cycles(cy_end)
            rt_dur = rt_end - rt_start
            obj = {'freqs': symp_freqs, 'dur': rt_dur}
            self.symp_packets.append(obj)
        path = 'JSON/sympathetics.JSON'
        json.dump(self.symp_packets, open(path, 'w'), cls=h_tools.NpEncoder)










    def make_popcorn(self):
        cy_durs = [s.cy_end - s.cy_start for s in self.sections]
        # init_rt_durs = [s.real_dur(0) for s in self.sections]
        # ^ this gets altered real time, nead standardized real time, non bendy instead...
        # going to have to be an array of 4 item arrays, for each of the possible iramas
        # which all would have different non bendy real times ... oy!
        # TODO ^
        init_rt_density = 0.4
        # init_cy_density = init_rt_density * init_rt_durs[0] / cy_durs[0]
        init_cy_densities = init_rt_density * self.rt_cy_props
        cy_densities = rsm(self.nos, 40) * self.nos
        cy_densities = np.broadcast_to(cy_densities, (4, len(cy_densities)))
        cy_densities = cy_densities * init_cy_densities.reshape((4, 1))
        vol_offsets = rsm(self.nos, 40) * self.nos
        onset_nCVIs = 40 * rsm(self.nos, 40) * self.nos
        vol_dist_nCVIs = 40 * rsm(self.nos, 40) * self.nos
        all_vol_dist_vals = []
        for i in range(self.nos):
            vol_dist_vals = rsm(3, vol_dist_nCVIs[i])
            vol_dist_vals /= np.prod(vol_dist_vals) ** (1/3)
            all_vol_dist_vals.append(vol_dist_vals)
        init_ctr_freq = 100
        # init_ctr_freq_log2 = np.log2(init_ctr_freq)
        ctr_freqs = rsm(self.nos, 40) * self.nos * init_ctr_freq
        ctr_freqs_log2 = np.log2(ctr_freqs)
        ctr_freq_bws = rsm(self.nos, 40) * self.nos
        max_freq_oct_bws = rsm(self.nos, 60) * self.nos * 1.3
        nCVI_amps = rsm(self.nos, 30) * self.nos * 30
        nCVI_durs = rsm(self.nos, 30) * self.nos * 30
        nCVI_bws = rsm(self.nos, 30) * self.nos * 30
        attack_avgs = rsm(self.nos, 40) * self.nos * 0.01
        attack_avg_max_bws = rsm(self.nos, 40) * self.nos * 1.5

        pan_ctr_gamut = np.linspace(-0.8, 0.8, 5)
        pan_ctr_starts = h_tools.dc_alg(5, self.nos, alpha=2)
        pan_ctr_starts = pan_ctr_gamut[pan_ctr_starts]
        pan_ctr_ends = h_tools.dc_alg(5, self.nos, alpha=2)
        pan_ctr_ends = pan_ctr_gamut[pan_ctr_ends]

        pan_bw_gamut = np.linspace(0, 0.3, 5)
        pan_bws = h_tools.dc_alg(5, self.nos, alpha=2)
        pan_bws = pan_bw_gamut[pan_bws]

        rest_prop_gamut = np.linspace(0.1, 0.5, 5)
        rest_props = h_tools.dc_alg(5, self.nos, alpha=2)
        rest_props = rest_prop_gamut[rest_props]

        rest_spread_gamut = np.linspace(0, 1, 5)
        rest_spreads = h_tools.dc_alg(5, self.nos, alpha=2)
        rest_spreads = rest_spread_gamut[rest_spreads]

        rest_nCVI_gamut = np.linspace(20, 60, 5)
        rest_nCVIs = h_tools.dc_alg(5, self.nos, alpha=2)
        rest_nCVIs = rest_nCVI_gamut[rest_nCVIs]


        self.timespans = []
        # breakpoint()
        for c in range(self.noc):
            print('cycle: ', c)
            cy_timespans = []
            for s in range(self.nos):
                print('section:', s)
                irama = self.cycles[c].irama[s]
                if np.size(irama) == 2:
                    irama = irama[0]
                sec_timespans = []
                cy_start_time = self.cycles[c].section_starts[s]
                for i in range(irama+1):
                    # breakpoint()
                    ts = Timespan(
                        cy_durs[s], cy_densities[i][s], vol_offsets[s], onset_nCVIs[s],
                        all_vol_dist_vals[s], ctr_freqs_log2[s] + i, ctr_freq_bws[s],
                        max_freq_oct_bws[s], nCVI_amps[s], nCVI_durs[s], nCVI_bws[s],
                        attack_avgs[s], attack_avg_max_bws[s], i, cy_start_time,
                        pan_ctr_starts[s], pan_ctr_ends[s], pan_bws[s] * 2 ** (i/3),
                        rest_props[s] * 2 ** (i/3), rest_spreads[s], rest_nCVIs[s])
                    ts.build()
                    sec_timespans.append(ts)
                cy_timespans.append(sec_timespans)
            self.timespans.append(cy_timespans)

        self.accumulate_kernals()
        self.save_popcorn_to_json()
        # breakpoint()


    def accumulate_kernals(self):
        # get into single list, then add rt onset time tag, then sort.
        self.kernals = []
        for ts in self.timespans:
            for sts in ts:
                for timespan in sts:
                    for k in timespan.kernals:
                        self.kernals.append(k)
        self.kernals.sort(key=lambda x: x['cy_onset_time'])
        for k in self.kernals:
            k['rt_onset_time'] = self.time.real_time_from_cycles(k['cy_onset_time'])
            rt_end = self.time.real_time_from_cycles(k['dur'] + k['cy_onset_time'])
            k['rt_dur'] = rt_end - k['rt_onset_time']

    def save_popcorn_to_json(self):
        path = 'JSON/kernals.JSON'
        json.dump(self.kernals, open(path, 'w'), cls=h_tools.NpEncoder)




    def consolidate_em(self):
        em = self.time.event_map
        self.consolidated_em = []
        for i in list(em.keys()):
            item = em[i]
            # breakpoint()
            for k in list(item.keys()):
                new_cy_start = i+k
                new_obj = item[k].copy()
                new_obj['cy_start'] = new_cy_start
                # breakpoint()
                self.consolidated_em.append(new_obj)
        self.cy_mode_transitions = np.array([i['cy_start'] for i in self.consolidated_em])
        self.cy_mode_transitions = np.append(self.cy_mode_transitions, self.noc)


    def assess_chord_substitutions(self):
        self.cs_matrix = np.zeros((self.nos, self.noc))
        for s, section in enumerate(self.sections):
            for i, instance in enumerate(section.instances):
                self.cs_matrix[s, i] = len(instance.event_map)


    def save_melody_JSON(self):
        """Stores melody notes and seciton timings (stored in sections) as
        json, to be used by Supercollider. """

        self.melody_packets = []
        for c, cycle in enumerate(self.cycles):
            for s, section in enumerate(cycle.sections):
                packet = {}
                packet['note'] = section.melody_note
                # packet['rt_start'] = section.rt_starts[c]
                packet['rt_dur'] = section.rt_durs[c]
                self.melody_packets.append(packet)
        json.dump(self.melody_packets, open('JSON/melody.JSON', 'w'),
                    cls=h_tools.NpEncoder)
        # print(self.offsets)

    def assign_params(self):
        for section in self.sections:
            packet = np.random.uniform(size=8)
            section.param_packet = packet

    def compile_plucks(self):
        self.all_plucks = []
        for cycle in self.cycles:
            for instance in cycle.instances:
                for plucks in instance.plucks:
                    for pluck in plucks:
                        if pluck['dur'] != 0:
                            self.all_plucks.append(pluck)

    def format_plucks_JSON(self):
        """Takes the all_plucks list, which consists of each item having an
        object that contains all of its parameters, to an object wherein each
        parameter has a list for all events, easier for getting into PBinds in
        Supercollider."""
        self.json_plucks = {}
        self.json_plucks['freqs'] = []
        self.json_plucks['coef'] = []
        self.json_plucks['decay'] = []
        self.json_plucks['delays'] = []
        self.json_plucks['vol'] = []
        self.json_plucks['rt_start'] = [] # might not need this
        self.json_plucks['rt_end'] = [] # might not need this
        self.json_plucks['rt_dur'] = []

        for pluck in self.all_plucks:
            # breakpoint()
            self.json_plucks['freqs'].append(pluck['freqs'])
            self.json_plucks['coef'].append(pluck['coef'])
            self.json_plucks['decay'].append(pluck['decay'])
            self.json_plucks['delays'].append(pluck['delays'])
            self.json_plucks['vol'].append(pluck['vol'])
            self.json_plucks['rt_start'].append(pluck['rt_start'])
            self.json_plucks['rt_end'].append(pluck['rt_end'])
            self.json_plucks['rt_dur'].append(pluck['rt_dur'])

        json.dump(self.json_plucks, open('JSON/all_plucks.JSON', 'w'),
                    cls=h_tools.NpEncoder)


    def get_irama_transitions(self):
        self.irama_transitions = []
        for c in range(self.noc):
            for s in range(self.nos):
                ir = self.cycles[c].irama[s]
                if np.size(ir) == 2:
                    self.irama_transitions.append((c, s))

        # also, gonna assign the proportion between an entire period of a
        # given irama with real time, for figuring out rhythmic things that will
        # be calculated "flat", that will ultamitely get transformed to slowing.
        cy_starts = [self.time.cycles_from_mm(2 ** -i) for i in range(4)]
        cy_ends = [self.time.cycles_from_mm(2 ** -i) for i in range(1, 5)]
        cy_durs = [cy_ends[i] - cy_starts[i] for i in range(4)]
        rt_starts = [self.time.real_time_from_cycles(i) for i in cy_starts]
        rt_ends = [self.time.real_time_from_cycles(i) for i in cy_ends]
        rt_durs = [rt_ends[i] - rt_starts[i] for i in range(4)]
        self.rt_cy_props = np.array([rt_durs[i] / cy_durs[i] for i in range(4)])
        return self.irama_transitions

    def make_klanks(self):
        self.klank_packets = []
        # start with just the first one, for irama 0
        fine_tuning = np.random.random(size=3)
        for i in range(4):
            klank = Klank(self, i, fine_tuning)
            self.klank_packets += klank.packets
        file = open('json/klank_packets.JSON', 'w')
        json.dump(self.klank_packets, file, cls=h_tools.NpEncoder)

    def make_moving_plucks(self):
        self.mp_packets = []
        self.mp = MovingPluck(self)
            # self.mp_packets += mp.packets


class Cycle:

    def __init__(self, cycle_num, time, sections, piece):
        self.time = time
        self.cycle_num = cycle_num
        self.sections = sections
        self.piece = piece
        # self.starting_tempo = time.mm_from_cycles(self.cycle_num)
        # self.ending_tempo - time.mm_from_cycles(self.cycle_num + 1)
        self.section_starts = time.cycle_starts + self.cycle_num
        self.section_ends = time.cycle_ends + self.cycle_num
        self.assign_irama()
        self.event_map = self.time.event_map[self.cycle_num]


    def assign_irama(self):
        starting_tempi = np.array([self.time.mm_from_cycles(i) for i in self.section_starts])
        ending_tempi = np.array([self.time.mm_from_cycles(i) for i in self.section_ends])
        starting_irama = np.floor(-1 * np.log2(starting_tempi)).astype(int)
        ending_irama = np.floor(-1 * np.log2(ending_tempi)).astype(int)
        self.irama = []
        for i in range(len(starting_irama)):
            if starting_irama[i] == ending_irama[i]:
                ir = starting_irama[i]
            else:
                ir = (starting_irama[i], ending_irama[i])
            self.irama.append(ir)
            self.sections[i].irama.append(ir)


    def assign_instances(self):
        self.instances = []
        for i in range(len(self.sections)):
            instance = Instance(self.cycle_num, i, self.piece)
            self.instances.append(instance)
            self.sections[i].instances.append(instance)
    # def realize_pluck(self):




class Section:

    def __init__(self, section_num, piece):
        self.section_num = section_num
        self.piece = piece
        self.time = self.piece.time
        self.irama = []
        self.instances = []
        self.cy_start = self.time.cycle_starts[self.section_num]
        self.cy_end = self.time.cycle_ends[self.section_num]
        rtfc = self.time.real_time_from_cycles
        self.rt_starts = [rtfc(self.cy_start+i) for i in range(self.piece.noc)]
        self.rt_ends = [rtfc(self.cy_end+i) for i in range(self.piece.noc)]
        self.rt_durs = [self.rt_ends[i]-self.rt_starts[i] for i in range(self.piece.noc)]

        # self.cycles is assigned in Piece __init__, line 14


    def real_dur(self, cycle_num):
        real_start = self.time.real_time_from_cycles(self.cy_start+cycle_num)
        real_end = self.time.real_time_from_cycles(self.cy_end+cycle_num)
        real_dur = real_end - real_start
        return real_dur

class Instance:
    """A class for a particular instance of a section in a cycle."""

    def __init__(self, cycle_num, section_num, piece):
        self.cycle_num = cycle_num
        self.section_num = section_num
        self.piece = piece
        self.time = self.piece.time
        self.cycle = self.piece.cycles[self.cycle_num]
        self.section = self.piece.sections[self.section_num]
        self.irama = self.cycle.irama[self.section_num]
        self.get_event_map()
        self.get_real_durs()
        self.rt_start = self.section.rt_starts[self.cycle_num]
        self.rt_dur = self.section.rt_durs[self.cycle_num]

    def get_event_map(self):
        em = self.cycle.event_map
        keys = self.cycle.event_map.keys()
        self.event_map = []
        for index, key in enumerate(keys):
            if em[key]['mode'] == self.section_num:
                obj = em[key]
                obj['start'] = key
                if index != len(keys) - 1:
                    obj['end'] = list(keys)[index+1]
                else:
                    obj['end'] = 1
                obj['dur'] = obj['end'] - obj['start']
                self.event_map.append(obj)

    def get_real_durs(self):
        """Get real durs of all segments in instance."""
        self.real_durs = []
        for obj in self.event_map:
            cy_start = obj['start'] + self.cycle_num
            cy_end = obj['end'] + self.cycle_num
            real_start = self.time.real_time_from_cycles(cy_start)
            real_end = self.time.real_time_from_cycles(cy_end)
            real_dur = real_end - real_start
            self.real_durs.append(real_dur)

    def make_plucks(self):

        """


        (note that, throughout all this, 'plucks' objects describe time from 0-1
        for their own entirety, not . )"""
        self.plucks = []
        for i, obj in enumerate(self.event_map):
            real_dur = self.real_durs[i]
            offsets = self.piece.offsets[self.section_num]
            variation = obj['variation']
            seg = obj['mode']
            mode = self.piece.modes[variation, seg]
            fund = self.piece.fund
            ct_section_start = obj['start'] + self.cycle_num
            rt_section_start = self.time.real_time_from_cycles(ct_section_start)
            if i > 0:
                prev_obj = self.event_map[i-1]
                ct_last_event = self.plucks[-1][-1]['start'] * prev_obj['dur'] + prev_obj['start'] + self.cycle_num
                rt_last_event = self.time.real_time_from_cycles(ct_last_event)
                rt_since_last = rt_section_start - rt_last_event
            elif self.section_num == 0:
                if self.cycle_num == 0:
                    rt_since_last = 10000 # doesn't need to be this big, but
                                          # anything biggish should suffice
                else:
                    prev_cy = self.piece.cycles[self.cycle_num-1]
                    prev_obj = prev_cy.instances[-1].event_map[-1]
                    prev_pluck = prev_cy.instances[-1].plucks[-1][-1]
                    ct_last_event = prev_pluck['start'] * prev_obj['dur'] + prev_obj['start'] + self.cycle_num - 1
                    rt_last_event = self.time.real_time_from_cycles(ct_last_event)
                    rt_since_last = rt_section_start - rt_last_event
            else:
                prev_inst = self.piece.cycles[self.cycle_num].instances[self.section_num-1]
                prev_pluck = prev_inst.plucks[-1][-1]
                ct_last_event = prev_pluck['start'] * prev_inst.event_map[-1]['dur'] + prev_inst.event_map[-1]['start'] + self.cycle_num
                rt_last_event = self.time.real_time_from_cycles(ct_last_event)
                rt_since_last = rt_section_start - rt_last_event

            p = Pluck(self.irama, real_dur, offsets, mode, fund, rt_since_last)
            packets = p.render()
            # breakpoint()
            # add in real times
            for packet in packets:
                p_start = packet['start']
                c_start = p_start * obj['dur'] + obj['start'] + self.cycle_num
                rt_start = self.time.real_time_from_cycles(c_start)
                p_end = packet['end']
                c_end = p_end * obj['dur'] + obj['start'] + self.cycle_num
                rt_end = self.time.real_time_from_cycles(c_end)
                packet['rt_start'] = rt_start
                packet['rt_end'] = rt_end
                packet['rt_dur'] = rt_end - rt_start
            self.plucks.append(packets)

def build(
save_piece_pickle=False, use_pickles=False, save_pickles=False, noc=9, dur_tot=1800,
f=0.5, num_of_modes=12):
    # noc = 9
    # dur_tot = 35*60
    fund = 150
    if use_pickles:
        t = pickle.load(open('pickles/t.p', 'rb'))
        modes = pickle.load(open('pickles/modes.p', 'rb'))
    modes = make_mode_sequence((num_of_modes, num_of_modes+1))
    events_per_cycle = np.shape(modes)[1]
    t = Time(dur_tot=dur_tot, f=f, noc=noc)
    t.set_cycle(len(modes[0]))
    if save_pickles:
        pickle.dump(t, open('pickles/t.p', 'wb'))
        pickle.dump(modes, open('pickles/modes.p', 'wb'))
    piece = Piece(t, modes, fund)
    if save_piece_pickle:
        pickle.dump(piece, open('pickles/piece.p', 'wb'))
    return piece

#
# piece = build()
# it = piece.get_irama_transitions()
# print(it)
# rt = piece.time.real_time_from_cycles(it[0][0] + (it[0][1] / piece.nos))
# print(rt)
# pickle.dump(piece, open('pickles/piece.p', 'wb'))
