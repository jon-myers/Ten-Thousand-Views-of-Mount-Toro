# from tempo_curve import Time
from rhythm_tools import Time
from mode_generation import make_mode_sequence, make_melody
import numpy as np
from instruments import Pluck
import json
from harmony_tools import utils as h_tools


class Piece:

    def __init__(self, time, modes, fund):
        self.time = time
        self.modes = modes
        self.fund = fund
        self.noc = time.noc
        self.nos = time.nos
        self.sections = [Section(i, self) for i in range(self.nos)]
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
        for cycle in self.cycles:
            for instance in cycle.instances:
                instance.make_plucks()
        self.compile_plucks()
        self.format_plucks_JSON()

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
        return self.irama_transitions



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

        # self.cycles is assigned in Piece __init__, line 14


    def real_dur(self, cycle_num):
        cy_start = self.time.cycle_starts[self.section_num] + cycle_num
        cy_end = self.time.cycle_ends[self.section_num] + cycle_num
        real_start = self.time.real_time_from_cycles(cy_start)
        real_end = self.time.real_time_from_cycles(cy_end)
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










noc = 7
dur_tot = 29*60
fund = 150
modes = make_mode_sequence((10, 20)) # a single np array with three columns,
                                     # (modes, variation_0, variation_1)
melody = make_melody(modes[0], modes[1:])
events_per_cycle = np.shape(modes)[1]
t = Time(dur_tot=dur_tot, f=0.5, noc=noc)
t.set_cycle(len(modes[0]))

# print(t.event_map[0])
# print(len(t.event_map.keys()))
piece = Piece(t, modes, fund)

it = piece.get_irama_transitions()
print(it)
print(melody)
# sec_last = piece.sections[-1]

# print(piece.all_plucks)
# json.dump(piece.all_plucks, open('JSON/all_plucks.JSON', 'w'), cls=h_tools.NPEncoder)
