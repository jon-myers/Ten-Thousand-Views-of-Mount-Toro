from tempo_curve import Time
from mode_generation import make_mode_sequence, make_melody
import numpy as np


class Piece:

    def __init__(self, time):
        self.time = time
        self.noc = time.noc
        self.nos = time.nos
        self.sections = [Section(i, self.time) for i in range(self.nos)]
        self.cycles = [Cycle(i, time, self.sections) for i in range(self.noc)]
        for section in self.sections:
            section.cycles = self.cycles
        self.assign_params()
    
    def assign_params(self):
        for section in self.sections:
            packet = np.random.uniform(size=8)
            section.param_packet = packet




class Cycle:

    def __init__(self, cycle_num, time, sections):
        self.time = time
        self.cycle_num = cycle_num
        self.sections = sections
        # self.starting_tempo = time.mm_from_cycles(self.cycle_num)
        # self.ending_tempo - time.mm_from_cycles(self.cycle_num + 1)
        self.section_starts = time.cycle_starts + self.cycle_num
        self.section_ends = time.cycle_ends + self.cycle_num
        self.assign_irama()
    
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
        
        # irama can either be a given number level, or a transition.
        




class Section:

    def __init__(self, section_num, time):
        self.section_num = section_num
        self.time = time
        self.irama = []
        # self.cycles is assigned in Piece __init__, line 14


    def real_dur(self, cycle_num):
        cy_start = self.time.cycle_starts[self.section_num] + cycle_num
        cy_end = self.time.cycle_ends[self.section_num] + cycle_num
        real_start = self.time.real_time_from_cycles(cy_start)
        real_end = self.time.real_time_from_cycles(cy_end)
        real_dur = real_end - real_start
        return real_dur
        

noc = 7
dur_tot = 29*60
fund = 150
modes, variations = make_mode_sequence((10, 20))
melody = make_melody(modes, variations)
events_per_cycle = len(modes)
t = Time(dur_tot=dur_tot, f=0.3, noc=noc)
t.set_cycle(len(modes))
piece = Piece(t)

cy = piece.cycles[0]
sec = piece.sections[0]
print(cy.irama)
irama_lens = [np.size(i) for i in cy.irama]
sec_index = irama_lens.index(2) - 1
real_dur = piece.sections[sec_index].real_dur(0)
print(real_dur)

# for section in piece.sections:
#     print(section.param_packet)
# for cy in piece.cycles:
    
