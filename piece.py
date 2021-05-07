from tempo_curve import Time
from mode_generation import make_mode_sequence, make_melody


class Piece:

    def __init__(self, time):
        self.time = time
        self.noc = time.noc
        self.nos = time.nos
        self.sections = [Section(i, self.time) for i in range(self.nos)]
        self.cycles = [Cycle(i, time, self.sections) for i in range(self.noc)]
        for section in self.sections:
            section.cycles = self.cycles




class Cycle:

    def __init__(self, cycle_num, time, sections):
        self.time = time
        self.cycle_num = cycle_num
        self.sections = sections
        
        self.starting_tempo = time.mm_from_cycles(self.cycle_num)




class Section:

    def __init__(self, section_num, time):
        self.section_num = section_num
        self.time = time
        # self.cycles is assigned in Piece __init__, line 14




noc = 7
dur_tot = 29*60
fund = 150
modes, variations = make_mode_sequence((10, 20))
melody = make_melody(modes, variations)
events_per_cycle = len(modes)
t = Time(dur_tot=dur_tot, f=0.3, noc=noc)
t.set_cycle(len(modes))

piece = Piece(t)
for cyc in piece.cycles:
    print(cyc.starting_tempo)
