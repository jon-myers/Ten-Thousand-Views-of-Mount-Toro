from scipy.integrate import quad as integrate
from scipy.misc import derivative
from scipy.special import gammainc as gamma
from scipy.optimize import fsolve
import math

class Time:
    """Creates a time object, for a looping cycle (or buffer) with a
    variable speed, fixed accelration, which can be queried to, for example,
    get the phase, tempo-level, and irama at a particular moment in real time.
    """

    def __init__(self, irama_levels=4, dur_tot=1740, z=0.5, f=0.5):
        self.dur_tot = dur_tot
        self.z = z
        self.f = f
        self.norm_factor = self.dur_tot / self.time_from_tempo(1/irama_levels)

    def mm(self, time):
        """Instantaneous tempo at a given time measured from begining of the
        piece."""
        return self.z ** (time ** self.f)

    # def b_alt(self, time):
    #     """Number of beats passed since beginning of piece. Anayltic, rather
    #     than numeric. (in order to get this to match b, I had to remove a
    #     negative sign from the answer provided by integral-calculator.com !?)"""
    #     part = -math.log(self.z) * (time ** self.f)
    #     return gamma(1/self.f, part) * time / (self.f * (part ** (1 / self.f)))

    def b(self, time):
        """Number of beats passed since beginning of piece."""
        return integrate(self.mm, 0, time)[0]

    def acc(self, time):
        """Instantaneous acceleration at a given time, measured from the
        beginning of the piece."""
        if time == 0:
            return -math.inf
        else:
            return self.mm(time) * math.log(self.z) * self.f * (time ** (self.f - 1))

    def time_from_tempo(self, tempo):
        """Returns the time when a given instantaneous tempo occurs."""
        return math.e ** (math.log(math.log(tempo, self.z)) / self.f)

    def time_from_beat(self, beat):
        """Returns the time when a given number of beats have elapsed."""
        return fsolve(lambda x: self.b(x) - beat, 0.5)
