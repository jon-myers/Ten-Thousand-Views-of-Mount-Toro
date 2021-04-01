class Time:
    """Creates a time object, for a looping cycle (or buffer) with a
    variable speed, fixed accelration, which can be queried to, for example,
    get the phase, tempo-level, and irama at a particular moment in real time.
    """

    def __init__(self, m_dur=10, init_tempo=1, acceleration=1):
        
