import json
import numpy as np
from harmony_tools import utils as h_tools



def melody_from_contour(modes, contour, min_cents=50):
    # start with a random pitch
    start = np.random.choice(modes[0])
    melody = [start]
    for i, c in enumerate(contour):
        mode = modes[(i+1)]
        if c > 0:
            above = mode[mode > h_tools.cents_to_hz(min_cents, melody[-1])]
            if len(above) <= c - 1:
                next = mode[c-1]
            else:
                next = above[c-1]
        elif c < 0:
            below = mode[mode < h_tools.cents_to_hz(-min_cents, melody[-1])]
            if len(below) <= np.abs(c) - 1:
                next = mode[c]
            else:
                next = below[c]
        melody.append(next)
    return melody

modes = np.array(json.load(open('JSON/modes.JSON', 'r')))

while np.any(modes>=2):
    modes = np.where(modes >= 2, modes / 2, modes)
modes = np.sort(modes)

contour = np.random.choice(np.array((-2, -1, 1, 2)), size=len(modes)-1)

melody = melody_from_contour(modes, contour)
melody = np.tile(melody, 15)
json.dump(melody, open('JSON/melody.JSON', 'w'), cls=h_tools.NpEncoder)
