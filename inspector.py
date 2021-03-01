import json
from fractions import Fraction
modes = json.load(open('modes.json', 'r'))
mode = modes[8]

for m, mode in enumerate(modes):
    print('mode # ', str(m), '\n')
    fracs = [Fraction(i/mode[0]).limit_denominator(1000) for i in mode]
    for f in fracs:
        print(f)
    print('\n\n')
