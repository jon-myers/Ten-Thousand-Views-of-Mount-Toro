import os
import sys
import json
<<<<<<< HEAD
from os.path import exists

mp = json.load(open('JSON/meta_params.JSON'))

i = 0
while i < 10:
=======

mp = json.load(open('JSON/meta_params.JSON'))
for i in range(1):
>>>>>>> afb91b3 (generating from meta_params)

    f, dur, cycles, chords = mp[i]


<<<<<<< HEAD
    os.system("python3 main.py " + str(f) + ' ' + str(dur) + ' ' + str(cycles) + ' ' + str(chords))
    os.system("sclang sc/nrt_all.scd " + str(dur + 30) + ' ' + str(i))
    path = '../audioGeneration/' + str(i) + '.wav'
    if exists(path):
        i += 1
=======
    os.system("python main.py " + str(f) + ' ' + str(dur) + ' ' + str(cycles) + ' ' + str(chords))
    os.system("sclang sc/nrt_all.scd " + str(dur + 30) + ' ' + str(i))
>>>>>>> afb91b3 (generating from meta_params)
