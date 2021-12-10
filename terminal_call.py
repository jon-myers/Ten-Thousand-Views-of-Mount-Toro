import os
import sys
import json

mp = json.load(open('JSON/meta_params.JSON'))
for i in range(5):

    f, dur, cycles, chords = mp[i]


    os.system("python3 main.py " + str(f) + ' ' + str(dur) + ' ' + str(cycles) + ' ' + str(chords))
    os.system("sclang sc/nrt_all.scd " + str(dur + 30) + ' ' + str(i))
