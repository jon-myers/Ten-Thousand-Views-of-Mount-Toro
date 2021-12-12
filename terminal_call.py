import os
import sys
import json
from os.path import exists
from pydub import AudioSegment

mp = json.load(open('JSON/meta_params.JSON'))

i = 0
while i < 10:
    f, dur, cycles, chords = mp[i]

    os.system("python3 main.py " + str(f) + ' ' + str(dur) + ' ' + str(cycles) + ' ' + str(chords))
    os.system("sclang sc/nrt_all.scd " + str(dur + 30) + ' ' + str(i))
    path = '../audioGeneration/' + str(i) + '.wav'
    audio = AudioSegment.from_wab(path)
    audio.export(str(i) + '.mp3', formap='mp3')
