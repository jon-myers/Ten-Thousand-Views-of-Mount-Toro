import os, re, sys, json, glob
from os.path import exists
from pydub import AudioSegment

mp = json.load(open('JSON/meta_params.JSON'))

audio_files = glob.glob("../audioGeneration/*")
af_nums = list(set([int(re.sub('\D', '', af)) for af in audio_files]))
af_nums.sort()
if len(af_nums) > 0: 
    af_min = af_nums[0]
else:
    af_min = 0


i = af_min
while i < (af_min + 100):
    f, dur, cycles, chords = mp[i]
    sub_path = '../audioGeneration/' + str(i)
    wav_path = sub_path + '.wav'
    mp3_path = sub_path + '.mp3'
    
    if os.path.exists(wav_path):
        os.system("rm " + wav_path)
    if os.path.exists(mp3_path):
        os.system("rm " + mp3_path)

    os.system("python3 main.py " + str(f) + ' ' + str(dur) + ' ' + str(cycles) + ' ' + str(chords))
    os.system("sclang sc/nrt_all.scd " + str(dur + 30) + ' ' + str(i))
    audio = AudioSegment.from_wav(wav_path)
    audio.export(mp3_path, format='mp3')
    
    if os.path.exists(wav_path) and os.path.exists(mp3_path):
        i += 1
    
