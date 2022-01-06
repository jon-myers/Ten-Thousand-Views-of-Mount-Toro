import os, re, sys, json, glob
from os.path import exists

def make_view(i, increment=True):
    print('Generating view no. ' + str(i) + '!')
    f, dur, cycles, chords = mp[i]
    sub_path = '../audioGeneration/' + str(i)
    wav_path = sub_path + '.wav'
    if os.path.exists(wav_path):
        os.system("rm " + wav_path)
    result = False
    while not result:
        os.system("python3 main.py " + str(f) + ' ' + str(dur) + ' ' + str(cycles) + ' ' + str(chords))
        result = test_JSON()
    os.system("sclang sc/nrt_all.scd " + str(dur + 30) + ' ' + str(i))
    if os.path.exists(wav_path):
        i += 1
    return i


def test_JSON():
    """make sure JSON file is all good for supercollider"""
    mpv = json.load(open('JSON/moving_pluck_phrases.JSON'))
    out = True
    for i in range(len(mpv)):
        if len(mpv[i]['notes']) != len(mpv[i]['rt_durs']):
            out = False
    return out 


mp = json.load(open('JSON/meta_params.JSON'))

audio_files = glob.glob("../audioGeneration/*")
pre_set_af_nums = [int(re.sub('\D', '', af)) for af in audio_files]
af_nums = list(set(pre_set_af_nums))
af_nums.sort()
pre_set_af_nums.sort()
if len(af_nums) > 0: 
    af_min = af_nums[0]
else:
    af_min = 0
if len(af_nums) == 0:
    af_max = -1
else:
    af_max = af_nums[-1]
i = af_max + 1


missing_views = [v for v in range(af_min, af_min + 100) if v not in af_nums]
for i in missing_views:
    make_view(i)

while i < (af_min + 100):
    i = make_view(i)
    
sys.exit()
