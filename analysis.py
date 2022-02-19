import json, pickle, pprint
import numpy as np


print(json.load(open('JSON/meta_params.JSON', 'r'))[23])

# pprint.pprint(json.load(open('JSON/meta_params.JSON', 'r'))[50:100])

def save_analysis_piece(piece_idx=48):
    from piece import build 
    meta_params = json.load(open('JSON/meta_params.JSON', 'r'))
    # piece_idx = 47
    # print(meta_params[piece_idx])

    f, dur_tot, cycles, chords = meta_params[piece_idx]
    cycles = int(cycles)
    chords = int(chords)
    # print(dur_tot)
    build(True, False, True, cycles, dur_tot, f, chords)

save_analysis_piece(23)

# 
# time_obj = pickle.load(open('pickles/t.p', 'rb'))
# # print(json.dumps(time_obj.event_map, indent=4, sort_keys=True))
# # pprint.pprint(time_obj.real_time_event_map)
# times = np.array(list(time_obj.real_time_event_map.keys()))
# print(times)
# ids = np.array([time_obj.real_time_event_map[i]['mode'] for i in times])
# # print(ids)
