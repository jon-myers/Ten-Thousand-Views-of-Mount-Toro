import pickle
from piece import Piece, Section, Instance, Cycle
from instruments import Klank, Pluck
piece = pickle.load(open('pickles/piece.p', 'rb'))
# breakpoint()
# klank = Klank(piece, 0)
# klank.make_packets()
# klank.add_notes()
# klank.save_as_json('json/klank_packets.JSON')
# print(piece.event_map)
# breakpoint()
# print(piece.irama)
# print(piece.klank_packets[0])

# rests = [float(p['rt_dur'][5:-2]) for p in piece.klank_packets if p['type'] == 'rest' ]
# norm_durs = [p['rt_dur'] for p in piece.klank_packets if p['type'] != 'rest']
# print(norm_durs)

# print(sum(norm_durs), sum(rests), sum(rests)/ (sum(rests) + sum(norm_durs)))
# print(piece.melody)
