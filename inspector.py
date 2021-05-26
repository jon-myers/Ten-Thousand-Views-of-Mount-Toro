import pickle
from piece import Piece, Section, Instance, Cycle
from instruments import Klank, Pluck
piece = pickle.load(open('pickles/piece.p', 'rb'))
klank = Klank(piece, 0)
# klank.make_packets()
# klank.add_notes()
# klank.save_as_json('json/klank_packets.JSON')
# print(piece.event_map)
# breakpoint()
# print(piece.irama)
