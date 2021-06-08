from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import pickle
from piece import Piece, Section, Instance, Cycle
from instruments import Klank, Pluck
import numpy as np
piece = pickle.load(open('pickles/piece.p', 'rb'))

# my_cmap = plt.get_cmap('turbo')
# rescale = lambda y: (y - np.min(y)) / np.max(y) - np.min(y)
#
# starts = [section.cy_start for section in piece.sections]
# durs = [section.cy_end - section.cy_start for section in piece.sections]
# print(durs)
#
# cmap_vals = np.linspace(0, 1, len(starts))
# adds = np.tile([0, 0.55], np.ceil(len(starts)/2).astype(int))
# cmap_vals += adds[:len(cmap_vals)]
# cmap_vals = cmap_vals % 1
# print(np.round(cmap_vals, 2))
# plt.bar(starts, 1, durs, align='edge', color=my_cmap(rescale(cmap_vals)))
# # for s, section in enumerate(piece.sections):
# #     cy_dur = section.cy_end-section.cy_start
# #     plt.bar(section.cy_start, 1, cy_dur, cmap=plt.get_cmap('viridis'))
# plt.show()


def plot_cycle(piece, path='temp_cycle.png', show=False, save=True):
    my_cmap = plt.get_cmap('turbo')
    rescale = lambda y: (y - np.min(y)) / np.max(y) - np.min(y)
    starts = [section.cy_start for section in piece.sections]
    durs = [section.cy_end - section.cy_start for section in piece.sections]
    cmap_vals = np.linspace(0, 1, len(starts))
    adds = np.tile([0, 0.55], np.ceil(len(starts)/2).astype(int))
    cmap_vals += adds[:len(cmap_vals)]
    cmap_vals = cmap_vals % 1
    fig = plt.figure(figsize=[8, 2],frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.bar(starts, 1, durs, align='edge', color=my_cmap(rescale(cmap_vals)))
    if show:
        plt.show()
    if save:
        # with open(path, 'w') as outfile:
        #     fig.canvas.print_png(outfile)
        plt.savefig(path, bbox_inches='tight', pad_inches=0)

def plot_all_cycles(piece, path='temp_all_cycles.png', show=False, save=True):
    my_cmap = plt.get_cmap('turbo')
    rescale = lambda y: (y - np.min(y)) / np.max(y) - np.min(y)
    starts = [section.cy_start for section in piece.sections]
    durs = [section.cy_end - section.cy_start for section in piece.sections]
    cmap_vals = np.linspace(0, 1, len(starts))
    adds = np.tile([0, 0.55], np.ceil(len(starts)/2).astype(int))
    cmap_vals += adds[:len(cmap_vals)]
    cmap_vals = cmap_vals % 1
    cmap_vals = np.tile(cmap_vals, piece.noc)
    rt_starts = []
    rt_durs = []
    for cycle in piece.cycles:
        for instance in cycle.instances:
            rt_starts.append(instance.rt_start)
            rt_durs.append(instance.rt_dur)
    fig = plt.figure(figsize=[8, 2],frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.bar(rt_starts, 1, rt_durs, align='edge', color=my_cmap(rescale(cmap_vals)))
    if show:
        plt.show()
    if save:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)



plot_all_cycles(piece)

# plot_cycle(piece)
