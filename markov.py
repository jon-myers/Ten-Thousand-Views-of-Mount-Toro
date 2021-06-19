import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rhythm_tools import nCVI
rng = np.random.default_rng()
p_init = np.array([0.1, 0.8, 0.1])
p_transition = np.array(
    [[0.90, 0.05, 0.05],
     [0.01, 0.90, 0.09],
     [0.07, 0.03, 0.9]]
)


from scipy.stats import multinomial
def markov_sequence(p_init=None, p_transition=None, sequence_length=None):
    if p_init is None:
        p_init = equilibrium_distribution(p_transition)
    initial_state = list(multinomial.rvs(1, p_init)).index(1)
    states = [initial_state]
    for _ in range(sequence_length - 1):
        p_tr = p_transition[states[-1]]
        new_state = list(multinomial.rvs(1, p_tr)).index(1)
        states.append(new_state)
    return states

states = markov_sequence(p_init, p_transition, sequence_length=1000)
p_transition = rng.dirichlet(np.ones(5) * 10, (20))
# print(p_transition)
p_init = rng.dirichlet(np.ones(5) * 10, 1)[0]
# print(p_init)
states = markov_sequence(p_init, p_transition, 10)
# print(states)

from harmony_tools import utils as h_tools
from harmony_tools import plot as h_plot
import json

modes = json.load(open('JSON/modes_and_variations.json', 'rb'))
hsv = h_tools.gen_ratios_to_hsv(modes[0][2], [3, 5, 7])
print(hsv)

ord = h_tools.cast_to_ordinal(hsv)
pts, hole_arr = h_tools.fix_collection(ord)
print(pts, hole_arr)
# h_tools.plot_basic_hsl(pts, 'test.png')
col_choices = ['black', 'grey']
colors = [col_choices[int(i)] for i in hole_arr]
h_plot.make_plot(pts, 'test', [3, 5, 7], oct_generalized=True, colors=colors)
