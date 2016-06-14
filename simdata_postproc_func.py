# this file contains functions of 6 strategies
# they return a list of 9 probabilities indicate 9 ports
# if the strategy is not applicable due to the config
# return all 0s.
import numpy as np
from constant import *


def randombet():
    return np.array([0.5, 0.5])


def utility(config,
            lott_mag, lott_prob, sure_mag,
            alpha, beta):
    cur_fix = config[0]
    cur_surebet = config[1]
    cur_lottbet = config[2]
    ulott = lott_mag**alpha * lott_prob
    usure = sure_mag**alpha
    plott = 1./(1.+np.exp(-beta*(ulott-usure)))
    probs = [0]*3
    probs[cur_surebet] = 1.-plott
    probs[cur_lottbet] = plott
    probs.remove(probs[cur_fix])
    return np.array(probs)
