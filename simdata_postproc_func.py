# this file contains functions of 5 strategies
# they return a list of 2 probabilities
import numpy as np
from constant import *


def randombet():
    return np.array([0.5, 0.5])


def utility(history,
            lott_mag, lott_prob, sure_mag,
            alpha, beta):
    config = history[-3:]
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


def sameport(history):
    probs = [0]*3
    cur_fix = history[-3]
    pre_poke = history[3]
    cur_sureport = history[-2]
    cur_lottport = history[-1]
    if pre_poke in [cur_sureport, cur_lottport]:
        poke = pre_poke
        probs[poke] = 1.0
    probs.remove(probs[cur_fix])
    return np.array(probs)


def samebet(history):
    probs = [0]*3
    cur_fix = history[-3]
    pre_poke = history[3]
    pre_sureport = history[1]
    pre_lottport = history[2]
    cur_sureport = history[-2]
    cur_lottport = history[-1]
    if pre_poke == pre_sureport:
        probs[cur_sureport] = 1.0
    elif pre_poke == pre_lottport:
        probs[cur_lottport] = 1.0
    probs.remove(probs[cur_fix])
    return np.array(probs)


def winstayloseshift(history):
    probs = [0]*3
    cur_fix = history[-3]
    pre_poke = history[3]
    pre_reward = history[4]
    pre_sureport = history[1]
    pre_lottport = history[2]
    cur_sureport = history[-2]
    cur_lottport = history[-1]
    if pre_poke == pre_sureport:
        probs[cur_sureport] = 1.0
    elif pre_poke == pre_lottport:
        if pre_reward > 0:
            probs[cur_lottport] = 1.0
        else:
            probs[cur_sureport] = 1.0
    probs.remove(probs[cur_fix])
    return np.array(probs)
