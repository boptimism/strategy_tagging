# this file contains functions of 5 strategies
# they return a list of 2 probabilities
import numpy as np
from constant import *

NUM_PORTS = 2


def randombet():
    return np.array([0.5, 0.5])


def utility(history,
            lott_mag, lott_prob, sure_mag,
            alpha, beta):
    # history: a list containing latest 2 trials:
    #         [pre_sureport, pre_lottport, pre_poke, pre_reward,
    #          cur_sureport, cur_lottport]
    # lott_mag: magnetitude of lottery reward
    # lott_prob: probability to get that reward
    # sure_mag: sure bet reward
    # alpha: risk coeff
    # beta: temperature coeff

    cur_surebet = history[-2]
    cur_lottbet = history[-1]
    ulott = lott_mag**alpha * lott_prob
    usure = sure_mag**alpha
    plott = 1./(1.+np.exp(-beta*(ulott-usure)))
    probs = [0]*NUM_PORTS
    probs[cur_surebet] = 1.-plott
    probs[cur_lottbet] = plott
    return np.array(probs)


def sameport(history):
    probs = [0]*NUM_PORTS
    pre_poke = history[2]
    poke = pre_poke
    probs[poke] = 1.0
    return np.array(probs)


def samebet(history):
    probs = [0]*NUM_PORTS

    pre_sureport = history[0]
    pre_lottport = history[1]
    pre_poke = history[2]
    cur_sureport = history[-2]
    cur_lottport = history[-1]

    if pre_poke == pre_sureport:
        probs[cur_sureport] = 1.0
    elif pre_poke == pre_lottport:
        probs[cur_lottport] = 1.0
    else:
        pass

    return np.array(probs)


def winstayloseshift(history):
    probs = [0]*NUM_PORTS
    pre_poke = history[2]
    pre_reward = history[3]
    pre_sureport = history[0]
    pre_lottport = history[1]
    cur_sureport = history[-2]
    cur_lottport = history[-1]
    if pre_poke == pre_sureport:
        probs[cur_sureport] = 1.0
    elif pre_poke == pre_lottport:
        if pre_reward > 0:
            probs[cur_lottport] = 1.0
        else:
            probs[cur_sureport] = 1.0
    else:
        pass
    return np.array(probs)
