# this file contains functions of 6 strategies
# they return a list of 9 probabilities indicate 9 ports
# if the strategy is not applicable due to the config
# return all 0s.
import numpy as np
from constant import *


def randombet():
    return np.ones(9)/9.0


def sameside(config, pre_action):
    pre_fixport = config[0]
    cur_fixport = config[3]
    if pre_action in LEFT[pre_fixport]:
        choice = LEFT[cur_fixport]
    elif pre_action in RIGHT[pre_fixport]:
        choice = RIGHT[cur_fixport]
    elif pre_action in UP[pre_fixport]:
        choice = UP[cur_fixport]
    else:
        choice = DOWN[cur_fixport]
    if len(choice) == 0:
        return np.zeros(9)
    p = 1./len(choice)
    probs = np.zeros(9)
    for c in choice:
        probs[c-1] = p
    return probs


def samechoice(config, pre_action):
    pre_sureport = config[1]
    pre_lottport = config[2]
    cur_sureport = config[4]
    cur_lottport = config[5]
    non_reward_port = range(1, 10)
    non_reward_port.remove(cur_lottport)
    non_reward_port.remove(cur_sureport)
    probs = np.zeros(9)
    if pre_action == pre_sureport:
        probs[cur_sureport-1] = 1.
    elif pre_action == pre_lottport:
        probs[cur_lottport-1] = 1.
    else:
        for j in non_reward_port:
            probs[j-1] = 1./len(non_reward_port)
    return probs


def sameaction(pre_action):
    probs = np.zeros(9)
    probs[pre_action-1] = 1.
    return probs


def utility(config, pre_action,
            lott_mag, lott_prob, sure_mag,
            alpha, beta):
    cur_sureport = config[4]
    cur_lottport = config[5]
    pre_sureport = config[1]
    pre_lottport = config[2]
    if pre_action not in [pre_sureport, pre_lottport]:
        return np.zeros(9)
    ulott = lott_mag**alpha * lott_prob
    usure = sure_mag**alpha
    plott = 1./(1.+np.exp(-beta*(ulott-usure)))
    probs = np.zeros(9)
    probs[cur_sureport-1] = 1.-plott
    probs[cur_lottport-1] = plott
    return probs


def winstayloseshift(config, pre_action, pre_reward):
    pre_sureport = config[1]
    pre_lottport = config[2]
    cur_sureport = config[4]
    cur_lottport = config[5]
    if pre_action not in [pre_sureport, pre_lottport]:
        return np.zeros(9)
    probs = np.zeros(9)
    if pre_action == pre_sureport:
        probs[cur_sureport-1] = 1.
    else:
        assert pre_action == pre_lottport
        if pre_reward > 0:
            probs[cur_lottport-1] = 1.
        else:
            probs[cur_sureport-1] = 1.
    return probs
