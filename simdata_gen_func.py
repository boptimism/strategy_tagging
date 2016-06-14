import numpy as np


def reward(poke, sure_port, lott_port, lott_mag, lott_prob, sure_mag):
    if poke == lott_port:
        arand = np.random.random()
        if arand <= lott_prob:
            return lott_mag
        else:
            return 0.0
    elif poke == sure_port:
        return sure_mag
    else:
        return 0.


def randombet(config,
              lott_mag, lott_prob, sure_mag,
              alpha=0.0, beta=0.0):
    # randomly pick from 2 ports
    cur_sureport = config[1]
    cur_lottport = config[2]
    arand = np.random.random()
    if arand > 0.5:
        poke = cur_sureport
    else:
        poke = cur_lottport
    re = reward(poke, cur_sureport, cur_lottport,
                lott_mag, lott_prob, sure_mag)
    return poke, re


def utility(config,
            lott_mag, lott_prob, sure_mag,
            alpha=0.0, beta=0.0):
    cur_sureport = config[1]
    cur_lottport = config[2]
    # utility: p*v^alpha
    ulott = lott_mag**alpha * lott_prob
    usure = sure_mag**alpha
    plott = 1./(1.+np.exp(-beta*(ulott-usure)))
    arand = np.random.random()
    if arand < plott:
        poke = cur_lottport
    else:
        poke = cur_sureport
    re = reward(poke, cur_sureport, cur_lottport,
                lott_mag, lott_prob, sure_mag)
    return poke, re
