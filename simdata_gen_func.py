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
        # punishment for not choosing rewarding port
        # when there are only sure/lott port, this
        # doesn't apply. Here it merely is used as
        # an indicator in debugging.
        return -1.0


def randombet(history,
              lott_mag, lott_prob, sure_mag,
              alpha=0.0, beta=0.0):
    # randomly pick from 2 ports
    # history: a list, containing latest 2 trials
    #          [pre_sure, pre_lottery, pre_poke, pre_reward,
    #           cur_sure, cur_lottery]
    cur_sureport = history[-2]
    cur_lottport = history[-1]
    arand = np.random.random()
    if arand > 0.5:
        poke = cur_sureport
    else:
        poke = cur_lottport
    re = reward(poke, cur_sureport, cur_lottport,
                lott_mag, lott_prob, sure_mag)
    return poke, re


def utility(history,
            lott_mag, lott_prob, sure_mag,
            alpha=0.0, beta=0.0):
    cur_sureport = history[-2]
    cur_lottport = history[-1]
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


def sameport(history,
             lott_mag, lott_prob, sure_mag,
             alpha=0.0, beta=0.0):
    # animal keep poking at the same port
    pre_poke = history[2]
    cur_sureport = history[-2]
    cur_lottport = history[-1]
    poke = pre_poke
    re = reward(poke, cur_sureport, cur_lottport,
                lott_mag, lott_prob, sure_mag)
    return poke, re


def samebet(history,
            lott_mag, lott_prob, sure_mag,
            alpha=0.0, beta=0.0):
    # if animal poke sure_port/lott_port in previous trial
    # animal will poke sure_port/lott_port in current trial
    pre_sureport = history[0]
    pre_poke = history[2]
    cur_sureport = history[-2]
    cur_lottport = history[-1]
    if pre_poke == pre_sureport:
        poke = cur_sureport
    else:
        poke = cur_lottport

    re = reward(poke, cur_sureport, cur_lottport,
                lott_mag, lott_prob, sure_mag)
    return poke, re


def winstayloseshift(history,
                     lott_mag, lott_prob, sure_mag,
                     alpha=0.0, beta=0.0):
    # win-stay/lose-shift
    # if animal went to sure bet in previous trial,
    # it will stay in sure bet
    # else if animal went to lottery, and gain reward,
    # it will stay in lottery
    # else if animal went to lottery, but gain nothing,
    # it will switch to sure bet
    pre_sureport = history[0]
    pre_poke = history[2]
    pre_reward = history[3]
    cur_sureport = history[4]
    cur_lottport = history[5]
    if pre_poke == pre_sureport:
        poke = cur_sureport
        re = reward(poke, cur_sureport, cur_lottport,
                    lott_mag, lott_prob, sure_mag)

    else:
        if pre_reward > 0:
            poke = cur_lottport
            re = reward(poke, cur_sureport, cur_lottport,
                        lott_mag, lott_prob, sure_mag)
        else:
            poke = cur_sureport
            re = reward(poke, cur_sureport, cur_lottport,
                        lott_mag, lott_prob, sure_mag)
    return poke, re
