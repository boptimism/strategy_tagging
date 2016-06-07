from constant import *
import numpy as np
# config:
# config[0:3] = fix_port, surebet_port, lottery_port at t-1
# config[3:6] = fix_port, surebet_port, lottery_port at t
# pre_action: action taken by agent at t-1
# pre_reward: reward taken by taking pre_action


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


def randombet(config, pre_action, pre_reward,
              lott_mag, lott_prob, sure_mag):
    # agent is allowed to randomly pick one of the 9 ports
    cur_sureport = config[4]
    cur_lottport = config[5]
    poke = np.random.randint(1, 10)
    re = reward(poke, cur_sureport, cur_lottport,
                lott_mag, lott_prob, sure_mag)
    return poke, re


def sameside(config, pre_action, pre_reward,
             lott_mag, lott_prob, sure_mag):
    # agent poke the port on the same side as the previous poke
    # when this strategy is not viable, e.g. agent poked
    # port on the right in previous trial, but is fixed at
    # the rightmost port, this strategy is not valid.
    # Agent will be left at the fixport under such situation.
    pre_fixport = config[0]
    cur_fixport = config[3]
    if pre_action in LEFT[pre_fixport]:
        choice = LEFT[cur_fixport]
    elif pre_action in RIGHT[pre_fixport]:
        choice = RIGHT[cur_fixport]
    elif pre_action in UP[pre_fixport]:
        choice = UP[cur_fixport]
    elif pre_action in DOWN[pre_fixport]:
        choice = DOWN[cur_fixport]
    else:
        choice = []
    if len(choice) == 0:
        poke = cur_fixport
    else:
        poke = choice[np.random.randint(len(choice))]
    re = reward(poke, cur_sureport, cur_lottport,
                lott_mag, lott_prob, sure_mag)
    return poke, re


def samechoice(config, pre_action, pre_reward,
               lott_mag, lott_prob, sure_mag):
    # same choice:
    # previously picked lottery, now pick lottery
    # previously picked surebet, now pick surebet
    # previously didn't pick any reward ports, now
    # pick non-reward ports
    pre_sureport = config[1]
    pre_lottport = config[2]
    cur_sureport = config[1]
    cur_lottport = config[2]
    non_reward_port = range(1, 10)
    non_reward_port.remove(cur_lottport)
    non_reward_port.remove(cur_sureport)

    if pre_action == pre_sureport:
        poke = cur_sureport
    elif pre_action == pre_lottport:
        poke = cur_lottport
    else:
        poke = non_reward_port[np.random.randint(7)]

    re = reward(poke, cur_sureport, cur_lottport,
                lott_mag, lott_prob, sure_mag)
    return poke, re


def sameaction(config, pre_action, pre_reward,
               lott_mag, lott_prob, sure_mag):
    # agent simply poke the same port
    poke = pre_action
    re = reward(poke, cur_sureport, cur_lottport,
                lott_mag, lott_prob, sure_mag)
    return poke, re


def utility(config, pre_action, pre_reward,
            lott_mag, lott_prob, sure_mag,
            alpha, beta):
    # agent uses utility curve to choose one of the following
    # 1) lottery
    # 2) sure
    # if the previous poke is not one of the two reward ports
    # this strategy is not applicable. Agent stays where
    # it is fixed.
    pre_sureport = config[1]
    pre_lottport = config[2]
    cur_fixport = config[3]
    cur_sureport = config[4]
    cur_lottport = config[5]
    if pre_action not in [pre_sureport, pre_lottport]:
        poke = cur_fixport
        re = reward(poke, cur_sureport, cur_lottport,
                    lott_mag, lott_prob, sure_mag)
        return poke, re
    ulott = lott_mag**alpha * lott_prob
    usure = sure_mag**alpha
    plott = 1./(1.+np.exp(-beta*(ulott-usure)))
    arand = np.random.random()
    if arand < plott:
        poke = cur_lottport
        re = reward(poke, cur_sureport, cur_lottport,
                    lott_mag, lott_prob, sure_mag)
        return poke, re
    else:
        poke = cur_sureport
        re = reward(poke, cur_sureport, cur_lottport,
                    lott_mag, lott_prob, sure_mag)
        return poke, re


def winstayloseshift(config, pre_action, pre_reward,
                     lott_mag, lott_prob, sure_mag,
                     alpha, beta):
    # win-stay-lose-shift:
    # if pick surebet, stays, since it always gives win
    # if pick lottery, but previous lottery doesn't
    # generate reward, switch.
    # if previous pick is not in the reward ports,
    # this strategy is not applicable
    pre_sureport = config[1]
    pre_lottport = config[2]
    cur_fixport = config[3]
    cur_sureport = config[4]
    cur_lottport = config[5]
    if pre_action not in [pre_sureport, pre_lottport]:
        poke = cur_fixport
        re = reward(poke, cur_sureport, cur_lottport,
                    lott_mag, lott_prob, sure_mag)
        return poke, re
    if pre_action == pre_sureport:
        assert pre_reward == sure_mag
        poke = cur_sureport
        re = reward(poke, cur_sureport, cur_lottport,
                    lott_mag, lott_prob, sure_mag)
        return poke, re
    else:
        if pre_reward > 0:
            poke = cur_lottport
            re = reward(poke, cur_sureport, cur_lottport,
                        lott_mag, lott_prob, sure_mag)
            return poke, re
        else:
            poke = cur_sureport
            re = reward(poke, cur_sureport, cur_lottport,
                        lott_mag, lott_prob, sure_mag)
            return poke, re
