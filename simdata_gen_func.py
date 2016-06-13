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
              lott_mag, lott_prob, sure_mag,
              alpha=0.0, beta=0.0):
    # agent is allowed to randomly pick one of the 9 ports
    cur_sureport = config[4]
    cur_lottport = config[5]
    poke = np.random.randint(1, 10)
    re = reward(poke, cur_sureport, cur_lottport,
                lott_mag, lott_prob, sure_mag)
    # applicable is a variable indicates if this strategy
    # is applicable for the settings.
    applicable = 1
    # whether this strategy contributes to surebet or lottery
    # contribute[0]: surebet contribution
    # contribute[1]: lottery contribution
    contribute = [0., 0.]
    return poke, re, applicable, contribute


def sameside(config, pre_action, pre_reward,
             lott_mag, lott_prob, sure_mag,
             alpha=0.0, beta=0.0):
    # agent poke the port on the same side as the previous poke
    # when this strategy is not viable, e.g. agent poked
    # port on the right in previous trial, but is fixed at
    # the rightmost port, this strategy is not valid.
    # Agent will be left at the fixport under such situation.
    pre_fixport = config[0]
    cur_fixport = config[3]
    cur_sureport = config[4]
    cur_lottport = config[5]
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
        applicable = 0
        contribute = [0., 0.]
    else:
        poke = choice[np.random.randint(len(choice))]
        applicable = 1
        if (cur_lottport in choice) and (cur_sureport in choice):
            contribute = [1./len(choice), 1./len(choice)]
        elif cur_sureport in choice:
            contribute = [1./len(choice), 0.]
        elif cur_lottport in choice:
            contribute = [0., 1./len(choice)]
        else:
            contribute = [0., 0.]
    re = reward(poke, cur_sureport, cur_lottport,
                lott_mag, lott_prob, sure_mag)
    return poke, re, applicable, contribute


def samechoice(config, pre_action, pre_reward,
               lott_mag, lott_prob, sure_mag,
               alpha=0.0, beta=0.0):
    # same choice:
    # previously picked lottery, now pick lottery
    # previously picked surebet, now pick surebet
    # previously didn't pick any reward ports, now
    # pick non-reward ports
    pre_sureport = config[1]
    pre_lottport = config[2]
    cur_sureport = config[4]
    cur_lottport = config[5]
    non_reward_port = range(1, 10)
    non_reward_port.remove(cur_lottport)
    non_reward_port.remove(cur_sureport)

    if pre_action == pre_sureport:
        poke = cur_sureport
        contribute = [1.0, 0.0]
    elif pre_action == pre_lottport:
        poke = cur_lottport
        contribute = [0.0, 1.0]
    else:
        poke = non_reward_port[np.random.randint(7)]
        contribute = [0., 0.]

    re = reward(poke, cur_sureport, cur_lottport,
                lott_mag, lott_prob, sure_mag)

    applicable = 1

    return poke, re, applicable, contribute


def sameaction(config, pre_action, pre_reward,
               lott_mag, lott_prob, sure_mag,
               alpha=0.0, beta=0.0):
    # agent simply poke the same port
    poke = pre_action
    cur_sureport = config[4]
    cur_lottport = config[5]
    re = reward(poke, cur_sureport, cur_lottport,
                lott_mag, lott_prob, sure_mag)
    applicable = 1
    if poke == cur_sureport:
        contribute = [1.0, 0.0]
    elif poke == cur_lottport:
        contribute = [0.0, 1.0]
    else:
        contribute = [0.0, 0.0]
    return poke, re, applicable, contribute


def utility(config, pre_action, pre_reward,
            lott_mag, lott_prob, sure_mag,
            alpha=0.0, beta=0.0):
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
        applicable = 0
        contribute = [0., 0.]
        return poke, re, applicable, contribute
    applicable = 1
    ulott = lott_mag**alpha * lott_prob
    usure = sure_mag**alpha
    plott = 1./(1.+np.exp(-beta*(ulott-usure)))
    arand = np.random.random()
    contribute = [1.-plott, plott]
    if arand < plott:
        poke = cur_lottport
        re = reward(poke, cur_sureport, cur_lottport,
                    lott_mag, lott_prob, sure_mag)
        return poke, re, applicable, contribute
    else:
        poke = cur_sureport
        re = reward(poke, cur_sureport, cur_lottport,
                    lott_mag, lott_prob, sure_mag)
        return poke, re, applicable, contribute


def winstayloseshift(config, pre_action, pre_reward,
                     lott_mag, lott_prob, sure_mag,
                     alpha=0.0, beta=0.0):
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
        applicable = 0
        contribute = [0., 0.]
        return poke, re, applicable, contribute
    if pre_action == pre_sureport:
        assert pre_reward == sure_mag
        poke = cur_sureport
        re = reward(poke, cur_sureport, cur_lottport,
                    lott_mag, lott_prob, sure_mag)
        applicable = 1
        contribute = [1.0, 0.0]
        return poke, re, applicable, contribute
    else:
        applicable = 1
        if pre_reward > 0:
            poke = cur_lottport
            re = reward(poke, cur_sureport, cur_lottport,
                        lott_mag, lott_prob, sure_mag)
            contribute = [0., 1.]
            return poke, re, applicable, contribute
        else:
            poke = cur_sureport
            re = reward(poke, cur_sureport, cur_lottport,
                        lott_mag, lott_prob, sure_mag)
            contribute = [1.0, 0.0]
            return poke, re, applicable, contribute
