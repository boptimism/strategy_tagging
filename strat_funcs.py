import numpy as np
from strategies import LEFT, RIGHT, UP, DOWN

# A set of functions based on strategy are defined to give probability of
# which port will be hit.

# Input: pre_conf, pre_action, pre_reward, pre_delta_reward,
#        cur_conf

# Output: 1X9 array, represent the probability of each port being hit

# The goal is to set a proper set of linear combination, that maximize
# loglikelihood of the observed poke carried out by one agent.

PORTS = np.arange(1, 10)


def f_randombet(pre_conf, pre_action, pre_reward,
                pre_delta_reward, cur_conf):
    cur_fixport = cur_conf[0]
    probs = np.ones(9)*0.125
    probs[cur_fixport-1] = 0.0

    assert np.sum(probs) > 1.0-1.e-7 and np.sum(probs) < 1.0 + 1.e-7

    return probs


def f_sameside(pre_conf, pre_action, pre_reward,
               pre_delta_reward, cur_conf):
    pre_fixport = pre_conf[0]
    cur_fixport = cur_conf[0]
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

    a = len(choice)
    probs = np.zeros(9)
    if a == 0:
        probs[cur_fixport - 1] = 1.0
    elif a == 1:
        probs[choice[0] - 1] = 1.0
    else:
        for i in choice:
            probs[i-1] = 1./a

    assert np.sum(probs) > 1.0-1.e-7 and np.sum(probs) < 1.0 + 1.e-7

    return probs


def f_samechoice(pre_conf, pre_action, pre_reward,
                 pre_delta_reward, cur_conf):
    pre_lotteryport = pre_conf[2]
    pre_surebetport = pre_conf[1]
    cur_lotteryport = cur_conf[2]
    cur_surebetport = cur_conf[1]
    cur_fixport = cur_conf[0]

    probs = np.zeros(9)
    if pre_action == pre_lotteryport:
        probs[cur_lotteryport-1] = 1.0
    elif pre_action == pre_surebetport:
        probs[cur_surebetport-1] = 1.0
    else:
        probs[cur_fixport-1] = 1.0

    assert np.sum(probs) > 1.0-1.e-7 and np.sum(probs) < 1.0 + 1.e-7

    return probs


def f_sameaction(pre_conf, pre_action, pre_reward,
                 pre_delta_reward, cur_conf):
    probs = np.zeros(9)
    probs[pre_action-1] = 1.0
    return probs


def f_utility(cur_conf, alpha, beta,
              reward_prob, reward_mag, reward_sure):

    ulottery = (reward_mag*reward_prob)**alpha
    usurebet = reward_sure**alpha
    unothing = 0.0
    a = np.exp(beta*(usurebet - ulottery)) + \
        np.exp(beta*(unothing - ulottery))
    b = np.exp(beta*(ulottery - usurebet)) + \
        np.exp(beta*(unothing - usurebet))
    p_lottery = 1./(1. + a)
    p_surebet = 1./(1. + b)
    p_nothing = 1. - p_lottery - p_surebet

    probs = np.ones(9)*p_nothing/7

    cur_surebetport = cur_conf[1]
    cur_lotteryport = cur_conf[2]

    probs[cur_surebetport-1] = p_surebet
    probs[cur_lotteryport-1] = p_lottery

    assert np.sum(probs) > 1.0-1.e-7 and np.sum(probs) < 1.0 + 1.e-7

    return probs


def f_winloseshift(pre_conf, pre_action, pre_reward,
                   pre_delta_reward, cur_conf, kappa):

    probs = np.zeros(9)

    pre_lotteryport = pre_conf[2]
    pre_surebetport = pre_conf[1]
    cur_lotteryport = cur_conf[2]
    cur_surebetport = cur_conf[1]
    cur_fixport = cur_conf[0]

    if pre_delta_reward <= 0:
        if pre_action == pre_lotteryport:
            probs[cur_surebetport-1] = 1.0
        elif pre_action == pre_surebetport:
            probs[cur_lotteryport-1] = 1.0
        else:
            probs[cur_fixport-1] = 1.0
    else:
        prob_stay = np.exp(-kappa*pre_delta_reward)
        if pre_action == pre_lotteryport:
            probs[cur_lotteryport-1] = prob_stay
            probs[cur_surebetport-1] = 1.-prob_stay
        elif pre_action == pre_surebetport:
            probs[cur_lotteryport-1] = 1.-prob_stay
            probs[cur_surebetport-1] = prob_stay
        else:
            probs[cur_fixport-1] = 1.0

    assert np.sum(probs) > 1.0-1.e-7 and np.sum(probs) < 1.0 + 1.e-7

    return probs
