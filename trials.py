import numpy as np
from scipy import stats
import strategies as strats

PRIOR = {"RandomBet": 0.1,
         "SameSide": 0.1,
         "SameAction": 0.1,
         "SameChoice": 0.1,
         "Utility": 0.5,
         "WinLoseShift": 0.1}

strat_prob = np.array(sorted(PRIOR.items()))
p = (strat_prob[:, 1]).astype(float)
s = strat_prob[:, 0]
idx = np.arange(6)
strat_gen = stats.rv_discrete(name='strat_gen', values=(idx, p))

num_trials = 100
ports = np.arange(1, 10)
pre_conf = np.random.choice(ports, 3, replace=False)
pre_action = 3
pre_reward = 0
pre_delta_reward = 0

alpha = 0.85
beta = 1.0
kappa = 1.0

for i in range(1, num_trials):

    # set surebet vs lottery bet parameter
    lottery_prob = 0.5
    lottery_max_reward = 10
    surebet_reward = 4.5

    # randomly generate a strategy at each trial
    # using getattr is a neat technique

    s = strat_prob[strat_gen.rvs(), 0]
    strat_used = getattr(strats, s)
    cur_conf = np.random.choice(ports, 3, replace=False)
    a_strat = strat_used(pre_conf,
                         cur_conf,
                         pre_action,
                         pre_reward,
                         pre_delta_reward)

    # ac: action taken by agent
    # re: reward received from the action ac
    # delta_re: change of reward comparing with previous trial

    if s == "Utility":
        ac = a_strat.action(alpha, beta,
                            lottery_prob, lottery_max_reward,
                            surebet_reward)

    elif s == "WinLoseShift":
        ac = a_strat.action(kappa)

    else:
        ac = a_strat.action()

    re, delta_re = a_strat.reward(lottery_prob,
                                  lottery_max_reward,
                                  surebet_reward)

    print '{0:d} {1:.2f} {2:.2f} {3:<10s}'.format(ac, re, delta_re, s)
