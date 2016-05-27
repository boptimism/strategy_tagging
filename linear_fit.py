import MySQLdb as mdb
import numpy as np
import strat_funcs as sf
from sklearn import linear_model

ALPHA = 0.85
BETA = 1.0
KAPPA = 1.0
REWARD_PROB = 0.5
REWARD_MAG = 10.0
REWARD_SURE = 4.5
NUM_STRAT = 6
NUM_PORTS = 9
# strategy will be indexed:
# 0: randombet
# 1: sameside
# 2: samechoice
# 3: sameaction
# 4: utility
# 5: winloseshift


def compute_xy(records):
    rec_size = len(records)
    xdata = np.zeros((rec_size, NUM_PORTS, NUM_STRAT))
    xdata.fill(np.nan)
    ydata = np.zeros((rec_size, NUM_PORTS))
    ydata.fill(-np.nan)

    for i, rec in enumerate(records):
        pre_conf = rec[1:4]
        pre_action = rec[4]
        pre_reward = rec[5]
        pre_delta_reward = rec[6]
        cur_conf = rec[7:10]
        cur_action = rec[10]

        ports = np.zeros(9)
        ports[cur_action-1] = 1.0
        ydata[i] = ports

        t1 = sf.f_randombet(pre_conf, pre_action,
                            pre_reward, pre_delta_reward,
                            cur_conf)

        t2 = sf.f_sameside(pre_conf, pre_action,
                           pre_reward, pre_delta_reward,
                           cur_conf)

        t3 = sf.f_samechoice(pre_conf, pre_action,
                             pre_reward, pre_delta_reward,
                             cur_conf)

        t4 = sf.f_sameaction(pre_conf, pre_action,
                             pre_reward, pre_delta_reward,
                             cur_conf)

        t5 = sf.f_utility(cur_conf, ALPHA, BETA,
                          REWARD_PROB, REWARD_MAG, REWARD_SURE)

        t6 = sf.f_winloseshift(pre_conf, pre_action,
                               pre_reward, pre_delta_reward,
                               cur_conf, KAPPA)

        xdata[i] = np.array([t1, t2, t3, t4, t5, t6]).T

    return xdata, ydata

# def compute_f_mat(data, seq):
#     fipj = np.zeros((len(seq), NUM_STRAT))
#     fipj.fill(np.nan)
#     for k, idx in enumerate(seq):
#         rec = data[idx]
#         pre_conf = rec[1:4]
#         pre_action = rec[4]
#         pre_reward = rec[5]
#         pre_delta_reward = rec[6]
#         cur_conf = rec[7:10]
#         cur_action = rec[10]
#         p = cur_action - 1
#         t = sf.f_randombet(pre_conf, pre_action,
#                            pre_reward, pre_delta_reward,
#                            cur_conf)

#         # fipj: ith trial, p_j port is poked
#         fipj[k, 0] = t[p]

#         t = sf.f_sameside(pre_conf, pre_action,
#                           pre_reward, pre_delta_reward,
#                           cur_conf)
#         fipj[k, 1] = t[p]
#         t = sf.f_samechoice(pre_conf, pre_action,
#                             pre_reward, pre_delta_reward,
#                             cur_conf)
#         fipj[k, 2] = t[p]
#         t = sf.f_sameaction(pre_conf, pre_action,
#                             pre_reward, pre_delta_reward,
#                             cur_conf)
#         fipj[k, 3] = t[p]
#         t = sf.f_utility(cur_conf, ALPHA, BETA,
#                          REWARD_PROB, REWARD_MAG, REWARD_SURE)
#         fipj[k, 4] = t[p]
#         t = sf.f_winloseshift(pre_conf, pre_action,
#                               pre_reward, pre_delta_reward,
#                               cur_conf, KAPPA)
#         fipj[k, 5] = t[p]

#     assert np.all(fipj >= 0) and np.all(fipj <= 1.)

#     return fipj

# it may not be necessary to use SGD. Just use ridge regression.
# def sgd(coeff, data, lbd, batchsize=1000, epochs=5, eps=0., check_ll=False):
#     datasize = len(data)
#     sequence = np.arange(datasize)
#     num_batch = datasize/batchsize

#     # if check loglikelihood
#     if check_ll:
#         ll = np.zeros((epochs, num_batch))
#         ll.fill(np.nan)

#     for i in np.arange(epochs):
#         # shuffle the records
#         np.random.shuffle(sequence)
#         sgdbatch = np.split(sequence, num_batch)
#         for j, single_batch_idx in enumerate(sgdbatch):
#             fipj = compute_f_mat(data, single_batch_idx)
#             denom = np.dot(fipj, coeff)
#             dervsum = np.sum(fipj.T/denom, 1)
#             coeff0 = coeff
#             coeff = coeff0 - lbd * dervsum
#             coeff = coeff/np.linalg.norm(coeff)

#         if check_ll:
#             fipj_all = compute_f_mat(data, sequence)
#             ll[i, j] = np.sum(np.log(fipj_all.dot(coeff)))
#     if check_ll:
#         return coeff, ll
#     else:
#         return coeff


con = mdb.connect(host='127.0.0.1',
                  user='bo',
                  passwd='Hbar10_34',
                  db='pa')
cur = con.cursor()

sql_selfjoin = """SELECT a.trialid,
                         a.fix_port,
                         a.surebet_port,
                         a.lottery_port,
                         a.poke,
                         a.reward,
                         a.delta_reward,
                         b.fix_port,
                         b.surebet_port,
                         b.lottery_port,
                         b.poke
                FROM strategy_tag AS a
                JOIN strategy_tag AS b
                ON a.trialid=b.trialid-1"""

cur.execute(sql_selfjoin)
data = cur.fetchall()
con.close()

# ldb:       learning rate in SGD optimization
# coeff:     coefficients of linear combination
# batchsize: the batchsize in SGD
# epoch:     number of epochs needs to be used
#            this indicates how many iterations are needed
# eps:       cutoff parameter. When norm of coeffs at two neighboring
#            iterations varies less by eps, iteration stops.

# lbd = 1.e-4
# coeff = np.ones(6) * 1./6.
# epoch = 10
# check_ll = True
# coeff_last = sgd(coeff, data, lbd, check_ll=check_ll)

xs, ys = compute_xy(data)

xs_unpack = xs.reshape((len(data)*NUM_PORTS, NUM_STRAT))
ys_unpack = ys.reshape((len(data)*NUM_PORTS, 1))

# clf = linear_model.Ridge(alpha=.5)
# clf.fit(xs_unpack, ys_unpack)
# coeff_fit = clf.coef_.flatten()

# prior_p = {'RandomBet': coeff_fit[0],
#            'SameSide': coeff_fit[1],
#            'SameChoice': coeff_fit[2],
#            'SameAction': coeff_fit[3],
#            'Utility': coeff_fit[4],
#            'WinLoseShift': coeff_fit[5]}

# for k, v in sorted(prior_p.items()):
#     print k, ":", v

import matplotlib.pyplot as plt

alphas = np.arange(0, 100, 10)*0.5

coeffs = np.zeros((len(alphas), 6))

for i, a in enumerate(alphas):
    clf = linear_model.Ridge(alpha=a)
    clf.fit(xs_unpack, ys_unpack)
    coeff_fit = clf.coef_.flatten()
    coeffs[i] = coeff_fit
    prior_p = {'RandomBet': coeff_fit[0],
               'SameSide': coeff_fit[1],
               'SameChoice': coeff_fit[2],
               'SameAction': coeff_fit[3],
               'Utility': coeff_fit[4],
               'WinLoseShift': coeff_fit[5]}

    print 'alpha: ', a, 'intercept: ', clf.intercept_
    print ''
    for k, v in sorted(prior_p.items()):
        print k, ":", v

plt.figure()
plt.plot(alphas, coeffs, '-o')
legendmsg = ('randombet',
             'sameside',
             'samechoice',
             'sameaction',
             'utility',
             'winloseshift')
plt.legend(legendmsg, loc='upper left', bbox_to_anchor=(1, 1))
plt.grid()
plt.show()
