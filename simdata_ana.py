import db_connect as dbc
import simdata_postproc_func as pfunc
import numpy as np
from scipy import optimize as opt

NUM_STRATEGY = 5
NUM_PORTS = 2


def prob_poke(coeff, alpha, beta, external):
    # coeff : the priors for each strategy
    # external: a list contains:
    # history, lott_mag, lott_prob, sure_mag
    # history: a list contains:
    # pre_fix, pre_sure, pre_lott, pre_poke, pre_reward,
    # cur_fix, cur_sure, cur_lott
    history = external[0:6]
    lott_mag = external[6]
    lott_prob = external[7]
    sure_mag = external[8]
    p1 = pfunc.randombet()
    p2 = pfunc.utility(history,
                       lott_mag, lott_prob, sure_mag,
                       alpha, beta)
    p3 = pfunc.sameport(history)
    p4 = pfunc.samebet(history)
    p5 = pfunc.winstayloseshift(history)

    est_p = np.dot(coeff, np.array([p1, p2, p3, p4, p5]))
    assert np.all(np.log(est_p) < 0)
    assert np.all(np.log(1.-est_p) < 0)
    return est_p


def cross_ent_loss(x, y):
    # x: coeff , alpha, beta
    # y: external controls
    #    a matrix with dimension n_trial X 7
    #    each row consists of :
    #    sure_port, lottery_port, poke, reward,
    #    lott_mag, lott_prob, sure_mag
    eps = 1.e-15
    coeff = np.abs(x[:NUM_STRATEGY])/sum(np.abs(x[:NUM_STRATEGY]))
    alpha = x[NUM_STRATEGY]
    beta = x[NUM_STRATEGY+1]
    entloss = 0.0
    y0 = y[0]
    for yi in y[1:]:
        p_i = yi[2]  # p_i: actually poke
        external = y0[:4] + yi[:2] + yi[-3:]
        q_i = prob_poke(coeff, alpha, beta, external)
        p_vec = np.zeros(NUM_PORTS)
        p_vec[p_i] = 1.0
        loss = np.dot(p_vec, np.log(q_i+eps))\
            + np.dot(1.-p_vec, np.log(1.-q_i + eps))
        entloss = entloss + loss
        y0 = yi
    entloss = -entloss/len(y)
    return entloss


def callbackF(x):
    likelyhood = np.abs(x[:NUM_STRATEGY])/sum(np.abs(x[:NUM_STRATEGY]))
    x_check = list(likelyhood) + [x[-2], x[-1]]
    print '{0:<8.3f} \
           {1:<8.3f} \
           {2:<8.3f} \
           {3:<8.3f} \
           {4:<8.3f} \
           {5:<8.3f} \
           {6:<8.3f}'.format(*x_check)


if __name__ == "__main__":
    cur, con = dbc.connect()
    sqlcmd = """SELECT
                a.surebet_port, a.lottery_port,
                b.poke, b.reward,
                a.lottery_mag, a.lottery_prob, a.sure_mag
                FROM config AS a
                JOIN results AS b
                ON a.trialid=b.trialid
                """
    cur.execute(sqlcmd)
    records = cur.fetchall()
    con.close()

    data = (records,)

    methds = 'SLSQP'
    print "methods: ", methds
    fieldstr = ('random',
                'utility',
                'samebet',
                'sameport',
                'wsls',
                'alpha',
                'beta')
    print '{0:8s} \
           {1:8s} \
           {2:8s} \
           {3:8s} \
           {4:8s} \
           {5:8s} \
           {6:8s}'.format(*fieldstr)

    paras = tuple([0.5]*NUM_STRATEGY) + (2.0, 0.5)

    methds_opt = {'disp': True}
    res = opt.minimize(cross_ent_loss,
                       paras,
                       args=data,
                       method=methds,
                       callback=callbackF,
                       options=methds_opt)
    print res.x
    p_f = res.x
    print "Finaly Results:"
    print list(np.abs(p_f[:5])/sum(np.abs(p_f[:5]))) + list(p_f[5:])
