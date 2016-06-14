import db_connect as dbc
import simdata_postproc_func as pfunc
import numpy as np
from scipy import optimize as opt


def prob_poke(coeff, alpha, beta, external):
    # coeff : the priors for each strategy
    # external: a list contains:
    # config, pre_action, pre_reward, lott_mag, lott_prob, sure_mag
    config = external[0:3]
    lott_mag = external[3]
    lott_prob = external[4]
    sure_mag = external[5]
    p1 = pfunc.randombet()
    p2 = pfunc.utility(config,
                       lott_mag, lott_prob, sure_mag,
                       alpha, beta)

    est_p = np.dot(coeff, np.array([p1, p2]))
    assert np.all(np.log(est_p) < 0)
    assert np.all(np.log(1.-est_p) < 0)
    return est_p


def cross_ent_loss(x, y):
    # x: coeff , alpha, beta
    # y: external controls
    #    a matrix with dimension n_trial X 11
    #    each row consists of :
    #    config, lott_mag, lott_prob, sure_mag, poke
    eps = 1.e-15
    coeff = np.abs(x[:2])/sum(np.abs(x[:2]))
    alpha = x[2]
    beta = x[3]
    entloss = 0.0
    for yi in y:
        q_i = prob_poke(coeff, alpha, beta, yi[:-1])
        p_i = yi[-1]
        p_vec = np.zeros(3)
        p_vec[p_i] = 1.0
        mask = np.ones(len(p_vec), dtype=bool)
        mask[yi[0]] = False
        ps = p_vec[mask]
        loss = np.dot(ps, np.log(q_i+eps))\
            + np.dot(1.-ps, np.log(1.-q_i + eps))
        entloss = entloss + loss
    entloss = -entloss/len(y)
    return entloss


def callbackF(x):
    x_check = list(np.abs(x[:2])/sum(np.abs(x[:2]))) + [x[2], x[3]]
    print '{0:<8.3f} {1:<8.3f} {2:<8.3f} {3:<8.3f}'.format(*x_check)


if __name__ == "__main__":
    cur, con = dbc.connect()
    sqlcmd = """SELECT a.fix_port, a.surebet_port, a.lottery_port,
                a.lottery_mag, a.lottery_prob, a.sure_mag,
                b.poke
                FROM config AS a
                JOIN results AS b
                ON a.trialid=b.trialid"""
    cur.execute(sqlcmd)
    records = cur.fetchall()
    con.close()

    data = (records,)

    methds = 'SLSQP'
    print "methods: ", methds
    fieldstr = ('random', 'utility', 'alpha', 'beta')
    print '{0:8s} {1:8s} {2:8s} {3:8s}'.format(*fieldstr)

    paras = tuple([0.5]*4)

    methds_opt = {'disp': True}
    res = opt.minimize(cross_ent_loss,
                       paras,
                       args=data,
                       method=methds,
                       callback=callbackF,
                       options=methds_opt)

    print res.x
