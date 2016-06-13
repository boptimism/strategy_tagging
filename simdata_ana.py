import simdata_postproc_func as pfunc
import numpy as np
import MySQLdb as mdb
from scipy import optimize as opt


def prob_poke(coeff, alpha, beta, external):
    # coeff : the priors for each strategy
    # external: a list contains:
    # config, pre_action, pre_reward, lott_mag, lott_prob, sure_mag
    config = external[0:6]
    pre_action = external[6]
    pre_reward = external[7]
    lott_mag = external[8]
    lott_prob = external[9]
    sure_mag = external[10]
    p1 = pfunc.randombet()
    p2 = pfunc.sameside(config, pre_action)
    p3 = pfunc.samechoice(config, pre_action)
    p4 = pfunc.sameaction(pre_action)
    p5 = pfunc.utility(config, pre_action,
                       lott_mag, lott_prob, sure_mag,
                       alpha, beta)
    p6 = pfunc.winstayloseshift(config, pre_action, pre_reward)

    est_p = np.dot(coeff, np.array([p1, p2, p3, p4, p5, p6]))
    return est_p


def cross_ent_loss(x, y):
    # x: coeff , alpha, beta
    # y: external controls
    #    a matrix with dimension n_trial X 11
    #    each row consists of :
    #    config, pre_action,pre_reward, lott_mag, lott_prob, sure_mag
    #    poke
    eps = 1.e-15
    coeff = x[:6]
    alpha = x[6]
    beta = x[7]
    entloss = 0.0
    for yi in y:
        q_i = prob_poke(coeff, alpha, beta, yi[:-1])
        p_i = yi[-1]
        p_vec = np.arange(9)
        p_vec[p_i-1] = 1.0
        # assert np.all(np.log(q_i) < 0)
        # assert np.all(np.log(1.-q_i) < 0)
        loss = np.dot(p_vec, np.log(q_i+eps))\
            + np.dot(1.-p_vec, np.log(1.-q_i + eps))
        entloss = entloss + loss
    entloss = -entloss/len(y)
    return entloss


def callbackF(x):
    x_check = list(x) + [int(np.all(x > 0)), sum(x[:6])]
    print '{0:<8.3f} {1:<8.3f} {2:<8.3f} {3:<8.3f} {4:<8.3f} {5:<8.3f}\
           {6:<8.3f} {7:<8.3f} {8:<8.2f} {9:<8.2f}'.format(*x_check)


if __name__ == "__main__":
    # lottery / surebet control
    # these parameters will be just readouts from SQL
    # for parameter scan
    lott_mag = 10
    lott_prob = 0.5
    sure_mag = 4.5
    # connect to SQL to record the trials
    credentials = {}
    with open('./credential_db.txt', 'r') as f:
        for l in f:
            fields = l.strip().split(":")
            try:
                credentials[fields[0]] = fields[1]
            except:
                pass

    con = mdb.connect(host=credentials['host'],
                      user=credentials['user'],
                      passwd=credentials['passwd'],
                      db=credentials['db'])

    cur = con.cursor()
    sql_join = """SELECT a.trialid,
                         a.fix_port, a.surebet_port, a.lottery_port,
                         b.fix_port, b.surebet_port, b.lottery_port,
                         a.poke, a.reward,
                         b.poke
                  FROM strategy_tag2 AS a
                  JOIN strategy_tag2 AS b
                  ON a.trialid=b.trialid-1
                  LIMIT 1000"""
    cur.execute(sql_join)
    readouts = cur.fetchall()
    con.close()
    num_trials = len(readouts)
    data = [0]*num_trials
    lottpara = [lott_mag, lott_prob, sure_mag]
    for i, r in enumerate(readouts):
        rec = list(r)
        data[i] = rec[1:-1]+lottpara+[rec[-1]]
    data = (tuple(data),)
    #paras = tuple([1./6.]*6 + [0.05, 0.05])
    paras = np.array([1./6.]*6 + [0.05, 0.05])

    # minimization of cross entropy
    # under the constraint that
    # sum(coeff) = 1
    # all parameter > 0
    tol = 1.e-7
    fieldstr = ('c1', 'c2', 'c3', 'c4', 'c5', 'c6',
                'alpha', 'beta', 'positiv', 'sum=1')
    print '{0:8s} {1:8s} {2:8s} {3:8s} {4:8s} {5:8s}\
           {6:8s} {7:8s} {8:8s} {9:8s}'.format(*fieldstr)

    # constr = {'type': 'eq',
    #           'fun': lambda x: np.all(np.array(x) > 0) - 1,
    #           'type': 'ineq',
    #           'fun': lambda x: sum(x[:6])-1. + tol,
    #           'type': 'ineq',
    #           'fun': lambda x: -sum(x[:6]) + 1. + tol}
    constr = {'type': 'eq',
              'fun': lambda x: sum(x[:6])-1.}
    bnds = ((0., 1.),)*8
    slsqp_opt = {'disp': True}

    res = opt.minimize(cross_ent_loss,
                       paras,
                       args=data,
                       method='SLSQP',
                       bounds=bnds,
                       constraints=constr,
                       callback=callbackF,
                       options=slsqp_opt)
    print res.x
