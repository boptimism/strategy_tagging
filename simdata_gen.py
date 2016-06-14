import simdata_gen_func as sfunc
from scipy import stats
import numpy as np
import db_connect as dbc


def trial_gen(prior, config,
              lott_mag, lott_prob, sure_mag,
              alpha0, beta0):
    # generate trial by trial simulated data
    # inputs:
    # prior: prior prob of each strategy used, a dictionary
    # config: 3 element list/array, in the order of
    #         fix_port, sure_port, lott_port
    # alpha, beta: parameters in evaluting utility curve

    # the data has the following fields:
    # fix_port, sure_port, lottery_port, action, reward, strategy,
    # applicable, contribute_sure, contribute_lott

    # randomize the strategy going to be used
    strat_prob = np.array(sorted(prior.items()))
    p = (strat_prob[:, 1]).astype(float)
    s = strat_prob[:, 0]
    idx = np.arange(2)
    strat_gen = stats.rv_discrete(name='strat_gen', values=(idx, p))
    s = strat_prob[strat_gen.rvs(), 0]
    f = getattr(sfunc, s)
    ac, re = f(config,
               lott_mag, lott_prob, sure_mag,
               alpha=alpha0, beta=beta0)
    return ac, re, s


if __name__ == '__main__':
    cur, con = dbc.connect()
    try:
        cur.execute("DELETE FROM results")
        con.commit()
    except:
        con.rollback()

    sqlcmd = 'SELECT * FROM config'
    cur.execute(sqlcmd)
    records = cur.fetchall()

    prior = {'randombet': 0.2,
             'utility': 0.8}

    alpha = 1.5
    beta = 1.0

    sqlstr = """INSERT INTO results(
                trialid,
                poke,
                reward,
                strategy) VALUES (
                "%d", "%d", "%f", "%s")"""

    for rec in records:
        trialid = rec[0]
        config = rec[1:4]
        lott_mag = rec[4]
        lott_prob = rec[5]
        sure_mag = rec[6]
        poke, reward, strat = trial_gen(prior, config,
                                        lott_mag,
                                        lott_prob,
                                        sure_mag,
                                        alpha,
                                        beta)
        sqlcmd = sqlstr % (trialid, poke, reward, strat)
        cur.execute(sqlcmd)
    con.close()
