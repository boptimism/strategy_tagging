import simdata_gen_func as sfunc
from scipy import stats
import numpy as np
import db_connect as dbc


def trial_gen(prior, history,
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
    idx = np.arange(len(strat_prob))
    strat_gen = stats.rv_discrete(name='strat_gen', values=(idx, p))
    s = strat_prob[strat_gen.rvs(), 0]
    f = getattr(sfunc, s)
    poke, re = f(history,
                 lott_mag, lott_prob, sure_mag,
                 alpha=alpha0, beta=beta0)
    return poke, re, s


if __name__ == '__main__':
    cur, con = dbc.connect()
    dbc.overwrite(cur, con, 'strattag_pokereward')
    # try:
    #     cur.execute("DELETE FROM strattag_pokereward")
    #     con.commit()
    # except:
    #     con.rollback()

    sqlcmd = 'SELECT * FROM strattag_config'
    cur.execute(sqlcmd)
    records = cur.fetchall()
    # chose initial bet to be lottery
    # initial reward = 0
    pre_poke = records[0][2]
    pre_reward = 0.0
    pre_config = records[0][1:3] + (pre_poke, pre_reward)
    pre_trialid = 1
    p_utility = 0.9
    p_rest = (1-p_utility)*0.25
    prior = {'randombet': p_rest,
             'sameport': p_rest,
             'samebet': p_rest,
             'winstayloseshift': p_rest,
             'utility': p_utility}

    # alpha = 1.5
    # beta = 1.0

    sqlstr = """INSERT INTO strattag_pokereward(
                trialid,
                poke,
                reward,
                strategy) VALUES (
                "%d", "%d", "%f", "%s")"""
    sqlcmd = sqlstr % (pre_trialid, pre_poke, pre_reward, 'utility')
    cur.execute(sqlcmd)

    for rec in records[1:]:
        alpha = rec[-2]
        beta = rec[-1]
        trialid = rec[0]
        cur_config = rec[1:3]
        history = pre_config + cur_config
        lott_mag = rec[3]
        lott_prob = rec[4]
        sure_mag = rec[5]
        poke, re, strat = trial_gen(prior, history,
                                    lott_mag,
                                    lott_prob,
                                    sure_mag,
                                    alpha,
                                    beta)
        pre_config = cur_config + (poke, re)
        sqlcmd = sqlstr % (trialid, poke, re, strat)
        cur.execute(sqlcmd)
    con.close()
