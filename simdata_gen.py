import simdata_gen_func as sfunc
from scipy import stats
import numpy as np
import MySQLdb as mdb


def trial_gen(prior,
              config, pre_action, pre_reward,
              lott_mag, lott_prob, sure_mag,
              alpha, beta):
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
    idx = np.arange(6)
    strat_gen = stats.rv_discrete(name='strat_gen', values=(idx, p))
    s = strat_prob[strat_gen.rvs(), 0]
    f = getattr(sfunc, s)
    ac, re, app, contri = f(config, pre_action, pre_reward,
                            lott_mag, lott_prob, sure_mag,
                            alpha=alpha, beta=beta)

    return ac, re, s, app, contri[0], contri[1]


if __name__ == '__main__':
    num_trials = 10000

    lott_mag = 10.
    lott_prob = 0.5
    sure_mag = 4.5

    # initialize config, arbitrary
    pre_config = [7, 6, 8]
    pre_action = pre_config[1]
    pre_reward = sure_mag

    prior = {'randombet': 0.05,
             'sameside': 0.1,
             'samechoice': 0.1,
             'sameaction': 0.05,
             'utility': 0.6,
             'winstayloseshift': 0.1}

    alpha = 1.0
    beta = 1.0
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
    sql_insertvals = """INSERT INTO strategy_tag2(
                        fix_port,
                        surebet_port,
                        lottery_port,
                        poke,
                        reward,
                        strategy,
                        applicable,
                        contribute_sure,
                        contribute_lott) VALUES (
                        "%d", "%d", "%d", "%d", "%f", "%s", "%d", "%f", "%f")
                        """
    # remove the previous records in SQL table.
    # This is more for debugging purpose
    sql_overwrite = "DELETE FROM strategy_tag2"
    cur.execute(sql_overwrite)
    sql_reset = "ALTER TABLE strategy_tag2 AUTO_INCREMENT = 1"
    cur.execute(sql_reset)
    try:
        con.commit()
    except:
        con.rollback()

    for i in range(num_trials):
        applicable = 0
        # strategy is employed only when it is applicable
        while not applicable:
            cur_config = list(np.random.choice(range(1, 10), 3, replace=False))
            config = pre_config + cur_config
            results = trial_gen(prior,
                                config, pre_action, pre_reward,
                                lott_mag, lott_prob, sure_mag,
                                alpha, beta)
            applicable = results[3]

        pre_config = cur_config
        pre_action = results[0]
        pre_reward = results[1]

        inputs = cur_config + list(results)
        sql_insert = sql_insertvals % tuple(inputs)
        cur.execute(sql_insert)

    con.close()
