import numpy as np
from scipy import stats
import strategies as strats
import MySQLdb as mdb

# all the constants
# PRIOR
# Four Sides defined in strategies
# Alpha, Beta, Kappa Reward_mag, prob, Reward_surebet
# will later be put in a single file for reference
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

ports = np.arange(1, 10)
pre_conf = np.random.choice(ports, 3, replace=False)
pre_action = 3
pre_reward = 0
pre_delta_reward = 0

alpha = 0.85
beta = 1.0
kappa = 1.0

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

sql_createtbl = """CREATE TABLE IF NOT EXISTS strategy_tag(
                    trialid INT UNSIGNED NOT NULL AUTO_INCREMENT,
                    fix_port TINYINT NOT NULL,
                    surebet_port TINYINT NOT NULL,
                    lottery_port TINYINT NOT NULL,
                    poke TINYINT,
                    reward FLOAT,
                    delta_reward FLOAT,
                    strategy CHAR(20),
                    PRIMARY KEY (trialid))
                    ENGINE=MyISAM DEFAULT CHARSET=latin1"""

sql_insertvals = """INSERT INTO strategy_tag(
                    fix_port,
                    surebet_port,
                    lottery_port,
                    poke,
                    reward,
                    delta_reward,
                    strategy) VALUES (
                    "%d", "%d", "%d", "%d", "%f", "%f", "%s")"""
try:
    cur.execute(sql_createtbl)
except mdb.Warning:
    pass

# remove the previous records in SQL table.
# This is more for debugging purpose
sql_overwrite = "DELETE FROM strategy_tag"
cur.execute(sql_overwrite)
sql_reset = "ALTER TABLE strategy_tag AUTO_INCREMENT = 1"
cur.execute(sql_reset)
try:
    con.commit()
except:
    con.rollback()

paras = list(pre_conf) + [pre_action, pre_reward,
                          pre_delta_reward, 'initialize']
sql_insert = sql_insertvals % tuple(paras)

cur.execute(sql_insert)

num_trials = 10000

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

    paras = list(cur_conf) + [ac, re, delta_re, s]
    sql_insert = sql_insertvals % tuple(paras)
    cur.execute(sql_insert)

    pre_conf = cur_conf
    pre_action = ac
    pre_reward = re
    pre_delta_reward = delta_re

con.close()
