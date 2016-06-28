import db_connect as dbc
import numpy as np
import json

NUM_PORTS = 2


if __name__ == '__main__':
    # generate configuration of the simulated trials
    # fix_port is a constant, while lottery and surebet
    # can alternate.
    # NO poke to fix_port is allowed

    cur, con = dbc.connect()
    # reset the table
    dbc.overwrite(cur, con, 'strattag_config')

    sqlstr = """INSERT INTO strattag_config(
                surebet_port,
                lottery_port,
                lottery_mag,
                lottery_prob,
                sure_mag,
                alpha,
                beta) VALUES(
                "%d", "%d", "%f", "%f", "%f", "%f", "%f")"""
    with open('inputs.json', 'r') as f:
        para = json.load(f)
    riskyVals = np.array(para['lottery_mag'])
    riskyProb = np.array(para['lottery_prob'])
    sure_mag = para['sure_mag']
    alpha = para['alpha']
    beta = para['beta']
    num_trial = para['num_trial']
    num_riskyVals = len(riskyVals)
    num_riskyProb = len(riskyProb)

    for i in range(num_trial):
        randports = list(np.random.choice(NUM_PORTS, NUM_PORTS, replace=False))
        idx_val = np.random.randint(0, num_riskyVals)
        idx_prob = np.random.randint(0, num_riskyProb)
        lott_mag = riskyVals[idx_val]
        lott_prob = riskyProb[idx_prob]
        inputs = randports + [lott_mag, lott_prob, sure_mag, alpha, beta]
        sqlcmd = sqlstr % tuple(inputs)
        cur.execute(sqlcmd)

    cur.close()
