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
    # try:
    #     cur.execute("DELETE FROM strattag_config")
    #     cur.execute("ALTER TABLE strattag_config AUTO_INCREMENT = 1")
    #     con.commit()
    # except:
    #     con.rollback()

    sqlstr = """INSERT INTO strattag_config(
                surebet_port,
                lottery_port,
                lottery_mag,
                lottery_prob,
                sure_mag,
                alpha,
                beta) VALUES(
                "%d", "%d", "%f", "%f", "%f", "%f", "%f")"""
    # number of trials
    num_trial = 100000
    # number of lottery mag e.g. 1-10
    # num_lott_mag = 20
    # lott_mag_start = 4.0
    # lott_mag_end = 12.0
    # delta = (lott_mag_end - lott_mag_start)/num_lott_mag
    # lott_prob = 0.5
    # sure_mag = 5.0
    # riskyVals = np.array([4, 5, 6, 7, 8, 10, 12,
    #                       14, 16, 19, 23, 27, 31,
    #                       37, 44, 52, 61, 73, 86, 101, 120])
    # riskyVals = np.random.choice(np.arange(1, 150), 50, replace=False)
    # riskyProb = np.array([0.25, 0.5, 0.75])
    with open('utility_parameter.json', 'r') as f:
        utility_para = json.load(f)
    riskyVals = np.array(utility_para['lottery_mag'])
    riskyProb = np.array(utility_para['lottery_prob'])
    sure_mag = utility_para['sure_mag']
    alpha = utility_para['alpha']
    beta = utility_para['beta']
    num_riskyVals = len(riskyVals)
    num_riskyProb = len(riskyProb)

    for i in range(num_trial):
        randports = list(np.random.choice(NUM_PORTS, NUM_PORTS, replace=False))
        #lott_mag = lott_mag_start + int(i/(num_trial/num_lott_mag))*delta
        idx_val = np.random.randint(0, num_riskyVals)
        idx_prob = np.random.randint(0, num_riskyProb)
        lott_mag = riskyVals[idx_val]
        lott_prob = riskyProb[idx_prob]
        inputs = randports + [lott_mag, lott_prob, sure_mag, alpha, beta]
        sqlcmd = sqlstr % tuple(inputs)
        cur.execute(sqlcmd)

    cur.close()
