import db_connect as dbc
import numpy as np

NUM_PORTS = 2

if __name__ == '__main__':
    # generate configuration of the simulated trials
    # fix_port is a constant, while lottery and surebet
    # can alternate.
    # NO poke to fix_port is allowed

    cur, con = dbc.connect()
    # reset the table
    try:
        cur.execute("DELETE FROM config")
        cur.execute("ALTER TABLE config AUTO_INCREMENT = 1")
        con.commit()
    except:
        con.rollback()

    sqlstr = """INSERT INTO config(
                surebet_port,
                lottery_port,
                lottery_mag,
                lottery_prob,
                sure_mag) VALUES(
                "%d", "%d", "%f", "%f", "%f")"""
    # number of trials
    num_trial = 50000
    # number of lottery mag e.g. 1-10
    num_lott_mag = 20
    lott_mag_start = 4.0
    lott_mag_end = 12.0
    delta = (lott_mag_end - lott_mag_start)/num_lott_mag
    lott_prob = 0.5
    sure_mag = 5.0

    for i in range(num_trial):
        randports = list(np.random.choice(NUM_PORTS, NUM_PORTS, replace=False))
        lott_mag = lott_mag_start + int(i/(num_trial/num_lott_mag))*delta
        inputs = randports + [lott_mag, lott_prob, sure_mag]
        sqlcmd = sqlstr % tuple(inputs)
        cur.execute(sqlcmd)

    cur.close()