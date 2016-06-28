import MySQLdb as mdb


def connect():
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
    return cur, con


def overwrite(cur, con, tbl_name):
    try:
        cur.execute("DELETE FROM %s" % tbl_name)
        cur.execute("ALTER TABLE %s AUTO_INCREMENT=1" % tbl_name)
        con.commit()
    except:
        con.rollback()
