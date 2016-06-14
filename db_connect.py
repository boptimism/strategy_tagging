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
