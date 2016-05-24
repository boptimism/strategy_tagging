import MySQLdb as mdb

credentials = {}

with open('./credential_db.txt','r') as f:
    for l in f:
        fields = l.strip().split(':')
        try:
            credentials[fields[0]]=fields[1]
        except:
            pass

con = mdb.connect(host=credentials['host'],
                  user=credentials['user'],
                  passwd=credentials['passwd'],
                  db=credentials['db'])

cur = con.cursor()

sql_createtbl = """CREATE TABLE IF NOT EXISTS strategy(
                   trialid INT UNSIGNED NOT NULL AUTO_INCREMENT,
                   fix_port TINYINT NOT NULL,
                   surebet_port TINYINT NOT NULL,
                   lottery_port TINYINT NOT NULL,
                   poke TINYINT,
                   reward FLOAT,
                   reward_change FLOAT,
                   strategy char(20),
                   PRIMARY KEY(trialid))
                   ENGINE=MyISAM DEFAULT CHARSET=latin1"""

cur.execute(sql_createtbl)

con.close()
