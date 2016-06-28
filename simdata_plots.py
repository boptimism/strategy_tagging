import numpy as np
import db_connect as dbc
import matplotlib.pyplot as plt
import json

with open('inputs.json', 'r') as f:
    para = json.load(f)
prior = para['prior']
alpha = para['alpha']
beta = para['beta']
random = prior['randombet']
utility = prior['utility']
samebet = prior['samebet']
sameport = prior['sameport']
wsls = prior['winstayloseshift']

samplesizes = para['samplesizes']
cur, con = dbc.connect()
sqlstr = """SELECT
            train_samples,
            COUNT(*),
            AVG(random),
            STD(random),
            AVG(utility),
            STD(utility),
            AVG(samebet),
            STD(samebet),
            AVG(sameport),
            STD(sameport),
            AVG(wsls),
            STD(wsls),
            AVG(alpha),
            STD(alpha),
            AVG(beta),
            STD(beta),
            AVG(fitting_time),
            STD(fitting_time),
            AVG(entloss_train),
            STD(entloss_train)
            FROM strattag_fitting
            GROUP BY train_samples"""
cur.execute(sqlstr)
stats = np.array(cur.fetchall())
con.close()
stats.sort(axis=0)
num_samples = stats[:, 0]
num_copies = stats[:, 1]
avg_random = stats[:, 2]
std_random = stats[:, 3]
avg_utility = stats[:, 4]
std_utility = stats[:, 5]
avg_samebet = stats[:, 6]
std_samebet = stats[:, 7]
avg_sampport = stats[:, 8]
std_sameport = stats[:, 9]
avg_wsls = stats[:, 10]
std_wsls = stats[:, 11]
avg_alpha = stats[:, 12]
std_alpha = stats[:, 13]
avg_beta = stats[:, 14]
std_beta = stats[:, 15]
avg_train_loss = stats[:, 18]
std_train_loss = stats[:, 19]
print ''
print '{0:20s} {1:20s}'.format('sample_size', 'number_fits')
for i in range(len(num_samples)):
    print '{0:<20d} {1:<20d}'.format(int(num_samples[i]), int(num_copies[i]))

fig = plt.figure()
plt.plot(num_samples, avg_random, '-o')
plt.fill_between(num_samples,
                 avg_random + 0.5*std_random,
                 avg_random - 0.5*std_random,
                 interpolate=True,
                 facecolor='#D1E9D1',
                 edgecolor='#D1E9D1',
                 alpha=0.5)
plt.show()
