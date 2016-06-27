import db_connect as dbc
import simdata_postproc_func as pfunc
import numpy as np
from scipy import optimize as opt
import time
from mpi4py import MPI
import sys

ROOT = 0
NUM_STRATEGY = 5
NUM_PORTS = 2


def prob_poke(coeff, alpha, beta, external):
    # coeff : the priors for each strategy
    # external: a list contains:
    # history, lott_mag, lott_prob, sure_mag
    # history: a list contains:
    # pre_fix, pre_sure, pre_lott, pre_poke, pre_reward,
    # cur_fix, cur_sure, cur_lott
    history = external[0:6]
    lott_mag = external[6]
    lott_prob = external[7]
    sure_mag = external[8]
    p1 = pfunc.randombet()
    p2 = pfunc.utility(history,
                       lott_mag, lott_prob, sure_mag,
                       alpha, beta)
    p3 = pfunc.sameport(history)
    p4 = pfunc.samebet(history)
    p5 = pfunc.winstayloseshift(history)

    est_p = np.dot(coeff, np.array([p1, p2, p3, p4, p5]))
    assert np.all(np.log(est_p) < 0)
    assert np.all(np.log(1.-est_p) < 0)
    return est_p


def cross_ent_loss(x, y):
    # x: coeff , alpha, beta
    # y: external controls
    #    a matrix with dimension n_trial X 7
    #    each row consists of :
    #    sure_port, lottery_port, poke, reward,
    #    lott_mag, lott_prob, sure_mag
    eps = 1.e-15
    coeff = np.abs(x[:NUM_STRATEGY])/sum(np.abs(x[:NUM_STRATEGY]))
    alpha = x[NUM_STRATEGY]
    beta = x[NUM_STRATEGY+1]
    entloss = 0.0
    y0 = y[0]
    for yi in y[1:]:
        p_i = yi[2]  # p_i: actually poke
        external = y0[:4] + yi[:2] + yi[-3:]
        q_i = prob_poke(coeff, alpha, beta, external)
        p_vec = np.zeros(NUM_PORTS)
        p_vec[p_i] = 1.0
        loss = np.dot(p_vec, np.log(q_i+eps))\
            + np.dot(1.-p_vec, np.log(1.-q_i + eps))
        entloss = entloss + loss
        y0 = yi
    entloss = -entloss/len(y)
    return entloss


def callbackF(x):
    likelyhood = np.abs(x[:NUM_STRATEGY])/sum(np.abs(x[:NUM_STRATEGY]))
    x_check = list(likelyhood) + [x[-2], x[-1]]
    print '{0:<8.3f} \
           {1:<8.3f} \
           {2:<8.3f} \
           {3:<8.3f} \
           {4:<8.3f} \
           {5:<8.3f} \
           {6:<8.3f}'.format(*x_check)


def read_trials():
    cur, con = dbc.connect()
    sqlcmd = """SELECT
                a.surebet_port, a.lottery_port,
                b.poke, b.reward,
                a.lottery_mag, a.lottery_prob, a.sure_mag
                FROM strattag_config AS a
                JOIN strattag_pokereward AS b
                ON a.trialid=b.trialid
                """
    cur.execute(sqlcmd)
    records = cur.fetchall()
    con.close()
    return records


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    numtask = comm.Get_size()
    rank = comm.Get_rank()

    # read SQL db from multiple processor

    t0 = time.time()
    records = read_trials()
    t1 = time.time()
    print 'rank {0:2d} sql-reading takes {1:.1f} sec'.format(
        rank, t1-t0)

    num_samples_total = len(records)
    sqlstr = """INSERT INTO strattag_fitting(
                rank,
                num_samples,
                fitting_time,
                random,
                utility,
                samebet,
                sameport,
                wsls,
                alpha,
                beta) VALUES(
                "%d", "%d", "%f", "%f", "%f", "%f", "%f", "%f", "%f", "%f")
                """
    # number of sample size used for fitting
    samplesizes = [100000]
    # number of fitting to do on each proc
    fitperprocs = [1]
    # connect to db to record the fittings
    cur, con = dbc.connect()
    if rank == ROOT:
        dbc.overwrite(cur, con, 'strattag_fitting')

    fitting_time_total = 0.0
    writetodb_time_total = 0.0

    for sample_size, fit_per_proc in zip(samplesizes, fitperprocs):
        sample_per_proc = sample_size*fit_per_proc
        num_samples_use = sample_per_proc*numtask

        if num_samples_use > num_samples_total:
            if rank == ROOT:
                print 'Insufficient samples.'
                print 'At least ', num_samples_use, 'samples needed'
            MPI.Finalize()
            sys.exit(1)

        idx_per_proc = np.zeros(sample_per_proc).astype(int)

        if rank == ROOT:
            data_seq = np.arange(num_samples_total)
            np.random.shuffle(data_seq)
            data_seq_use = data_seq[:num_samples_use]
            data_idx_ranks = data_seq_use.reshape((numtask, sample_per_proc))
        else:
            data_idx_ranks = []

        idx_per_proc = comm.scatter(data_idx_ranks, root=ROOT)
        data_per_proc = [-1]*sample_per_proc
        for i, j in enumerate(idx_per_proc):
            data_per_proc[i] = records[j]

        for i in range(fit_per_proc):

            data_to_fit = data_per_proc[i*sample_size:(i+1)*sample_size]
            data_to_fit = (tuple(data_to_fit),)

            t4 = time.time()
            methds = 'SLSQP'
            # initial guess
            paras = tuple([0.2]*NUM_STRATEGY) + (2.0, 0.5)

            methds_opt = {'disp': False}
            res = opt.minimize(cross_ent_loss,
                               paras,
                               args=data_to_fit,
                               method=methds,
                               callback=callbackF,
                               options=methds_opt)
            t5 = time.time()

            fitting = res.x
            fitting[:5] = np.abs(fitting[:5])/sum(np.abs(fitting[:5]))
            fit_time = t5-t4

            fitting_time_total += fit_time

            sql_ins = (rank, sample_size, fit_time) + tuple(fitting)
            sqlcmd = sqlstr % sql_ins

            t6 = time.time()
            cur.execute(sqlcmd)
            t7 = time.time()
            writetodb_time_total += (t7-t6)

    print 'rank {0:2d}: \
           Total Fitting {1:.1f}, \
           Total Write2DB {2:.1f}, \
           Write/Fit {3:.4f}'.\
        format(rank,
               fitting_time_total,
               writetodb_time_total,
               writetodb_time_total/fitting_time_total)
    con.close()
    MPI.Finalize()
