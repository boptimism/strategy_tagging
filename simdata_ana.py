import sys
import argparse
import time
import db_connect as dbc
import simdata_postproc_func as pfunc
import numpy as np
from scipy import optimize as opt
from mpi4py import MPI


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
    assert np.all(est_p <= 1.0) and np.all(est_p >= 0.0)
    # test1 = np.all(np.log(est_p) < 0)
    # test2 = np.all(np.log(1.-est_p) < 0)
    # if not test1 or not test2:
    #     print external, est_p
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
        # p_i: actually poke
        p_i = yi[2]
        external = y0[:4] + yi[:2] + yi[-3:]
        # q_i: theoretical predictions
        q_i = prob_poke(coeff, alpha, beta, external)
        p_vec = np.zeros(NUM_PORTS)
        p_vec[p_i] = 1.0
        loss = np.dot(p_vec, np.log(q_i+eps))
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
    parser = argparse.ArgumentParser(
        description='Fitting Utility paramerters and likelyhoodof strategies')
    parser.add_argument('-o', '--overwriteSQL=', type=int,
                        dest='overwriteSQL',
                        help='flag to overwrite existing SQL table')
    parser.add_argument('-t', '--testloss=', type=int, default=0,
                        dest='testloss',
                        help="""Flag to split train-test, and run test loss.
                                Default to False""")
    parser.add_argument('-r', '--testratio=', type=float, default=0.1,
                        dest='testratio',
                        help="""\
                        Percentage of test set over (test+training),
                        Must be used jointly with -t 1 (--testloss=1).
                        Otherwise this option is not called.
                        Default to 0.1""")
    args = parser.parse_args()
    overwriteSQL = args.overwriteSQL
    testloss = args.testloss
    testratio = args.testratio

    # read SQL db from multiple processor
    t0 = time.time()
    records = read_trials()
    t1 = time.time()
    print 'rank {0:2d} sql-reading takes {1:.1f} sec'.format(
        rank, t1-t0)
    num_samples_total = len(records)

    if testloss:
        sqlstr = """INSERT INTO strattag_fitting(
                    rank,
                    train_samples,
                    test_samples,
                    fitting_time,
                    random,
                    utility,
                    samebet,
                    sameport,
                    wsls,
                    alpha,
                    beta,
                    entloss_train,
                    entloss_test) VALUES(
                    "%d", "%d", "%d",
                    "%f", "%f", "%f",
                    "%f", "%f", "%f",
                    "%f", "%f", "%f",
                    "%f")
                    """
    else:
        sqlstr = """INSERT INTO strattag_fitting(
            rank,
            train_samples,
            fitting_time,
            random,
            utility,
            samebet,
            sameport,
            wsls,
            alpha,
            beta,
            entloss_train) VALUES(
            "%d", "%d",
            "%f", "%f", "%f",
            "%f", "%f", "%f",
            "%f", "%f", "%f"
            )
            """
    # number of sample size used for fitting
    samplesizes = [2500]
    # number of fitting to do on each proc
    fitperprocs = [1]
    # connect to db to record the fittings
    cur, con = dbc.connect()
    if rank == ROOT and overwriteSQL:
        dbc.overwrite(cur, con, 'strattag_fitting')

    for sample_size, fit_per_proc in zip(samplesizes, fitperprocs):
        # split between training and testing
        # default: using 0% of training as testing
        if testloss:
            num_test_samples = int(sample_size * testratio)
            num_train_samples = sample_size - num_test_samples
        else:
            num_train_samples = sample_size
        sample_per_proc = sample_size*fit_per_proc
        num_samples_need = sample_per_proc*numtask

        if num_samples_need > num_samples_total:
            if rank == ROOT:
                print 'Insufficient samples.'
                print 'At least ', num_samples_need, 'samples needed'
                print 'Number of samples given: ', num_samples_total
            MPI.Finalize()
            sys.exit(1)

        # indices of records required by each processor
        idx_per_proc = np.zeros(sample_per_proc).astype(int)

        if rank == ROOT:
            data_seq = np.arange(num_samples_total)
            # SHOULD NOT SHUFFLE, since trial is dependent on HISTORY!
            # np.random.shuffle(data_seq)
            data_seq_use = data_seq[:num_samples_need]
            data_idx_ranks = data_seq_use.reshape((numtask, sample_per_proc))
        else:
            data_idx_ranks = []

        idx_per_proc = comm.scatter(data_idx_ranks, root=ROOT)
        data_per_proc = [-1]*sample_per_proc
        for i, j in enumerate(idx_per_proc):
            data_per_proc[i] = records[j]

        for i in range(fit_per_proc):

            data_to_use = data_per_proc[i*sample_size:(i+1)*sample_size]
            training = data_to_use[:num_train_samples]
            data_to_fit = (tuple(training),)
            if testloss:
                testing = data_to_use[num_train_samples:]
                data_to_test = (tuple(testing), )

            t4 = time.time()
            methds = 'SLSQP'
            # initial guess
            paras = tuple([0.2]*NUM_STRATEGY) + (2.0, 0.5)

            methds_opt = {'disp': True}
            res = opt.minimize(cross_ent_loss,
                               paras,
                               args=data_to_fit,
                               method=methds,
                               # callback=callbackF,
                               options=methds_opt)
            t5 = time.time()
            fit_time = t5-t4

            # computing cross entropy losses
            fitting = res.x
            # fitting[:5] = np.abs(fitting[:5])/sum(np.abs(fitting[:5]))
            train_loss = cross_ent_loss(fitting, data_to_fit[0])
            print 'train size: {0:d},\
                   train-loss: {1:.6g}'.format(num_train_samples, train_loss)
            if testloss:
                test_loss = cross_ent_loss(fitting, data_to_test[0])
                print 'test size: {0:d},\
                       test-loss: {1:.6g}'.format(num_test_samples, test_loss)
            if not testloss:
                sql_ins = (rank,
                           num_train_samples,
                           fit_time) + tuple(fitting) + (train_loss, )
                sqlcmd = sqlstr % sql_ins
                cur.execute(sqlcmd)
            else:
                sql_ins = (rank,
                           num_train_samples,
                           num_test_samples,
                           fit_time) + tuple(fitting) + (train_loss, test_loss)
                sqlcmd = sqlstr % sql_ins
                cur.execute(sqlcmd)

    con.close()
    MPI.Finalize()
