from mpi4py import MPI
import numpy as np
import sys

NUM_SAMPLE_TOTAL = 40
NUM_FEATURE = 4
PROCS = 4
ROOT = 0


def idx_sampling(num_trials, num_sample, num_procs):
    ids = np.arange(num_trials)
    # np.random.shuffle(ids)
    idx_ranks = -np.ones((num_procs, num_sample))
    for i in range(num_procs):
        idx_ranks[i] = ids[i*num_sample:(i+1)*num_sample]
    return idx_ranks.astype(int)


if __name__ == '__main__':
    dummy = np.arange(NUM_FEATURE*NUM_SAMPLE_TOTAL)
    data = dummy.reshape((NUM_SAMPLE_TOTAL, NUM_FEATURE))
    num_trials = len(data)
    num_sample = 5

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    idx_rank = np.zeros(num_sample).astype(int)

    if size != PROCS:
        print PROCS, " cores needed"
        sys.exit(1)

    if rank == ROOT:
        # idx = list(idx_sampling(num_trials, num_sample, size))
        idx = idx_sampling(num_trials, num_sample, size)
    else:
        idx = []

    idx_rank = comm.scatter(idx, root=ROOT)
    mappings = np.sum(np.sin(idx_rank))
    print rank, mappings

    sin_map = np.zeros(4)
    sin_map = comm.gather(mappings, root=0)

    if rank == 0:
        print sin_map
