import numpy as np
import json

# magnitude of lottery bet
vs = np.random.choice(np.arange(1, 150, 0.5), 150, replace=False)
vs.sort()
vals = vs.tolist()
# probabilities of lottery bet
probs = [0.25, 0.5, 0.75]
# utility risk
alpha = 1.5
# utility temperature
beta = 1.0
# surebet magnitude
surebet = 5.0
# prior of the weights of each strategy
prior = {'randombet': 0.075,
         'sameport': 0.025,
         'samebet': 0.04,
         'winstayloseshift': 0.06,
         'utility': 0.8}
# number of simulated record to be generated
num_trial = 50000
# array of samplesize , to check convergence
samplesizes = [5000, 2500]
# number of fittings done per proc
fitperprocs = [1, 2]

data = {'lottery_mag': vals,
        'lottery_prob': probs,
        'sure_mag': surebet,
        'alpha': alpha,
        'beta': beta,
        'prior': prior,
        'num_trial': num_trial,
        'samplesizes': samplesizes,
        'fitperprocs': fitperprocs}

with open('inputs.json', 'w') as f:
    json.dump(data, f)
