import numpy as np
import json

vs = np.random.choice(np.arange(1, 150, 0.5), 150, replace=False)
vs.sort()
vals = vs.tolist()
probs = [0.25, 0.5, 0.75]
alpha = 1.5
beta = 1.0
surebet = 5.0
prior = {'randombet': 0.075,
         'sameport': 0.025,
         'samebet': 0.04,
         'winstayloseshift': 0.06,
         'utility': 0.8}

data = {'lottery_mag': vals,
        'lottery_prob': probs,
        'sure_mag': surebet,
        'alpha': alpha,
        'beta': beta,
        'prior': prior}
with open('parameter_setup.json', 'w') as f:
    json.dump(data, f)
