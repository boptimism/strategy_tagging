import numpy as np
import json

vs = np.random.choice(np.arange(1,150), 50, replace=False)
vs.sort()
vals = vs.tolist()
probs = [0.25, 0.5, 0.75]
alpha = 1.5
beta = 1.0
surebet = 5.0

data = {'lottery_mag': vals,
        'lottery_prob': probs,
        'sure_mag': surebet,
        'alpha': alpha,
        'beta': beta}
with open('utility_parameter.json','w') as f:
    json.dump(data, f)
