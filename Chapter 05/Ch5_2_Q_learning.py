import numpy as np
Q_table = np.zeros((state_size, action_size))

import random
epsilon = 0.3
if random.uniform(0, 1) < epsilon:
"""
- - - - - -
Explore: choose a random action
- - - - - -
"""
else:

"""
- - - - - -
Exploit: choose an action having the highest q-value.
- - - - - -
Q [state, action] = Q [state, action] + α (rewards+ γ × np.max(Q [new-state,:]) - Q [state, action])

"""