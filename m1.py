import numpy as np
from random import random
import matplotlib.pyplot as plt

P = np.array([[0.8, 0.2],
             [0.4, 0.6]])

t = 20
N = 100000


p_t = [0] * t

for n in range(N):
    state = 0
    for i in range(t):
        p_t[i] += state ^ 1
        next_distr_p = random()
        if next_distr_p < P[state][0]:
            state = 0
        else:
            state = 1

p_t[:] = (i / N for i in p_t)

distr = np.array([1, 0])
p_t_theor = [0] * t

for i in range(t):
    p_t_theor[i] = distr[0]
    distr = distr.dot(P)


fig = plt.figure()
plt.plot(range(0, t), p_t, color="g", label="p_t")
plt.plot(range(0, t), p_t_theor, color="m", label="p_t_theor")
plt.legend()
plt.show()

