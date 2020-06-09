import math
from random import random
import numpy as np
import matplotlib.pyplot as plt

#buf_size = 100
buf_size = 4

#lyambda = np.arange(0.1, 1.5, 0.1)
lyambda = np.arange(0.5, 1.5, 0.1)
mes_count = 100000


def get_e(lyambda):
    return (-1 / lyambda) * math.log(random())


def gen_mes(lyambda_i):
    m = 0;
    p = 1
    L = math.exp(-lyambda_i)
    while p > L:
        m += 1
        u = random()
        p = p * u
    return m - 1

    count = 0
    lam = 0
    sumLam = 0

    while sumTao < 1:
        lam = get_e(lyambda_i)
        sumLam += lam
        if sumLam < 1:
            count += 1

    return count


def append_to_buf(buf, num_mes):
    while num_mes > 0 and len(buf) < buf_size:
        buf = np.append(buf, 0)
        num_mes -= 1
    return buf


def get_prob(i, L):
    return math.pow(L, i) * np.exp(-L) / math.factorial(i)


def gen_matrix(L):
    states_num = buf_size + 1

    P = np.zeros([states_num, states_num], dtype=float)

    for i in range(states_num - 1):
        P[0, i] = P[1, i] = get_prob(i, L)
    P[0, states_num - 1] = 1 - sum(P[0])
    P[1, states_num - 2] = 0

    for i in range(2, states_num - 1):
        deg = 0
        for j in range(i - 1, states_num - 2):
            P[i, j] = get_prob(deg, L)
            deg += 1

    for i in range(1, states_num):
        P[i, states_num - 2] = 1 - sum(P[i])

    P[states_num - 1, states_num - 2] = 1
    
    #print('Matr: ', P)
    
    for i in range(states_num):
        P[i, i] -= 1
        
    P = np.transpose(P)
    P[states_num - 1] = np.array([1] * states_num)

    return P


d = np.zeros(len(lyambda), dtype=int)
avg_N = [0] * len(lyambda)
d_theor = np.zeros(len(lyambda), dtype=float)
avg_N_theor = [0] * len(lyambda)

for i in range(len(lyambda)):
    sended = 0
    t = 0
    N = 0
    buf = np.array([])
    total_mes = 0
    empty = True

    while sended < mes_count:
        mes_per_window = gen_mes(lyambda[i])
        total_mes += mes_per_window
        buf = append_to_buf(buf, mes_per_window)

        if not empty:
            sended += 1
            d[i] += buf[0]
            buf = buf[1:]

        t += 1
        N += len(buf)
        buf += 1
        if len(buf) > 0:
            empty = False
        else:
            empty = True

    avg_N[i] = N / t

    P = gen_matrix(lyambda[i])
    #print('Matr: ', P)
    vector = np.zeros((buf_size + 1), dtype=int)
    vector[buf_size] = 1
    Pi = np.linalg.solve(P, vector)
    for j in range(0, len(Pi)):
        avg_N_theor[i] += j * Pi[j]

    l_out = 1 - Pi[0] #sended / total_mes
    d_theor[i] = avg_N_theor[i] / l_out

d = d / mes_count

print("d    : " + str(d.tolist()))
print("d (t): " + str(d_theor.tolist()))
print("N    : " + str(avg_N))
print("N (t): " + str(avg_N_theor))

figure = plt.figure()
plt.subplot(211)
plt.plot(lyambda, d, color = "g", label="d")
plt.plot(lyambda, d_theor, color = "m", label="d (t)")
plt.legend()
plt.subplot(212)
plt.plot(lyambda, avg_N, color = "g", label="N")
plt.plot(lyambda, avg_N_theor, color = "m", label="N (t)")
plt.legend()
plt.show()
