#ввод библиотек
import numpy as nump
from numpy import linalg as la
import matplotlib.pyplot as pl

#загрузка и подготовка данных
values = nump.genfromtxt('testLab1Var23.csv',delimiter=',')
t = values[:, 0]
t = t[:, nump.newaxis]
cur = values[:, 1]
cur = cur[:, nump.newaxis]
volt = values[:, 2]
volt = volt[:, nump.newaxis]
print('Напряжение:', int(volt[10]), end=';')
print(' время:', float(t[10]))

#построение графиков тока и напряжения
figure, (f1,f2) = pl.subplots(2, 1, sharex=True)
T_period = 0.1
f1.plot(t[t < 2 * T_period],volt[t < 2 * T_period])
f1.grid()
f1.set_xlabel('time,s')
f1.set_ylabel('voltage,V')
f2.plot(t[t < 2 * T_period],cur[t < 2 * T_period])
f2.grid()
f2.set_xlabel('time,s')
f2.set_ylabel('current,A')
pl.show()
figure.savefig('Data(part)')

#расчёт значений и параметров L и R
X = nump.concatenate([volt[0:len(volt) - 2], cur[0:len(cur) - 2]], axis = 1)
Y = cur[1:len(cur) - 1]
K = nump.dot(nump.dot(la.inv(nump.dot(X.T, X)), X.T), Y)
Td = 0.001
R = 1 / K[0] * (1 - K[1])
T = -Td / nump.log(K[1])
L = T * R
cur_mod = X.dot(K)
print('Сопротивление =', float(R), '', 'Индуктивность = ', float(L))

figure, f3 = pl.subplots(1,1)
pl.plot(t[t < T_period], cur[t < T_period], label='I_initial')
pl.plot(t[t < T_period], cur_mod[t[0:len(cur) - 2] < T_period], label='I_estimated')
f3.grid()
f3.set_xlabel('time,s')
f3.set_ylabel('current,A')
f3.legend()
pl.show()
figure.savefig('Compared current')

#Расчёт средних значений и стандартного отколнения значений параметров L и R
R_mod = []
L_mod = []
n = 1000
for i in range(0, n - 1):
    ind = (t >= T_period * i) & (t <= T_period * (i + 1))
    new_cur = cur[ind]
    new_cur = new_cur[:,nump.newaxis]
    new_volt = volt[ind]
    new_volt = new_volt[:,nump.newaxis]
    X = nump.concatenate([new_volt[0:len(new_volt) - 2], new_cur[0:len(new_cur) - 2]], axis = 1)
    Y = new_cur[1:len(new_cur) - 1]
    K = nump.dot(nump.dot(la.inv(nump.dot(X.T, X)), X.T), Y)

    if K[1] > 0:
        R = 1 / K[0] * (1 - K[1])
        T = -Td / nump.log(K[1])
        R_mod.append(R)
        L_mod.append(T * R)

R_mod = nump.array(R_mod)
L_mod = nump.array(L_mod)

print('Mean value of R:', nump.mean(R_mod), 'Ohm')
print('Standart deviation of R:', nump.std(R_mod), 'Ohm')
print('Mean value of L:', nump.mean(L_mod), 'Hn')
print('Standart deviation of L:', nump.std(L_mod), 'Hn')