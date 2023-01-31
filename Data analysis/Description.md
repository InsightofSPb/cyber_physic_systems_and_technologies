# Introduction to Data Analysis with Python

## Task
There is a dataset which contains data about voltage, current and measurement time. Measurement period is 0.001s. The test signal period is 0.1s. \
It is necessary to load and prepare data, draw graphs of current and voltage, calculate estimated values of parameters L and R, calculate mean values and standart deviation of estimated values of parameters L and R.

## Main Theory
In this laboratory the DC motor model is used. This model is described by the following system of differential equations:
$$
\begin{equation*}
 \begin{cases}
   \frac{di}{dt} = \frac{1}{L}u - \frac{R}{L}i - c_e\Omega\\
   \frac{d\Omega}{dt} = \frac{c_e}{J}i - \frac{1}{J}M_d
 \end{cases}
\end{equation*}
$$
where <i>u</i> is voltage, <i>i</i> is the current, <i>$\Omega$</i> is the rotation speed of the motor, <i>M_d</i> is the disturbance torque, <i>R</i> is the resistance, <i>L</i> is the inductance of the motor, <i>J</i> is the monet of inertia. 
\
In the laboratory work the simplified solution of the problem is considered. The estimation of the paramers L and R and considered with the use of the test data, where $\Omega=0$. Thus, only the first equation is interesting: $\frac{di}{dt} = \frac{1}{L}u - \frac{R}{L}i$ \
The voltage changes with the period of T_d = 0.001s, the current is measured with the same period.The equation of the transient process for any time interval has the form:\
$i(t) = (\frac{u}{R} - i(0))^{e^{\frac{-R}{L}t}}+\frac{u}{R}$\
and for adjacent measurement intervals:
$i(n) = (\frac{u(n-1)}{R} - i(n-1))^{e^{\frac{-R}{L}T_d}}+\frac{u(n-1)}{R}$\
where <i>n</i> is the number of the measurement interval.\
\
The above formula is linear regression. It can be rewritten as follows:\
$i(n) = K_1u(n-1) + K_2i(n-1) +\varepsilon$ where\
$K_1 = \frac{1 - e^{\frac{-R}{L}Td}}{R}$\
$K_2 = e^{\frac{-R}{L}Td}$\
$\varepsilon$ is sensor measurement noise.\
The matrix of factors is written as follows:
```math
$X = \begin{bmatrix}
u(1)& i(1)\
u(2)& i(2)\
:& :\
u(n-1)& i(n-1)
\end{bmatrix}$\
```
The vector of the considered variable looks like $y = [i(2),i(3),...,i(n)]^T$\
This equation $i(n) = K_1u(n-1) + K_2i(n-1) +\varepsilon$ can be written as follows for a series of measurements:\
$y = X * k + \varepsilon$ and the parametr <i>k</i> can by found by OLS method (ordinary least squares method):
$k = (X^TX)^{-1}X^Ty$\
Estimations of the unknown parameters are:\
$\hat{R} = \frac{1 - \hat{k}_2}{\hat{k}_1}$\
$\hat{T_e} = -\frac{T_d}{ln \hat{k}_2}$\
$\hat{L} = \hat{T_e} * \hat{R}$

## Practical part
#### Loading librarues
```python
# ввод библиотек
import numpy as nump
from numpy import linalg as la
import matplotlib.pyplot as pl
```
#### Loading and preparing data
```python
# загрузка и подготовка данных
values = nump.genfromtxt('testLab1Var23.csv', delimiter=',')
t = values[:, 0] # time 
t = t[:, nump.newaxis]
cur = values[:, 1] # current
cur = cur[:, nump.newaxis]
volt = values[:, 2] # voltage
volt = volt[:, nump.newaxis]
print('Напряжение:', int(volt[10]), end=';')
print(' время:', float(t[10]))
```
#### Drawing graphs of the current and voltage
```python
# построение графиков тока и напряжения
figure, (f1,f2) = pl.subplots(2, 1, sharex=True) # sharex - same scale along X
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
```
![](https://github.com/InsightofSPb/cyber_physic_systems_and_technologies/blob/main/Data%20analysis/Data(part).png?raw=true)



#### Calculation of the estimated values of parameters L and R
```python
# расчёт значений и параметров L и R
X = nump.concatenate([volt[0:len(volt) - 2], cur[0:len(cur) - 2]], axis = 1) # calc of X-matrix (matrix of factors)
Y = cur[1:len(cur) - 1]
K = nump.dot(nump.dot(la.inv(nump.dot(X.T, X)), X.T), Y) # calc of the parametr k of the linear regression OLS
Td = 0.001
R = 1 / K[0] * (1 - K[1]) # estimate of resistance
T = -Td / nump.log(K[1]) 
L = T * R # estimate of inductance
cur_mod = X.dot(K)
print('Сопротивление =', float(R), '', 'Индуктивность = ', float(L))
# comparison of the initial data and estimated
figure, f3 = pl.subplots(1,1)
pl.plot(t[t < T_period], cur[t < T_period], label='I_initial')
pl.plot(t[t < T_period], cur_mod[t[0:len(cur) - 2] < T_period], label='I_estimated')
f3.grid()
f3.set_xlabel('time,s')
f3.set_ylabel('current,A')
pl.show()
figure.savefig('Compared current')
```
![](https://github.com/InsightofSPb/cyber_physic_systems_and_technologies/blob/main/Data%20analysis/Compared%20current.png?raw=true)


#### Calculation of the mean values and st. deviation of the estimated R and L
```python
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
```
