# cyber_systems_and_technologies
Laboratory work on the subject Cyber-physical systems and technologies "Introduction to Neural Network Design with Python"

## Task
Design neural network which is able to classify handwritten numbers from well-known MNIST digit dataset with Python. \
Спроектировать нейронную сеть, написанную на Python, способную распознавать изображения рукописных чисел из известного датасета

## Main theory

Artificial neural networks are made up of neurons, so it is a good idea to start with a mathematical model of a neuron. An artificial neuron is a simplified model of a natural neuron and it has many inputs with its own weight(w<sub>n</sub>) for each input. Inside the neuron there is an activisation function F(S), where S - the sum of inputs. The result of calculating this function became the output of the neuron. The picture is below.

![аа](https://neerc.ifmo.ru/wiki/images/a/a5/%D0%98%D1%81%D0%BA%D1%83%D1%81%D1%81%D1%82%D0%B2%D0%B5%D0%BD%D0%BD%D1%8B%D0%B9_%D0%BD%D0%B5%D0%B9%D1%80%D0%BE%D0%BD_%D1%81%D1%85%D0%B5%D0%BC%D0%B0.png)

Thus, a neuron can be mathematically described:
$$ Y = F(W * X),\; where \\
W = [w_1 \; w_2 \; w_3 \; ... \; w_n] \;\;\; is \; a \; vector\;of\;weights\;of\;input\;signals\\
X = [x_1 \; x_2 \; x_3 \; ... \; x_n]^T\;\;\;is\;the\;column\;vector\;of\;input\;signal\;values.$$

Nonlinear functions are most often used as an activation function as only such neurons allow solving non-trivial problems with small number of nodes. The two most common functions are:
+ $F(S) = th(S)$, which is normalized to the interval [-1, 1];
+ $F(S) =\frac{1}{1 + e^{-S}}$, which is normalized to the interval [0,1]; 

Both functions are called sigmoid functions as they resemble S. In the laboratory the second one is used.

Neurons are collected in layers: input, hidden, output. First signals are sent to the input layer, then they are transmitted to hidden layers where information is processed according to the algorithm written above. From last hidden layer information is transferred to the output layer giving output signals which are the desired result of processing inpormation inside neural network.
![](https://neerc.ifmo.ru/wiki/images/b/b3/%D0%9D%D0%B5%D0%B9%D1%80%D0%BE%D0%BD%D0%BD%D0%B0%D1%8F_%D1%81%D0%B5%D1%82%D1%8C.png)

The output layer can be mathematically calculated as:
$O=F(W_s*I)$ \
where\
$W = [w_1 \; w_2 \; w_3 \; ... \; w_n]$ is a vector of input signal weights;\
$I = [i_1 \; i_2 \; i_3 \; ... \; i_n]^T$ is a column vector of layer input values;\
$O = [o_1 \; o_2 \; o_3 \; ... \; o_n]^T$ is a column vector of layer output values;\
$$
W = \left(\begin{array}{cc} 
w_{11} & w_{12} & ... & w_{1n}\\
w_{21} & w_{22} & ... & w_{2n}\\
: & : & : & :\\
w_{m1} & w_{m2} & ... & w_{mn}\\
\end{array}\right)
$$
is a matrix of weights for the input signals of the layer (<i>n</i> is the number of input signals, m is the number of neurons in the layer). 

In this work the training process can be described the next way: there is a set of test signals as an input signals <b>I</b> of the neural network and the set of outputs <b>O</b> is calculated which will be compared with the desired test results. The error of the output layer is calculated as follows:\
$E_{OL} =T_{OL} - O_{OL}$\
where\
$T = [t_1 \; t_2 \; t_3 \; ... \; t_n]^T$ is a column vector of the values of the desired test results of the neural network;\
$O_{OL} = [o_{OL.1} \; o_{OL.2} \; o_{OL.3} \; ... \; o_{OL.n}]^T$ is a column vector of layer output values;\
$E_{OL} = [e_{OL.1} \; e_{OL.2} \; e_{OL.3} \; ... \; e_{OL.n}]^T$ is a column vector of output layer error values.


The error of the hidden layer is calculated as follows:
$E_{HL} = {W_{HL}}^TE_{OL}$\
If there are more than one hidden layer, the error is calculated by sequental multiplication by transposed weight matrices.

The weights are changed using the gradient descent method:\
$\Delta w_{OL.jk} = -2 * \alpha * e_{OL.k} * o_{OL.k} * (1-o_{OL.k}) * o_{HL.j}$\
$\Delta w_{HL.jk} = -2 * \alpha * e_{HL.k} * o_{HL.k} * (1-o_{HL.k}) * i_j$

Here $\alpha$ is a learning factor, which characterizes the rate of change of the weights.

## Practical part


#### Import of necessary libraries
```python
import numpy as nump
import matplotlib.pyplot as plt
from scipy.special import expit as fmain
```
#### Function to set main parametrs of network
```python
# Функция для установки основных сетевых параметров
def ust_neur():
    input_nodes = 784  # 28x28 пикселей, размер каждого образца
    print('Input the number of hidden neurons:')
    hidden_nodes = int(input())  # кол-во нейронов в скрытом слое
    output_nodes = 10 # поскольку ожидается 10 выходных сигналов - на каждое число
    print('Input the training speed(0.5):')
    training_speed = float(input())
    return input_nodes, hidden_nodes, output_nodes, training_speed
```
#### Function to set initial weights
```python
# Функция установки начальных значений весов нейронной сети
# Поскольку начальные значения весов не даны, можно выбрать случайное число с помощью генератора
# равномерного распределения псевдослучайных чисел
def create_net(input_nodes, hidden_nodes, output_nodes):
    w_in2hidden = nump.random.uniform(-0.5, 0.5, (hidden_nodes, input_nodes)) 
    w_hidden2out = nump.random.uniform(-0.5, 0.5, (output_nodes, hidden_nodes))
    return w_in2hidden, w_hidden2out
```
#### Function to calculate the output of the neural network
To calculate output the next formula can be used:\
$O = F(W_{OL}*F(W_{HL}*I))$
```python
# Функция вычисления вывода нейронной сети
def net_output(w_in2hidden, w_hidden2out, input_signal, return_hidden):
    inputs = nump.array(input_signal, ndmin = 2).T # ndim - column two-dim vector
    hid_in = nump.dot(w_in2hidden,inputs) # calculatin Whl*I
    hid_out = fmain(hid_in) # fmain = sigmoid func of Whl * I
    fin_in = nump.dot(w_hidden2out,hid_out) # calc Wol * F(...)
    fin_out = fmain(fin_in) # calc F(...)
    if return_hidden == 0: # if False returns signals from the last and all other hidden layers
        return fin_out
    else:
        return fin_out, hid_out
```
#### Function to train network
```python
# Функция обучения нейронной сети
def net_train(targets, input_signal, w_in2hidden, w_hidden2out, learning_speed): # targets = desired outputs
    target = nump.array(targets, ndmin = 2).T
    inputs = nump.array(input_signal, ndmin = 2).T
    fin_out, hid_out = net_output(w_in2hidden, w_hidden2out, input_signal, 1) # calc outputs of network
    out_errors = target - fin_out # calc errors
    hid_errors = nump.dot(w_hidden2out.T, out_errors) # calc hidden layer errors

    w_hidden2out += learning_speed * nump.dot((out_errors * fin_out * (1 - fin_out)), hid_out.T)
    w_in2hidden += learning_speed * nump.dot((hid_errors * hid_out * (1 - hid_out)),inputs.T)
    return w_in2hidden, w_hidden2out
```
#### Function to train on real data
```python
# Функция обучения сети на тренировочных данных
def train_set(w_in2hidden, w_hidden2out, learning_speed):
    data = open(r'C:\Users\Пользователь\Documents\Обучение\Магистратура\1 семестр\Киберфизические системы и технологии'
                r'\ПР2\mnist_train.csv', 'r') # reading data
    training_list = data.readlines()
    data.close()
    for record in training_list:
        values = record.split(',')
        inputs = (nump.asfarray(values[1:]) / 255.0 * 0.999) + 0.001 # to normalize data in range [0.001; 1.000]
        # 0.001 and not 0 to avoid not updating weights
        target = nump.zeros(10) + 0.001
        target[int(values[0])] = 1.0 # real value
        net_train(target,inputs,w_in2hidden,w_hidden2out,learning_speed)
    return w_in2hidden, w_hidden2out
```
#### Verification function
```python
# Функция проверки сети
def test_set(w_in2hidden, w_hidden2out):
    data = open(r'C:\Users\Пользователь\Documents\Обучение\Магистратура\1 семестр\Киберфизические системы и технологии'
                r'\ПР2\mnist_test.csv', 'r')
    test_list = data.readlines()
    data.close()
    test = []
    for record in test_list:
        values = record.split(',')
        inputs = (nump.asfarray(values[1:]) / 255.0 * 0.999) + 0.001
        out_session = net_output(w_in2hidden, w_hidden2out, inputs, 0)
        # finding if the network output index with the maximum value which is equal to the neural network digit selection
        if int(values[0]) == nump.argmax(out_session):
            test.append(1)
        else:
            test.append(0)
    test = nump.asarray(test)
    print('Net efficiency % =', (test.sum() / test.size) * 100)
```
#### Function to display images of numbers
```python
# Функция отображения изображения числа из набора данных
def plot(pixels: nump.array):
    plt.imshow(pixels.reshape((28, 28)), cmap='gray') # black and white picture
    plt.show()
```
#### Training network and calculating its effectivness
```python
# Расчёт эффективности сети
input_nodes, hidden_nodes, output_nodes, learning_speed = ust_neur()
w_in2hidden, w_hidden2out = create_net(input_nodes, hidden_nodes, output_nodes)
Var = 23 # number of variant
for i in range(5):
    print('Test#', i + 1)
    train_set(w_in2hidden, w_hidden2out, learning_speed)
    test_set(w_in2hidden, w_hidden2out)
data = open(r'C:\Users\Пользователь\Documents\Обучение\Магистратура\1 семестр\Киберфизические системы и технологии'
            r'\ПР2\mnist_test.csv', 'r')
test_list = data.readlines()
data.close()
values = test_list[int(Var - 1)].split(',')
inputs = (nump.asfarray(values[1:]) / 255.0 * 0.999) + 0.001
out_session = net_output(w_in2hidden, w_hidden2out, inputs, 0)
print('Нейросеть выводит значение:', nump.argmax(out_session), 'Реальное значение из файла:', test_list[Var][0])
plot(nump.asfarray(values[1:]))
```

