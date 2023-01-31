import numpy as nump
import matplotlib.pyplot as plt
from scipy.special import expit as fmain


# Функция для установки основных сетевых параметров
def ust_neur():
    input_nodes = 784  # 28x28, размер каждого образца
    print('Input the number of hidden neurons:')
    hidden_nodes = int(input())  # кол-во нейронов в скрытом слое
    output_nodes = 10
    print('Input the training speed(0.5):')
    training_speed = float(input())
    return input_nodes, hidden_nodes, output_nodes, training_speed


# Функция установки начальных значений весов нейронной сети
def create_net(input_nodes, hidden_nodes, output_nodes):
    w_in2hidden = nump.random.uniform(-0.5, 0.5, (hidden_nodes, input_nodes))
    w_hidden2out = nump.random.uniform(-0.5, 0.5, (output_nodes, hidden_nodes))
    return w_in2hidden, w_hidden2out


# Функция вычисления вывода нейронной сети
def net_output(w_in2hidden, w_hidden2out, input_signal, return_hidden):
    inputs = nump.array(input_signal, ndmin = 2).T
    hid_in = nump.dot(w_in2hidden,inputs)
    hid_out = fmain(hid_in)
    fin_in = nump.dot(w_hidden2out,hid_out)
    fin_out = fmain(fin_in)
    if return_hidden == 0:
        return fin_out
    else:
        return fin_out, hid_out


# Функция обучения нейронной сети
def net_train(targets, input_signal, w_in2hidden, w_hidden2out, learning_speed):
    target = nump.array(targets, ndmin = 2).T
    inputs = nump.array(input_signal, ndmin = 2).T
    fin_out, hid_out = net_output(w_in2hidden, w_hidden2out, input_signal, 1)
    out_errors = target - fin_out
    hid_errors = nump.dot(w_hidden2out.T, out_errors)

    w_hidden2out += learning_speed * nump.dot((out_errors * fin_out * (1 - fin_out)), hid_out.T)
    w_in2hidden += learning_speed * nump.dot((hid_errors * hid_out * (1 - hid_out)),inputs.T)
    return w_in2hidden, w_hidden2out


# Функция обучения сети на тренировочных данных
def train_set(w_in2hidden, w_hidden2out, learning_speed):
    data = open(r'C:\Users\Пользователь\Documents\Обучение\Магистратура\1 семестр\Киберфизические системы и технологии'
                r'\ПР2\mnist_train.csv', 'r')
    training_list = data.readlines()
    data.close()
    for record in training_list:
        values = record.split(',')
        inputs = (nump.asfarray(values[1:]) / 255.0 * 0.999) + 0.001
        target = nump.zeros(10) + 0.001
        target[int(values[0])] = 1.0
        net_train(target,inputs,w_in2hidden,w_hidden2out,learning_speed)
    return w_in2hidden, w_hidden2out


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
        if int(values[0]) == nump.argmax(out_session):
            test.append(1)
        else:
            test.append(0)
    test = nump.asarray(test)
    print('Net efficiency % =', (test.sum() / test.size) * 100)


# Функция отображения изображения числа из набора данных
def plot(pixels: nump.array):
    plt.imshow(pixels.reshape((28, 28)), cmap='gray')
    plt.show()


# Расчёт эффективности сети
input_nodes, hidden_nodes, output_nodes, learning_speed = ust_neur()
w_in2hidden, w_hidden2out = create_net(input_nodes, hidden_nodes, output_nodes)
Var = 23
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

