# Created by Zhang Siru at 2021/11/30 15:30 with PyCharm
# Feature: 用梯度下降完成逻辑回归
# Python Version: 3.6
# SourceFile: ./data/*.json

import numpy as np
import math
from array import array
import json
import matplotlib.pyplot as plt


def process_data(_file, _flag):
    """
    :param _file: .json数据文件
    :param _flag: _flag=0读取的是trainset和devset；_flag=1 读取的是testset
    :return: vecs_list 读取向量, 例如trainset.json，得到的就是向量, 标签；而testset.json得到的就是向量
    """
    label_list = []
    vecs_list = []
    with open(_file, 'r') as load_f:
        json_data = json.load(load_f)
        for item in json_data:
            # print(item)
            if _flag == 0:  # vectors+labels
                vecs_list.append(item[0])  # vectors
                label_list.append(item[1])  # labels
            elif _flag == 1:
                vecs_list.append(item)

    if _flag == 0:
        return np.array(vecs_list), np.array(label_list)
    else:
        return np.array(vecs_list)


def h_func(_x, _weights):
    """
    Logistic function theta(s) = (e^s)/(1+e^s)
    h(x) = theta(wx) = 1/(1+e**(-wx))
    :param _x:
    :param _weights: 权重矩阵
    :return: 一个浮点数
    """
    return math.e**(np.dot(_x, _weights))/(1.+math.e**(np.dot(_x, _weights)))


def gradient_descent(x, y, weights, learning_rate=0.001, max_iter=1000, debug=True, for_train=False):
    """
    梯度下降逻辑回归，得到权重矩阵
    :param x: vectors_list
    :param y: labels_list
    :param weights: 如果是输入训练集，则在该函数内初始化
    :param learning_rate:
    :param max_iter: 最大迭代次数
    :param debug:
    :param for_train:
    :return: weights, err
    """
    if debug:
        err = array('f', [])

    # 设置初始参数
    m, n = x.shape  # X样本数目，维度
    # print(m, n)

    grad_Wt = 0
    z = np.arange(m)  # 生成一个[0,1, ..., m-1]的数组，主要是可以随机选点
    for t in range(max_iter):
        cost = 0
        np.random.shuffle(z)  # shuffle
        # print(z)
        for i in z:  # compute gradient
            Ai = h_func(_x=-y[i]*x[i], _weights=weights)  # theta function
            Bi = -y[i]*x[i]
            grad_Wt = grad_Wt + Ai*Bi

            cost = cost + np.log(1+math.e**(-y[i]*np.dot(x[i], weights)))
            # if (Ai >= 0.5 and y[i] == 1) or (Ai < 0.5 and y[i] == 0) 分类正确
            # Ai = np.clip(Ai, eps, 1-eps)
            # cost = cost-(y[i]*np.log(Ai) + (1.-y[i])*np.log(1.-Ai))  # cross-entropy error

        weights = weights - learning_rate*(grad_Wt/m)  # update weights
        if debug:
            err.append(cost/m)

    if debug:
        return weights, err

    return weights


def pred_classification(_x, _weights):
    """
    利用得到的权重矩阵预测分类
    :param _x:
    :param _weights:
    :return:
    """
    pred_labels = []
    for i in range(len(_x)):
        pred_label = h_func(_x[i], _weights)  # predict label
        if pred_label >= 0.5:
            pred_labels.append(1)
        else:
            pred_labels.append(0)
    return pred_labels


def get_acc(old_labels, new_labels):
    count = 0
    for i in range(len(old_labels)):
        if old_labels[i] != new_labels[i]:
            count = count + 1
    return 1.-count/len(old_labels)


def draw_graph(_loss1, _loss2, _iter):
    fig, ax = plt.subplots()
    x_axis = np.arange(_iter)  # 横坐标iterations
    y_axis_train = _loss1
    y_axis_dev = _loss2

    x_ticks = [i*100 for i in range(11)]

    ax.set_xlabel('iterations')
    ax.set_ylabel('loss')
    ax.set_xticks(x_ticks)
    ax.set_ylim(min(min(y_axis_train), min(y_axis_dev)), max(max(y_axis_train), max(y_axis_dev)))
    ax.plot(x_axis, y_axis_train, label='trainset loss')
    ax.plot(x_axis, y_axis_dev, label='devset loss')
    ax.legend()
    plt.savefig('./output/loss.jpg')
    plt.show()


def fill_testset(_x, _weight):
    pred_labels = pred_classification(_x=_x, _weights=_weight)

    fill_testset = './data/fill-testset.json'
    testset_fp = open(file=fill_testset, mode='w+')
    fill_list = []
    vecs_list = []
    for i in range(len(_x)):
        vecs_list.append(list(_x[i]))
        vecs_list.append(pred_labels[i])
        fill_list.append(vecs_list)
        vecs_list = []
    json.dump(fill_list, testset_fp)


def test_gd():
    """
    运行逻辑回归模型
    :return:
    """
    train_file = './data/trainset.json'
    dev_file = './data/devset.json'
    test_file = './data/testset.json'

    acc_path = './output/accuracy.txt'
    acc_file = open(acc_path, 'w+')

    train_vectors, train_labels = process_data(_file=train_file, _flag=0)
    dev_vectors, dev_labels = process_data(_file=dev_file, _flag=0)
    test_vectors = process_data(_file=test_file, _flag=1)

    # use trainset to get weights matrix
    lr = 0.001
    max_iter = 1000
    m, n = test_vectors.shape
    _weight = np.random.random(n)
    weight, loss_train = gradient_descent(x=train_vectors,
                                          y=train_labels,
                                          weights=_weight,
                                          learning_rate=lr,
                                          max_iter=max_iter,
                                          for_train=True)
    pred_labels_train = pred_classification(_x=train_vectors, _weights=weight)
    pred_labels_dev = pred_classification(_x=dev_vectors, _weights=weight)

    # requirement 2: 给出train训练集和valid验证集的准确率
    acc_train = get_acc(old_labels=train_labels, new_labels=pred_labels_train)
    acc_dev = get_acc(old_labels=dev_labels, new_labels=pred_labels_dev)
    print("accuracy of trainset: {:.2%}\naccuracy of devset: {:.2%}\n".format(acc_train, acc_dev))
    acc_file.write("max_iteration = {}\n"
                   "learning_rate = {}\n"
                   "accuracy of trainset: {:.2%}\n"
                   "accuracy of devset: {:.2%}\n\n".format(max_iter, lr, acc_train, acc_dev))

    # requirement 3: 根据每次迭代的loss绘制train和valid的loss曲线图，横坐标为iterations，纵坐标为loss
    weight, loss_dev = gradient_descent(x=dev_vectors,
                                        y=dev_labels,
                                        weights=weight,
                                        learning_rate=lr,
                                        max_iter=max_iter)
    draw_graph(_loss1=loss_train, _loss2=loss_dev, _iter=max_iter)

    # 将testset.json补充完整
    fill_testset(_x=test_vectors, _weight=weight)


if __name__ == '__main__':
    test_gd()
