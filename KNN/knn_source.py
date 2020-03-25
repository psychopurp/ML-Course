# -*- coding:utf-8 -*-
import os
import numpy as np


# 获取类标签
def getLabel(hot, size):
    for i in range(size):
        if hot[i] == '1':
            return i
    return -1

# 读训练集和测试集数据，并进行归一化处理


def loadData():
    # 训练集
    train = []
    with open(os.path.dirname(__file__)+'/semeion_train.csv', 'r') as train_file:
        for row in train_file:
            line = row.strip().split(' ')
            train.append(line[:-10]+[getLabel(line[-10:], 10)])
    # 归一化处理
    train = np.array(train, dtype=float)
    m, n = np.shape(train)
    data = train[:, 0:n-1]
    min_data = data.min(0)
    max_data = data.max(0)

    data = (data - min_data) / (max_data - min_data)

    train[:, 0:n-1] = data

    # 测试集
    test = []
    with open(os.path.dirname(__file__)+'/semeion_test.csv', 'r') as test_file:
        for row in test_file:
            line = row.strip().split(' ')
            test.append(line[:-10]+[getLabel(line[-10:], 10)])
    # 归一化处理
    test = np.array(test, dtype=float)
    m, n = np.shape(test)
    data = test[:, 0:n - 1]
    min_data = data.min(0)
    max_data = data.max(0)
    data = (data - min_data) / (max_data - min_data)
    test[:, 0:n - 1] = data
    # print(test)
    return train, test

# 计算欧式距离


def euclidean_Dist(data1, data2):
    det = (data1-data2)[:, 0:n-1]
    return np.linalg.norm(det, axis=1)

# 计算测试集样本的k近邻，标签


def getKNNPredictedLabel(train_data, test_data, train_label, test_label, k):
   # 通过K近邻分类，获取测试集的预测标签
    correct_cnt = 0
    pre_label = []
    i = 0
    for test in test_data:

        dist = euclidean_Dist(train_data, test)
        distSorted = np.argsort(dist)
        classCount = {}
        for num in range(k):
            voteLabel = train_label[distSorted[num]]
            classCount[voteLabel] = classCount.get(voteLabel, 0)+1
        sortedClassCount = sorted(
            classCount.items(), key=lambda x: x[1], reverse=True)
        # print('类标签按其个数排序:{}'.format(sortedClassCount))
        predictedLabel = sortedClassCount[0][0]
        pre_label.append(predictedLabel)
        if (predictedLabel == test_label[i]):
            correct_cnt += 1
        i = i+1
    return correct_cnt, pre_label


if __name__ == '__main__':
    (train_data, test_data) = loadData()
    m, n = np.shape(train_data)
    train_label = train_data[:, -1]
    test_label = test_data[:, -1]
    for k in (1, 3, 5):
        correct_cnt, pre_label = getKNNPredictedLabel(
            train_data, test_data, train_label, test_label, k)
        acc = correct_cnt/np.shape(test_data)[0]
        print('k为', k, '时，分类正确的个数为', correct_cnt,
              ',分类精度为%.2f' % (acc*100), '%')
        #print (pre_label)
