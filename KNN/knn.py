import pandas as pd
import numpy as np
import os

'''
K-Nearest Neighbor, KNN
'''


class KNN:

    def __init__(self, k=3):
        self.k = k

    def get_label(self, hot):
        '''获取标签'''
        try:
            index = hot.index('1')
        except Exception as e:
            index = -1
        return index

    def draw(self, num_list):
        print(num_list[-1])
        num_list = [int(i[0]) for i in num_list[:-1]]
        for i, k in enumerate(num_list):
            print(k, end=' ')
            if (i+1) % 16 == 0:
                print()

    def normalize(self, data):
        '''数据转换成numpy array 并进行归一化处理'''
        data = np.array(data, dtype=float)
        result = data[:, :-1]
        # 每列中的最小值
        min_data = result.min(0)
        max_data = result.max(0)
        result = (result - min_data) / (max_data - min_data)
        data[:, :-1] = result
        return data

    def load_data(self, *path):
        '''加载数据'''
        data_list = []
        for p in path:
            with open(p, 'r') as train_file:
                data = []
                for i, row in enumerate(train_file):
                    line = row.strip().split(' ')
                    line = line[:-10]+[self.get_label(line[-10:])]
                    data.append(line)
                    if i >= 200:
                        pass
            data_list.append(data)
        train, test = data_list
        self.train = self.normalize(train)
        self.test = self.normalize(test)
        self.train_label = self.train[:, -1]
        self.test_label = self.test[:, -1]

     # 获取k个近邻的类别标签
    def get_k_neighbor_labels(self, k):
        correct_count = 0
        for index, test_item in enumerate(self.test):
            predict_label = self.predict(test_item, k)
            if predict_label == self.test_label[index]:
                correct_count += 1
        return correct_count

    # 计算一个样本与训练集中所有样本的欧氏距离
    def euclidean_distance(self, one_sample, X_train):
        dist = (X_train - one_sample)[:, :-1]
        # axis=1 时计算行向量二范数
        return np.linalg.norm(dist, axis=1)

    def predict(self, test, k):
        '''
        预测单个label
        :return 该测试集的标签
        '''
        dist = self.euclidean_distance(test, self.train)
        # 从小到大排序
        distSorted = np.argsort(dist)
        # 取前 k 个标签进行投票
        classCount = {}
        for num in range(k):
            label = self.train_label[distSorted[num]]
            # 进行标签统计，得票最多的标签就是该测试样本的预测标签
            classCount[label] = classCount.get(label, 0) + 1

        sortedClassCount = sorted(
            classCount.items(), key=lambda x: x[1], reverse=True)
        # 取投票数最高的为标签
        predict_label = sortedClassCount[0][0]
        return predict_label

    def accuracy(self, correct_count):
        return correct_count/len(self.test_label)


def main():
    knn = KNN()
    train_path = os.path.dirname(__file__) + '/semeion_train.csv'
    test_path = os.path.dirname(__file__)+'/semeion_test.csv'
    knn.load_data(train_path, test_path)
    for k in range(1, 20):
        correct_count = knn.get_k_neighbor_labels(k)
        acc = knn.accuracy(correct_count)*100
        print("k为 {} 时，分类正确个数为 {} ,分类精度为 {}% ".format(k, correct_count, acc))


def test():
    knn = KNN()
    # a1 = np.array([[1, 2, 3], [4, 5, 6]])
    # a3 = np.array([[4, 1, 3], [8, 7, 6]])
    # a2 = np.array([7, 7, 7])
    # result = (a2 - a1)[:, :-1]
    # print(np.linalg.norm(a1, axis=1))
    # print(np.argsort(a3))
    a = {}
    a[0.0] = 1
    print(a.items())


if __name__ == "__main__":
    main()
    # test()
