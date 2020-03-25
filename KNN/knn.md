# 第一次作业

- 姓名：艾力亚尔·阿布里米提
- 学号：1711306
- 专业：计算机科学与技术

## 1.问题描述

- 使用 KNN 分类器进行手写数字识别
- 计算 KNN 分类器参数 k=1，3，5 时的分类精度

## 2.解决思路

- 将测试数据跟已知的训练数据进行比较，来判断测试值更像哪个数
- 相同类别的样本之间在特征空间中应当聚集在一起
- 使用 KNN 算法

## 3.算法理论

- 概述: KNN，全称 k-NearestNeighbor，即常说的 k 邻近算法。
- 核心思想：一个样本 x 与样本集中的 k 个最相邻的样本中的大多数属于某一个类别 yLabel，那么该样本 x 也属于类别 yLabel，并具有这个类别样本的特性。简而言之，一个样本与数据集中的 k 个最相邻样本中的大多数的类别相同。
- 应用：由其思想可以看出，KNN 是通过测量不同特征值之间的距离进行分类，而且在决策样本类别时，只参考样本周围 k 个“邻居”样本的所属类别。因此比较适合处理样本集存在较多重叠的场景，主要用于聚类分析、预测分析、文本分类、降维等，也常被认为是简单数据挖掘算法的分类技术之一。
- 优点
  - 简单，易于理解，易于实现，无需估计参数，无需训练
  - 可以用于多分类分类、回归等
- 缺点
  - 该算法在分类时有个主要的不足是，当样本不平衡时，如一个类的样本容量很大，而其他类样本容量很小时，有可能导致当输入一个新样本时，该样本的 K 个邻居中大容量类的样本占多数。 该算法只计算“最近的”邻居样本，某一类的样本数量很大，那么或者这类样本并不接近目标样本，或者这类样本很靠近目标样本。无论怎样，数量并不能影响运行结果。可以采用权值的方法（和该样本距离小的邻居权值大）来改进；
  - 该方法的另一个不足之处是计算量较大，因为对每一个待分类的文本都要计算它到全体已知样本的距离，才能求得它的 K 个最近邻点。

## 4.算法流程

- 收集数据，并将数据划分为训练集和测试集
- 将数据集进行归一化处理
- 计算测试数据与各个训练数据之间的距离
- 对距离从小到大进行排序
- 选取距离最小的 k 个点
- 确定前 k 点类别的出现频率
- 出现频率最高的类别作为预测分类

## 5.实验数据

- semeion_test.csv 为测试集
- semeion_train.csv 为训练集

## 6.实验设计

- ##### 数据读取

  ```
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
  ```

- #### 将数据集进行归一化处理
  ```
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
  ```
- #### 计算测试数据与各个训练数据之间的距离

  ```
    # 计算一个样本与训练集中所有样本的欧氏距离
    def euclidean_distance(self, one_sample, X_train):
        dist = (X_train - one_sample)[:, :-1]
        # axis=1 时计算行向量二范数
        return np.linalg.norm(dist, axis=1)
  ```

- #### 对距离从小到大进行排序并选取距离最小的 k 个点

  ```
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
  ```

- #### 准确率计算
  ```
  def accuracy(self, correct_count):
      return correct_count/len(self.test_label)
  ```

## 7.实验结果

- #### 运行

  ```
  def main():
    knn = KNN()
    train_path = os.path.dirname(__file__) + '/semeion_train.csv'
    test_path = os.path.dirname(__file__)+'/semeion_test.csv'
    knn.load_data(train_path, test_path)
    for k in range(1, 20):
        correct_count = knn.get_k_neighbor_labels(k)
        acc = knn.accuracy(correct_count)*100
        print("k为 {} 时，分类正确个数为 {} ,分类精度为 {}% ".format(k, correct_count, acc))
  ```

- #### 实验结果
  ```
  root@LAPTOP-QCG6IJQS:Python_projects/机器学习 # python3 -u "/mnt/f/WorkSpace/Python_projects/机器学习/KNN/knn.py"
  k为 1 时，分类正确个数为 409 ,分类精度为 85.56485355648536%
  k为 2 时，分类正确个数为 409 ,分类精度为 85.56485355648536%
  k为 3 时，分类正确个数为 409 ,分类精度为 85.56485355648536%
  k为 4 时，分类正确个数为 414 ,分类精度为 86.61087866108787%
  k为 5 时，分类正确个数为 410 ,分类精度为 85.77405857740585%
  k为 6 时，分类正确个数为 413 ,分类精度为 86.40167364016736%
  k为 7 时，分类正确个数为 407 ,分类精度为 85.14644351464436%
  k为 8 时，分类正确个数为 404 ,分类精度为 84.51882845188284%
  k为 9 时，分类正确个数为 411 ,分类精度为 85.98326359832636%
  k为 10 时，分类正确个数为 406 ,分类精度为 84.93723849372385%
  k为 11 时，分类正确个数为 404 ,分类精度为 84.51882845188284%
  k为 12 时，分类正确个数为 404 ,分类精度为 84.51882845188284%
  k为 13 时，分类正确个数为 398 ,分类精度为 83.26359832635984%
  k为 14 时，分类正确个数为 394 ,分类精度为 82.42677824267783%
  k为 15 时，分类正确个数为 393 ,分类精度为 82.21757322175732%
  k为 16 时，分类正确个数为 392 ,分类精度为 82.00836820083683%
  k为 17 时，分类正确个数为 392 ,分类精度为 82.00836820083683%
  k为 18 时，分类正确个数为 390 ,分类精度为 81.58995815899581%
  k为 19 时，分类正确个数为 389 ,分类精度为 81.3807531380753%
  ```
- #### 实验结果分析
  - 由结果我们可以知道不一定 K 越大，准确率就越高
  - 当 K=4 可以取得当前数据集里的最优解
  - 该算法比较适用于样本容量比较大的类域的自动分类，而那些样本容量较小的类域采用这种算法比较容易产生误分。

## 代码

```
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

```
