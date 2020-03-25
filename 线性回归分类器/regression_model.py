import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


class LinearRegression:

    def __init__(self, learn_rate):
        self.learn_rate = learn_rate

    def load_data(self, file_name):
        '''加载数据'''
        data = pd.read_csv(file_name)
        array = np.array(data.values.astype(float))
        self.x = array[:, :-1]
        self.x = self.normalize(self.x)
        self.y = array[:, -1]
        return self.x, self.y

    def normalize(self, data):
        '''进行归一化处理'''
        data = (data - np.mean(data, axis=0)) / \
            (np.max(data, axis=0) - np.min(data, axis=0))
        return data

    def predict(self, alpha, beta, x):
        arr = alpha * x
        return np.sum(arr) + beta

    def gradient_descent_random(self, x_train, y_train, alpha, beta, learn_rate):
        '''随机梯度下降'''
        randomId = int(np.random.random_sample() * x_train.shape[0])
        x = x_train[randomId, :]
        y = y_train[randomId]
        gradient_arr = np.zeros(x.shape[0])
        gradient_beta = 0
        mean_s_err = 0
        err = y - self.predict(alpha, beta, x)
        mean_s_err += err ** 2

        gradient_arr += err * x
        gradient_beta += err

        # arr 是 alpha vector的梯度vec， alpha0 是 arr[0]
        gradient_arr = gradient_arr * 2
        gradient_beta = gradient_beta * 2
        mean_s_err = mean_s_err

        alpha += np.reshape(gradient_arr, alpha.shape) * learn_rate
        beta += gradient_beta * learn_rate

        return alpha, beta, mean_s_err

    def train(self, x_train, y_train, loop_times):
        '''训练模型'''
        # 用随机数据初始化 alpha 和 beta
        alpha = np.random.random_sample(x_train.shape[1])
        beta = np.random.random_sample()
        err_vec = []
        for i in range(loop_times):
            alpha, beta, mean_s_err = self.gradient_descent_random(
                x_train, y_train, alpha, beta, self.learn_rate)
            err_vec.append(mean_s_err)
        return alpha, beta


def main():
    learn_rate = input('Input learn_rate: ')
    model = LinearRegression(learn_rate=float(learn_rate))
    file_path = os.path.join(os.path.dirname(
        __file__), 'winequality-white.csv')
    x, y = model.load_data(file_name=file_path)
    mean_s_err_vec = []
    mean_s_err = 0
    x_loop = [i for i in range(5, 100, 10)]  # 迭代次数
    for loop in x_loop:
        for i in range(x.shape[0]):
            # 留一法
            X_test = x[i, :]
            Y_test = y[i]
            X_train = np.delete(x, i, axis=0)
            Y_train = np.delete(y, i)

            alpha, beta = model.train(
                X_train, Y_train,  loop)
            err = Y_test - model.predict(alpha, beta, X_test)
            mean_s_err += err ** 2
        mean_s_err /= x.shape[0]
        mean_s_err_vec.append(np.sqrt(mean_s_err))
    print('loop:', x_loop)
    print('RMSE:', mean_s_err_vec)
    plt.plot(x_loop, mean_s_err_vec, "-", label="learn_rate="+learn_rate)
    plt.xlabel("loop time")
    plt.ylabel("RMSE")
    plt.title("RMSE of regression model")
    plt.legend(loc="best")
    plt.savefig("fig0.2.png")


if __name__ == "__main__":
    main()
