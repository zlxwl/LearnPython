# -*- coding: utf-8 -*-
# @Time    : 2021/8/2 下午4:20
# @Author  : Zhong Lei
# @FileName: KernalKmm.py
import numpy as np
import sklearn.metrics
from cvxopt import matrix, solvers
from TansferLearning.Knnbase import load_data, knn_classify


def kernel(ker, X1, X2, gamma):
    K = None
    if ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1), np.asarray(X2))
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1))
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1), np.asarray(X2), gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1), None, gamma)
    return K


class KMM:
    def __init__(self, kernel_type="linear", gamma=1.0, B=1.0, eps=None):
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.B = B
        self.eps = eps

    def fit(self, Xs, Xt):
        ns = Xs.shape[0]
        nt = Xt.shape[0]
        if self.eps == None:
            self.eps = self.B / np.sqrt(ns)
        K = kernel(self.kernel_type, Xs, None, self.gamma)
        kappa = np.sum(kernel(self.kernel_type, Xs, Xt, self.gamma) * float(ns) / float(nt), axis=1)

        K = matrix(np.float64(K))
        kappa = matrix(np.float64(kappa))
        G = matrix(np.r_[np.ones((1, ns)), -np.ones((1, ns)), np.eye(ns), -np.eye(ns)])
        h = matrix(np.r_[ns * (1 + self.eps), ns * (self.eps - 1), self.B * np.ones((ns,)), np.zeros((ns,))])

        sol = solvers.qp(K, -kappa, G, h)
        beta = np.array(sol['x'])
        return beta


if __name__ == '__main__':
    # Xs = [[1, 2, 3], [4, 7, 4], [3, 3, 3], [4, 4, 4], [5, 5, 5], [3, 4, 5], [1, 2, 3], [4, 7, 4], [3, 3, 3], [4, 4, 4],
    #       [5, 5, 5], [3, 4, 5], [1, 2, 3], [4, 7, 4], [3, 3, 3], [4, 4, 4], [5, 5, 5], [3, 4, 5], [1, 2, 3], [4, 7, 4],
    #       [3, 3, 3], [4, 4, 4], [5, 5, 5], [3, 4, 5]]
    # Xt = [[5, 9, 10], [4, 5, 6], [10, 20, 30], [1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9], [100, 100, 100],
    #       [11, 22, 33], [12, 11, 5], [5, 9, 10], [4, 5, 6], [10, 20, 30], [1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9],
    #       [100, 100, 100], [11, 22, 33], [12, 11, 5]]
    # Xs, Xt = np.asarray(Xs), np.asarray(Xt)
    # print(Xs.shape, Xt.shape)
    # kmm = KMM(kernel_type='rbf', B=10)
    # beta = kmm.fit(Xs, Xt)
    # print(beta)
    # print(beta.shape)

    folder = "/home/zhonglei/LearnPython/TansferLearning/office31_decaf"
    src_domain = "amazon"
    tar_domain = "webcam"
    Xs, Ys = load_data(folder, src_domain)
    Xt, Yt = load_data(folder, tar_domain)
    print("Source:", src_domain, Xs.shape, Ys.shape)
    print("Target:", tar_domain, Xt.shape, Yt.shape)
    kmm = KMM(kernel_type="rbf", B=10)
    beta = kmm.fit(Xs, Xt)
    Xs_hat = beta * Xs
    knn_classify(Xs_hat, Ys, Xt, Yt)