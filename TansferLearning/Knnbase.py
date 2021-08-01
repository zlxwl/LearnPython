# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : Knnbase.py
# Time       ：2021/8/1 21:12
# Author     ：Zhong Lei
"""

def load_data(folder, domain):
    from scipy import io
    import os
    data = io.loadmat(os.path.join(folder, domain+"_fc6.mat"))
    return data["fts"], data["labels"]

def knn_classify(Xs, Ys, Xt, Yt, k=1):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    model = KNeighborsClassifier(n_neighbors=k)
    Ys = Ys.ravel()
    Yt = Yt.ravel()
    model.fit(Xs, Ys)
    Yt_pred = model.predict(Xt)
    acc = accuracy_score(Yt, Yt_pred)
    print("Accuracy using kNN:{:.2f}".format(acc * 100))

if __name__ == '__main__':
    folder = "D:\BaiduNetdiskDownload\office31_decaf"
    src_domain = "amazon"
    tar_domain = "webcam"
    Xs, Ys = load_data(folder, src_domain)
    Xt, Yt = load_data(folder, tar_domain)
    print("Source:", src_domain, Xs.shape, Ys.shape)
    print("Target:", tar_domain, Xt.shape, Yt.shape)
    knn_classify(Xs, Ys, Xt, Yt)