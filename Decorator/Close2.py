# -*- coding: utf-8 -*-
# @Time    : 5/26/2021 10:37 PM
# @Author  : Zhong Lei
# @File    : Close2.py

mylist = [1, 2, 3, 4, 5]


def func(obj):
    print("func:", obj)
    def func1():
        obj[0] += 1
        print(obj)
    return func1


if __name__ == '__main__':
    f = func(mylist)
    f()