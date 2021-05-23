# -*- coding: utf-8 -*-
# @Time    : 5/23/2021 10:11 PM
# @Author  : Zhong Lei
# @File    : GeneratorMultiThread.py
import time


def task_1():
    while True:
        print("---1---")
        time.sleep(0.1)
        yield


def task_2():
    while True:
        print("---2---")
        time.sleep(0.1)
        yield


def main():
    t1 = task_1()
    t2 = task_2()
    while True:
        next(t1)
        next(t2)


if __name__ == '__main__':
    main()