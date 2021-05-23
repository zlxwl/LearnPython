# -*- coding: utf-8 -*-
# @Time    : 5/23/2021 10:47 PM
# @Author  : Zhong Lei
# @File    : GreenletMultiThread.py
from greenlet import greenlet
import time


def t1():
    while True:
        print("---A---")
        gr2.switch()
        time.sleep(0.3)


def t2():
    while True:
        print("---B---")
        gr1.switch()
        time.sleep(0.3)


if __name__ == '__main__':
    gr1 = greenlet(t1)
    gr2 = greenlet(t2)
    # gr1.switch()

