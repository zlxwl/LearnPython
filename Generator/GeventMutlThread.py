# -*- coding: utf-8 -*-
# @Time    : 5/23/2021 10:56 PM
# @Author  : Zhong Lei
# @File    : GeventMutlThread.py
import time
import gevent
from gevent import monkey

monkey.patch_all()


def f(n, name):
    for i in range(n):
        # gevent.sleep(1)
        time.sleep(0.1)
        print(name, i)


if __name__ == '__main__':
    gevent.joinall([
        gevent.spawn(f, 5, name="work1"),
        gevent.spawn(f, 6, name="work2")
    ])
    # g1 = gevent.spawn(f, 5)
    # g2 = gevent.spawn(f, 5)
    # g3 = gevent.spawn(f, 5)
    # g1.join()
    # g2.join()
    # g3.join()

