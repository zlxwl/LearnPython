# -*- coding: utf-8 -*-
# @Time    : 5/24/2021 10:28 PM
# @Author  : Zhong Lei
# @File    : Property.py


class Goods:
    @property
    def size(self):
        return 100


class Money:
    def __int__(self):
        self.__money = 0

    def getMoney(self):
        return self.__money

    def setMoney(self, value):
        if isinstance(value, int):
            self.__money = value
        else:
            print("error")


## 升级版本使用装饰器装饰
class Money(object):
    def __init__(self):
        self.__money = 0

    @property
    def money(self):
        return self.__money

    @money.setter
    def money(self, value):
        if isinstance(value, int):
            self.__money == value
        else:
            print("error")


if __name__ == '__main__':
    # obj = Goods()
    # ret = obj.size
    # print(ret)
    m = Money()
    print(m.money)
    m.money = 100
    print(m.money)
