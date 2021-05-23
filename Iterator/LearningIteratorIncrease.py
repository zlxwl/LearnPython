# -*- coding: utf-8 -*-
# @Time    : 5/23/2021 7:34 PM
# @Author  : Zhong Lei
# @File    : LearningIteratorIncrease.py


class Classmate(object):
    def __init__(self):
        self.count = 0
        self.names = list()

    def add(self, name):
        self.names.append(name)

    def __iter__(self):
        return self

    def __next__(self):
        if self.count < len(self.names):
            res = self.names[self.count]
            self.count += 1
            return res
        else:
            raise StopIteration


class FibIterator(object):
    def __init__(self, n):
        self.n = n
        self.current = 0
        self.num1 = 0
        self.num2 = 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.n:
            ret = self.num1
            self.num1, self.num2 = self.num2, self.num2 + self.num1
            self.current += 1
            return ret
        else:
            raise StopIteration


if __name__ == '__main__':
    Fibs = FibIterator(10)
    for fib in Fibs:
        print(fib)