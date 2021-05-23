# -*- coding: utf-8 -*-
# @Time    : 5/23/2021 12:34 PM
# @Author  : Zhong Lei
# @File    : LearnIterator.py
import time
from collections import Iterable
from collections import Iterator


class Classmate(object):
    def __init__(self):
        self.names = list()

    def add(self, name):
        self.names.append(name)

    def __iter__(self):
        return ClassIterator(self)


class ClassIterator(object):
    def __init__(self, obj):
        self.obj = obj
        self.count = 0

    def __iter__(self):
        pass

    def __next__(self):
        if self.count < len(self.obj.names):
            res = self.obj.names[self.count]
            self.count += 1
            return res
        else:
            raise StopIteration


if __name__ == '__main__':
    classmate = Classmate()
    classmate.add("张三")
    classmate.add("李四")
    classmate.add("王五")
    print(isinstance(classmate, Iterable))
    class_iterator = iter(classmate)
    print(isinstance(class_iterator, Iterator))
    print(next(class_iterator))
    for next in classmate:
        print(next)
