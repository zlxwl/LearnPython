# -*- coding: utf-8 -*-
# @Time    : 5/26/2021 9:10 PM
# @Author  : Zhong Lei
# @File    : Magic.py


class Cat:
    """
    说明
    """
    def __init__(self, name):
        self.count = 0
        self.name = name
        self.__private_name = name
        print("I am Cat")
        self.cats = []

    def add(self, name):
        self.cats.append(name)
        return self.cats

    def __del__(self):
        print("析构函数")

    def __call__(self, *args, **kwargs):
        print("cat:", args[0] + args[1])

    def __str__(self):
        return "调用str" + self.name

    def __len__(self):
        return len(self.name)

    def __iter__(self):
        return self

    def __next__(self):
        if self.count < len(self.cats):
            ret = self.cats[self.count]
            self.count += 1
            return ret

        else:
            raise StopIteration

    def __getitem__(self, item):
        if item == self.name:
            return self.name
        else:
            return None

    def __setitem__(self, key, value):
        if key == self.name:
            self.name = value

    # def __delitem__(self, key):
    #     if key == self.name:
    #         def self.name
    def __add__(self, other):
        if isinstance(other, Cat):
            return [self, other]
        if isinstance(other, list):
            other.append(self)
            return other


