# -*- coding: utf-8 -*-
# @Time    : 5/26/2021 11:15 PM
# @Author  : Zhong Lei
# @File    : Decorate2.py

def func1(func):
    def func2():
        print("aaa")
        return func()
    return func2


@func1
def myprint():
    print("hello")


def arg_func(sex):
    def func1(func):
        def func2():
            if sex == "man":
                print("a")
            else:
                print("b")
            return func()
        return func2
    return func1


@arg_func(sex="man")
def man():
    print("work")


@arg_func(sex="woman")
def woman():
    print("work")


if __name__ == '__main__':
    man()