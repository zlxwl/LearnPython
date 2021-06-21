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

# 最外层传递的是装饰器参数，第二次wapper传递的是func
# 最后一层传递进行装饰器实际的运行逻辑并执行被装饰函数的及逻辑func()
# 实际调用过程:args_func(sex)(func)()

# 最外层传递装饰器参数，第二层传递函数，第三层传递数据。
def arg_func(sex):
    def wapper(func):
        def func2(*args):
            if sex == "man":
                print("a")
            else:
                print("b")
            return func()
        return func2
    return wapper


@arg_func(sex="man")
def man():
    print("work")


@arg_func(sex="woman")
def woman():
    print("work")


if __name__ == '__main__':
    woman()