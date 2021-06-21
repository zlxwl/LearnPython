# -*- coding: utf-8 -*-
# @Time    : 5/26/2021 10:25 PM
# @Author  : Zhong Lei
# @File    : Close.py


# 闭包：内部函数对外部函数作用域里面变量的引用，函数的内的变量有生命周期，
# 在函数执行期间执行。
# 闭包内的闭包函数私有化变量，完成数据封装。类似面向对象作用域
# 装饰器 不影响函数原有功能，还能添加新的功能呢。
def func():
    a = 1 # 外部作用域里面的变量
    print("this is func")

    def func1(num):
        print("this is func1")
        print(num + a)
    return func1


if __name__ == '__main__':
    var = func()
    var(4)


