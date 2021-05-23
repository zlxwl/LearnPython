# -*- coding: utf-8 -*-
# @Time    : 5/23/2021 9:28 PM
# @Author  : Zhong Lei
# @File    : Generator.py


def fib(n):
    current = 0
    num1, num2 = 0, 1
    while current < n:
        yield num1
        num1, num2 = num2, num1 + num2
        current += 1
    return "done"


if __name__ == '__main__':
    x = fib(10)
    for number in x:
        print(number)