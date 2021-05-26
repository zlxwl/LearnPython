# -*- coding: utf-8 -*-
# @Time    : 5/24/2021 9:47 PM
# @Author  : Zhong Lei
# @File    : Decorator.py
import time


def display_time(func):
    def wrapper(*args):
        t1 = time.time()
        result = func(*args)
        t2 = time.time()
        print("total timeL {:.4} s".format(t2-t1))
        return result
        # 被装饰函数如果有return则需要返回被装饰函数，
        # 如果没有return可以不返回func
    return wrapper


def is_prime(num):
    if num < 2:
        return False
    elif num == 2:
        return True
    else:
        for i in range(2, num):
            if num % 2 == 0:
                return True


# @display_time
# # def prime_nums():
# #     for i in range(2, 10000):
# #         if is_prime(i):
# #             print(i)
@display_time
def count_prime_nums(number):
    count = 0
    for i in range(2, number):
        if is_prime(i):
            print(i)
            count += 1
    return count


if __name__ == '__main__':
    # prime_nums()
    print(count_prime_nums(10000))
