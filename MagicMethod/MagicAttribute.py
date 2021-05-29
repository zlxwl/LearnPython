# -*- coding: utf-8 -*-
# @Time    : 5/29/2021 12:48 PM
# @Author  : Zhong Lei
# @File    : MagicAttribute.py


class Human(object):
    def __init__(self):
        self.name = "Bob"
        self.age = 18
        self.sex = "male"

    def __getattribute__(self, item):
        print("getattribute is triggered")
        # 不能使用当前对象（self）来访问，会递归调用__getattribute__
        # 使用object
        result = object.__getattribute__(self, "name")
        new_name = result[0] + "*" + result[-1]
        return new_name

    def eat(self):
        print("eating")

    def drink(self):
        print("drinking")


huamn = Human()
print(huamn)
print(huamn.sex)