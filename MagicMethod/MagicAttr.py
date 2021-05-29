# -*- coding: utf-8 -*-
# @Time    : 5/29/2021 1:42 PM
# @Author  : Zhong Lei
# @File    : MagicAttr.py


class Man(object):
    def __init__(self):
        self.name = "Bob"
        self.sex = "male"
        self.age = 18

    def __getattr__(self, item):
        if "name" in item:
            return "getattr triggered"

    def eat(self):
        print("eating ...")

    def drinking(self):
        print("drinking ...")


man = Man()
print(man)
print(man.name23)