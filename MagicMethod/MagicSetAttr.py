# -*- coding: utf-8 -*-
# @Time    : 5/29/2021 1:56 PM
# @Author  : Zhong Lei
# @File    : MagicSetAttr.py


class Man(object):
    def __init__(self):
        self.name = "Bob"
        self.age = 19
        self.sex = "Male"

    def __setattr__(self, key, value):
        if key == "sex":
            pass
        else:
            object.__setattr__(self, key, value)

        print("setattr triggered")

    def eating(self):
        print("eating ...")

    def drinking(self):
        print("drinking ...")

man = Man()
man.name = "Peter"
man.sex = "Female"
print(man.__dict__)
