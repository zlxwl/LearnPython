# -*- coding: utf-8 -*-
# @Time    : 5/29/2021 10:46 AM
# @Author  : Zhong Lei
# @File    : MagicPractice.py
from collections.abc import Iterator
from collections.abc import Iterable

from MagicMethod.Magic import Cat

# __init__()
cat = Cat("cat")
# __del__()
# del cat
# __doc__, __class__, __module__
print(cat.__doc__)
print(cat.__class__)
print(cat.__module__)


# __call__() 让实例对象具备调用的能力。
def func(a, b):
    print(a + b)


func(1, 2)

cat(2, 3)
print(callable(cat))

# __dict__
print(cat.__dict__)

# __str__()
print(cat.__str__)
print(str(cat))

# __len__()
print(len(cat))


# iterable __iter__() 迭代器，实现该__iter__()是一个可迭代对象，
# 实现__next__() 方法后可以使用迭代器调用。
print(isinstance(cat, Iterator))
print(isinstance(cat, Iterable))
cat1 = Cat("small")
cat1.add("big")
cat1.add("huge")
cat1.add("large")
for a in cat1:
    print(a)

# __getitem__() 实现key-value取数据
# __setitem__() 实现key-value修改数据
print(cat["cat"])
cat["cat"] = "b"
print(cat["cat"])

# 支持算数操作符 __add__(), __sub__(), __mul__(), __div__(), __pow__(), __mod__()
cat1 = Cat("fisrt")
cat2 = Cat("second")
print(cat1+cat2)
cats = cat1 + cat2
cat3 = Cat("zero")
print(cat3 + cats)
