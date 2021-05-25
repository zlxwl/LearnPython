# -*- coding: utf-8 -*-
# @Time    : 2021/5/25 下午12:29
# @Author  : Zhong Lei
# @FileName: Property.py


class Foo:
    def get_bar(self):
        return "laowang"

    def set_bar(self, value):
        return value

    def del_bar(self):
        print("deleting")

    Bar = property(get_bar, set_bar, del_bar, "discriptiuon")


class Goods:
    @property
    def price(self):
        print("property")

    @price.setter
    def price(self, value):
        print("@property.setter")

    @price.deleter
    def price(self):
        print("@proper.setter")


if __name__ == '__main__':
    foo = Foo()
    print(foo.Bar)
    print(foo.Bar.__doc__)

    good = Goods()
    good.price
    good.price = 123
    del good.price