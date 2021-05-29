# -*- coding: utf-8 -*-
# @Time    : 5/29/2021 4:42 PM
# @Author  : Zhong Lei
# @File    : Meta.py
from collections import UserDict, UserList
from collections import abc


Undefined = object()


class JsonObject(UserDict):
    def __missing__(self, key):
        return Undefined

    def __getattr__(self, item):
        ret = self[item]
        return jsify(ret)


class JsonArray(UserList):
    def __missing__(self, key):
        return Undefined

    def __getattr__(self, item):
        try:
            ret = super().__getitem__(item)
            return jsify(ret)
        except IndexError:
            return Undefined


def jsify(value):
    if isinstance(value, abc.Mapping):
        return JsonObject()
    elif isinstance(value, (list, tuple)):
        return JsonArray()
    else:
        return value


def test_json():
    user_dict = jsify({
        "name": "Bob",
        "age": 18,
        "profile": {
            "sex": "male",
            "weight": "60kg"
        },
        "skills": ["python", "java"]
    })

    assert user_dict["age"] == 18
    assert user_dict["name"] == "Bob"
    assert user_dict.sex == Undefined
    # assert user_dict.skills[3] == Undefined
    assert user_dict.profile.sex == Undefined

#
# def test_array():
#     user_array = JsonArray([{"name": "Bob"}, {"name", "Peter"}])
#     assert user_array[5] == Undefined
#     assert user_array[0].name == "Bob"