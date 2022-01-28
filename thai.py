# -*- coding: utf-8 -*-
# @Time    : 1/28/2022 4:21 PM
# @Author  : Zhong Lei
# @File    : thai.py
from pythainlp.tag.named_entity import ThaiNameTagger
from typing import List, Tuple


def reformat_ner(entities: List[Tuple[str, str, str]]) -> List[Tuple[str, str]]:
    cur_word = ''
    reformat_text_entity = []
    flag = False
    for text, _, entity in entities:
        if str(entity).startswith('O'):
            cur_word = ''
            flag = False
            continue
        elif str(entity).startswith('B'):
            if not cur_word:
                cur_word += text
                flag = True
        elif str(entity).startswith('I'):
            if cur_word:
                cur_word += text
                flag = True
        if cur_word and flag:
            reformat_text_entity.append((cur_word, entity))
    return reformat_text_entity


if __name__ == '__main__':
    ner = ThaiNameTagger()
    text_tag_entity = ner.get_ner(
        "1109/231 หมู่บ้านเดอะฟอร์จูน ถนนประชาพัฒนา แขวงทับยาว, เขตลาดกระบัง, จังหวัดกรุงเทพมหานคร")
    print(text_tag_entity)
    data = reformat_ner(text_tag_entity)
    print(data)
