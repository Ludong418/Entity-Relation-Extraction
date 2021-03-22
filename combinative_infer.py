#!/usr/bin/python

# encoding: utf-8

"""
@author: dong.lu

@contact: ludong@cetccity.com

@software: PyCharm

@file: infer.py

@time: 2019/04/9 10:30

@desc: 关系抽取

"""
import os
from itertools import chain

from config_ import model_config
from predicate_classification_infer import PredicateInfer
from sequnce_labeling_infer import EntityInfer


class Infer(object):
    def __init__(self):
        self.predicate_infer = PredicateInfer(model_config.VOCAB_PATH,
                                              os.path.join(model_config.PREDICATE_PB_MODEL, '1598840746'))
        self.entity_infer = EntityInfer(model_config.VOCAB_PATH,
                                        os.path.join(model_config.SEQUENCE_PB_MODEL, '1598858751'))

    def infer(self, sentences):
        """
        预测调用代码
        :arg
        sentences: list， 句子， ['xxx', 'xxx']
        :return:
        [
        [{{'predicate': predicate, 'subject': subj, 'object': entity}, {'predicate': predicate...],
         [{'predicate': predicate, 'subject': subj, 'object': entity}, {'predicate': predicate...],
         [{'predicate': predicate, 'subject': subj, 'object': entity}, {'predicate': predicate...]
         ...
         ]
        """
        result = []
        sentence_predicate_data = self.sentence_predicate(sentences)
        for data in sentence_predicate_data:
            if data:
                sentences_i, predicate_labels_i = zip(*data)
                predicate_i, probabilities_i = zip(*predicate_labels_i)
                triplets = self.entity_infer.infer(sentences_i, predicate_i, 128,
                                                   predicate_probabilities=probabilities_i)
                triplets = list(chain.from_iterable(triplets))
                result.append(triplets)
            else:
                result.append(data)

        return result

    def sentence_predicate(self, sentences):
        """
        预测句子关系，并生成进入实体提取的数据集
        sentences: list， 句子， ['xxx', 'xxx']
        :return:
        [[('句子', '关系'), ('句子', '关系')],
        [('句子', '关系'), ('句子', '关系')],
        [('句子', '关系'), ('句子', '关系')]]
        """
        sentence_predicate_data = []
        predicates = self.predicate_infer.infer(sentences, 128, top_n=5)
        for i, sentence in enumerate(sentences):
            predicates_i = list(filter(lambda x: x[1] >= 0.5, predicates[i]))
            sentence_predicate_i = list(map(lambda x: (sentence, x), predicates_i))
            sentence_predicate_data.append(sentence_predicate_i)

        return sentence_predicate_data


if __name__ == '__main__':
    texts = ['南京京九思新能源有限公司于2015年05月15日在南京市江宁区市场监督管理局登记成立',
             '内容简介《宜兴紫砂图典》由故宫出版社出版',
             '《单亲爱》是2008年万卷出版公司出版的一本图书，作者是蔡卓妍']
    infer = Infer()
    print(infer.infer(texts))
