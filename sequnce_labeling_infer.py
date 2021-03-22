#!/usr/bin/python

# encoding: utf-8

"""
@author: dong.lu

@contact: ludong@cetccity.com

@software: PyCharm

@file: infer.py

@time: 2019/04/9 10:30

@desc: 实体提取推理部分

"""
import time
import os
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.core.framework import types_pb2

from load import LoadModelBase
from bert.tokenization import FullTokenizer
from config_ import model_config


class Entity(object):
    def __init__(self, types):
        self.__begin = None
        self.types = types
        self.__intermediate = []

    @property
    def intermediate(self):
        return self.__intermediate

    @intermediate.setter
    def intermediate(self, intermediate):
        self.__intermediate.append(intermediate)

    @property
    def begin(self):
        return self.__begin

    @begin.setter
    def begin(self, begin):
        self.__begin = begin

    def get_entity_types(self):
        return self.__begin + ''.join(self.__intermediate), self.types


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids,):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class EntityInfer(LoadModelBase):
    def __init__(self, vocab_file, export_dir=None, url=None, model_name='models',
                 signature_name=None, do_lower_case=True):
        super(EntityInfer, self).__init__(export_dir, url, model_name, signature_name)
        self.tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
        # 通过 grpc
        if url:
            self.stub, self.request = self.load_grpc_connect()

        if export_dir:
            self.predict_fn = self.load_pb_model()

        self.id_map_predicate = self.id_to_label(model_config.PREDICATE_LABEL)
        self.predicate_map_id = self.label_to_id(model_config.PREDICATE_LABEL)
        self.id_map_sequence = self.id_to_label(model_config.SEQ_LABEL)

    def id_to_label(self, labels):
        return dict([(i, label) for i, label in enumerate(labels)])

    def label_to_id(self, labels):
        return dict([(label, i) for i, label in enumerate(labels)])

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def process(self, sentences, predicate_labels, max_seq_length=64):
        if not sentences or (not isinstance(sentences, list) and not isinstance(sentences, tuple)):
            raise ValueError('`sentences` must be list object and not a empty list !')

        examples = []
        for sentence, predicate_label in zip(sentences, predicate_labels):
            feature = self.convert_single_example(sentence, predicate_label, max_seq_length)
            example = self.convert_single_feature(feature)
            examples.append(example)

        return examples

    def convert_single_example(self, sentence, predicate_label, max_seq_length):
        tokens = []
        for token in sentence:
            tokens.extend(self.tokenizer.tokenize(token))

        tokens_b = [predicate_label] * len(tokens)
        predicate_label_id = self.predicate_map_id[predicate_label]

        # 把 tokens 和 tokens_b 都截断到相等长度，并且长度的和小于 max_seq_length - 3
        self._truncate_seq_pair(tokens, tokens_b, max_seq_length - 3)

        tokens_a = []
        segment_ids = []
        tokens_a.append("[CLS]")
        segment_ids.append(0)
        for token in tokens:
            tokens_a.append(token)
            segment_ids.append(0)

        tokens_a.append("[SEP]")
        segment_ids.append(0)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens_a)

        # bert_tokenizer.convert_tokens_to_ids(["[SEP]"]) --->[102]
        # 1-100 dict index not used
        bias = 1
        for token in tokens_b:
            # add  bias for different from word dict
            tokens.append(token)
            input_ids.append(predicate_label_id + bias)
            segment_ids.append(1)

        tokens.append('[SEP]')
        # `[SEP]` index 等于 102
        input_ids.append(self.tokenizer.convert_tokens_to_ids(["[SEP]"])[0])
        segment_ids.append(1)

        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            tokens.append("[Padding]")

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids)

        return feature

    def convert_single_feature(self, feature):
        features = dict()
        features['input_ids'] = tf.train.Feature(int64_list=tf.train.Int64List(value=feature.input_ids))
        features['input_mask'] = tf.train.Feature(int64_list=tf.train.Int64List(value=feature.input_mask))
        features['segment_ids'] = tf.train.Feature(int64_list=tf.train.Int64List(value=feature.segment_ids))
        example = tf.train.Example(features=tf.train.Features(feature=features))

        return example.SerializeToString()

    def infer(self, sentences, predicate_labels, max_seq_length, predicate_probabilities=None):
        """
        预测调用
        sentences: list，句子，['xxxx', 'xxxx'...]
        predicate_labels: list, 标签, ['作者', '出生地'...]
        max_seq_length: int
        predicate_probabilities: list, [0.92, 0.01, ...]
        :return:
        list, [
        [{'predicate': predicate, 'subject': subj, 'object': entity}, {'predicate': predicate...],
        [{'predicate': predicate, 'subject': subj, 'object': entity}, {'predicate': predicate...]...
        ]
        """
        examples = self.process(sentences, predicate_labels, max_seq_length)
        if self.url:
            predictions = self.tf_serving_infer(examples)
        else:
            s = time.time()
            predictions = self.local_infer(examples)
            print('sequence:', time.time() - s)

        token_label_predictions = predictions['token_label_predictions']
        predicate_predictions = predictions['predicate_predictions']
        predicate_labels_index = np.argmax(predicate_predictions, -1)

        result = []
        for i in range(len(sentences)):
            token_label = list(map(lambda x: self.id_map_sequence[x], token_label_predictions[i]))
            entities = self.entity_extract(sentences[i], token_label[1: token_label.index('[SEP]')])
            predicate_label_index = predicate_labels_index[i]
            # 关系分类的模型输出 与 序列标注模型输出的结果比较
            if predicate_probabilities:
                predicate_label = max([(predicate_labels[i], predicate_probabilities[i]),
                                       (self.id_map_predicate[predicate_label_index],
                                        predicate_predictions[i][predicate_label_index])],
                                      key=lambda x: x[1])
            else:
                predicate_label = predicate_predictions[i][predicate_label_index]

            triplets = self.organize_triplet(entities, predicate_label[0])
            if triplets:
                result.append(triplets)

        return result

    def organize_triplet(self, entities, predicate):
        """
        把三元组转成字典形式, 可解决一个关系、一个主体（subject）、多个客体（object）
        entities: list, [('xx公司', 'SUB'), ('xx公司', 'OBJ')]
        predicate: str, 关系
        :return:
        list, [{'predicate': predicate, 'subject': subj, 'object': entity},
               {'predicate': predicate, 'subject': subj, 'object': entity}...]
        """
        triplets = []
        subj = None
        for entity, tag in entities:
            if tag == 'SUB':
                subj = entity
                break

        for entity, tag in entities:
            if tag == 'OBJ':
                triplet = {'predicate': predicate, 'subject': subj, 'object': entity}
                triplets.append(triplet)

        return triplets

    def entity_extract(self, sentence, tags):
        """
        依据tags，从sentence抽取实体
        sentence: str,句子
        tags: list, 序列标记，例如 ['O', 'B-SUB', 'I-SUB'...]
        :return:
        list, [('xx公司', 'SUB'), ('xx公司', 'OBJ')]
        """
        entities = []
        sentence_len = len(sentence)
        if sentence_len != len(tags):
            warnings.warn('Token and tags have different lengths.\ndetails:\n{}\n{}'.format(sentence, tags))

        entity = Entity(None)
        t_zip = zip(sentence, tags)

        for i, (token, tag) in enumerate(t_zip):
            if tag == 'O':
                if entity.types:
                    entities.append(entity.get_entity_types())
                    entity = Entity(None)
                continue

            elif tag[0] == 'B':
                if entity.types:
                    entities.append(entity.get_entity_types())
                entity = Entity(tag[2:])
                entity.begin = token

            elif tag[0] == 'I':
                if i == sentence_len - 1:
                    entity.intermediate = token
                    entities.append(entity.get_entity_types())
                    break

                try:
                    entity.intermediate = token
                except Exception as e:
                    print(e)

        return entities

    def tf_serving_infer(self, examples):
        self.request.inputs['examples'].CopyFrom(tf.make_tensor_proto(examples, dtype=types_pb2.DT_STRING))
        response = self.stub.Predict(self.request, 5.0)
        predictions = {}
        for key in response.outputs:
            tensor_proto = response.outputs[key]
            nd_array = tf.contrib.util.make_ndarray(tensor_proto)
            predictions[key] = nd_array

        return predictions

    def local_infer(self, examples):
        """
        本地进行预测，参数解释同上
        """
        predictions = self.predict_fn({'examples': examples})

        return predictions


if __name__ == '__main__':
    data = ['《中国区域卫生规划实施效果评估》是2010年7月1日上海交通大学出版社出版的图书，作者是赵大海',
            '《中国区域卫生规划实施效果评估》是2010年7月1日上海交通大学出版社出版的图书，作者是赵大海']

    labels = ['作者', '国籍']
    probabilities = [0.9961533, 0.9949555]
    infer = EntityInfer(model_config.VOCAB_PATH,
                        os.path.join(model_config.SEQUENCE_PB_MODEL, '1598858751'))
    print(infer.infer(data, labels, 128, predicate_probabilities=probabilities))

    # while True:
    #     text = input('请输入`句子` 和 `关系`，以`@@`分隔：')
    #     t, p = text.split('@@')
    #     print(infer.infer([t], [p], 128))
