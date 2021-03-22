#!/usr/bin/python

# encoding: utf-8

"""
@author: dong.lu

@contact: ludong@cetccity.com

@software: PyCharm

@file: infer.py

@time: 2019/04/9 10:30

@desc: 关系多标签分类模型推理部分

"""
import time
import os

import tensorflow as tf
from tensorflow.core.framework import types_pb2

from load import LoadModelBase
from bert.tokenization import FullTokenizer
from config_ import model_config


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class PredicateInfer(LoadModelBase):
    def __init__(self, vocab_file, export_dir=None, url=None, model_name='models',
                 signature_name=None, do_lower_case=True):
        super(PredicateInfer, self).__init__(export_dir, url, model_name, signature_name)
        # 加载段落处理器
        # self.sen_processor = SentenceProcessor()
        # 加载 bert tokenizer
        self.tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

        # 通过 grpc
        if url:
            self.stub, self.request = self.load_grpc_connect()

        if export_dir:
            self.predict_fn = self.load_pb_model()

        self.id_map_predicate = self.id_to_label(model_config.PREDICATE_LABEL)

    def process(self, sentences, max_seq_length=64):
        if not sentences or not isinstance(sentences, list):
            raise ValueError('`sentences` must be list object and not a empty list !')

        examples = []
        for sentence in sentences:
            feature = self.convert_single_example(sentence, max_seq_length)
            example = self.convert_single_feature(feature)
            examples.append(example)

        return examples

    def convert_single_example(self, sentence, max_seq_length):
        """
        处理单个语句
        sentence: str, 预测句子
        max_seq_length: int，句子最大长度
        :return:
        """
        sentence = self.tokenizer.tokenize(sentence)
        if len(sentence) > max_seq_length - 2:
            sentence = sentence[0:(max_seq_length - 2)]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in sentence:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

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

    def infer(self, sentences, max_seq_length, top_n=3):
        """
        预测调用
        sentences: list，输入一批预测句子
        max_seq_length: int, 输入最大长度
        top_n: int，返回前多少个类别
        :return:
        list，例如 [[('作者', 0.98), ('出生地', 0.02)...], ]
        """
        result = []
        examples = self.process(sentences, max_seq_length)
        if self.url:
            predictions = self.tf_serving_infer(examples)

        else:
            s = time.time()
            predictions = self.local_infer(examples)
            print('predicate:', time.time() - s)

        predictions = predictions['predictions']

        for p in predictions:
            top_n_idx = p.argsort()[:: -1][0: top_n]
            label = list(map(lambda x: (self.id_map_predicate[x], p[x]), top_n_idx))

            result.append(label)

        return result

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

    def id_to_label(self, labels):
        return dict([(i, label) for i, label in enumerate(labels)])


if __name__ == '__main__':
    test = ['南京京九思新能源有限公司于2015年05月15日在南京市江宁区市场监督管理局登记成立',
            '内容简介《宜兴紫砂图典》由故宫出版社出版',
            '《单亲爱》是2008年万卷出版公司出版的一本图书，作者是蔡卓妍']
    infer = PredicateInfer(model_config.VOCAB_PATH,
                           os.path.join(model_config.PREDICATE_PB_MODEL, '1598840746'))
    while True:
        text = input('请输入：')
        print(infer.infer([text], 128, top_n=5))
