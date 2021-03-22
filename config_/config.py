#!/usr/bin/python

# encoding: utf-8

"""
@author: ludong

@contact: liulijun@cetccity.com

@software: PyCharm

@file: base.py

@time: 2019/09/26 10:30

@desc: 配置文件
"""
import os
import yaml
import json
yaml.warnings({'YAMLLoadWarning': False})


sep = os.sep


class YamlConfig(object):
    def __init__(self, yml_file_path='config.yml'):
        # config_ 文件夹路径
        self.CUR_PATH = os.path.abspath(os.path.dirname(__file__))
        # 项目路径
        self.PROJECT_PATH = os.path.abspath(os.path.join(self.CUR_PATH, ".."))
        self._yml_file = os.path.join(self.CUR_PATH, yml_file_path)
        with open(self._yml_file, 'r', encoding='utf-8') as f:
            self._yml_obj = yaml.load(f)
        self._label_json = os.path.join(self.CUR_PATH, 'label.json')
        with open(self._label_json, 'r', encoding='utf-8') as js:
            self._label_obj = json.load(js)


class ModelConfig(YamlConfig):
    def __init__(self):
        super(ModelConfig, self).__init__()
        # 预训练模型文件目录地址
        self.MODEL_DIR = os.path.join(os.path.join(self.PROJECT_PATH, 'pretrained_model'),
                                      self._yml_obj['MODEL']['MODEL_DIR'])
        # 原始数据文件夹地址
        self.RAW_DATA_DIR = os.path.join(self.PROJECT_PATH, self._yml_obj['DATA']['RAW_DATA_DIR'])
        # 训练数据输出文件夹
        self.PREDICATE_DATA_OUTPUT_DIR = os.path.join(self.PROJECT_PATH,
                                                      self._yml_obj['DATA']['PREDICATE_DATA_OUTPUT_DIR'])
        self.SEQUENCE_DATA_OUTPUT_DIR = os.path.join(self.PROJECT_PATH,
                                                     self._yml_obj['DATA']['SEQUENCE_DATA_OUTPUT_DIR'])
        # 关系识别模型保存地址, pb模型和mate模型
        self.PREDICATE_CHECKPOINT = os.path.join(self.PROJECT_PATH,
                                                 'output' + sep + 'predicate_classification_model' + sep + 'checkpoint')
        self.PREDICATE_PB_MODEL = os.path.join(self.PROJECT_PATH,
                                               'output' + sep + 'predicate_classification_model' + sep + 'pb_model')
        self.SEQUENCE_CHECKPOINT = os.path.join(self.PROJECT_PATH,
                                                'output' + sep + 'sequence_labeling_model' + sep + 'checkpoint')
        self.SEQUENCE_PB_MODEL = os.path.join(self.PROJECT_PATH,
                                              'output' + sep + 'sequence_labeling_model' + sep + 'pb_model')
        # 关系标签
        self.PREDICATE_LABEL = self._label_obj['predicate']
        # 序列标注标签
        self.SEQ_LABEL = self._label_obj['seq']

        # bert 字典
        self.VOCAB_PATH = os.path.join(self.MODEL_DIR, 'vocab.txt')


if __name__ == '__main__':
    y = ModelConfig()
    print(y.VOCAB_PATH)
