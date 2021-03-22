# encoding: utf-8
"""
@author: dong.lu

@contact: ludong@cetccity.com

@software: PyCharm

@file: predicate_data_manager.py

@time: 2020/03/4 10:30

@desc: 原始数据处理程序
"""
import os
import json
from bert import tokenization
from config_ import model_config

print("if not have raw data, please dowload data from http://lic2019.ccf.org.cn/kg !")


def unzip_and_move_files():
    """解压原始文件并且放入 raw_data 文件夹下面"""
    os.system("unzip dev_data.json.zip")
    os.system("mv dev_data.json raw_data/dev_data.json")
    os.system("unzip train_data.json.zip")
    os.system("mv train_data.json raw_data/train_data.json")


class Model_data_preparation(object):
    def __init__(self, vocab_file_path="vocab.txt", do_lower_case=True, competition_mode=False, valid_model=False):
        """
        :param vocab_file_path: 词表路径，一般是预先训练的模型的词表路径
        :param do_lower_case: 默认TRUE
        :param competition_mode: 非比赛模式下，会把验证valid数据作为测试test数据生成
        :param valid_model: 验证模式下，仅仅会生成test测试数据
        """
        # BERT 自带WordPiece分词工具，对于中文都是分成单字
        self.bert_tokenizer = tokenization.FullTokenizer(vocab_file=self.get_vocab_file_path(vocab_file_path),
                                                         do_lower_case=do_lower_case)
        self.DATA_INPUT_DIR = self.get_data_input_dir()
        self.DATA_OUTPUT_DIR = self.get_data_output_dir()
        self.competition_mode = competition_mode
        self.valid_model = valid_model
        print("数据输入路径：", self.DATA_INPUT_DIR)
        print("数据输出路径：", self.DATA_OUTPUT_DIR)
        print("是否是比赛模式（非比赛模式下，会把验证valid数据作为测试test数据生成）：", self.competition_mode)
        print("是否是验证模式（验证模式下，仅仅会生成test测试数据）：", self.valid_model)

    # 获取输入文件路径
    def get_data_input_dir(self):
        return model_config.RAW_DATA_DIR

    def get_data_output_dir(self):
        return model_config.PREDICATE_DATA_OUTPUT_DIR

    # 获取词汇表路径
    def get_vocab_file_path(self, vocab_file_path):
        return os.path.join(model_config.MODEL_DIR, vocab_file_path)

    # 处理原始数据
    def separate_raw_data_and_token_labeling(self):
        if not os.path.exists(self.DATA_OUTPUT_DIR):
            os.makedirs(os.path.join(self.DATA_OUTPUT_DIR, "train"))
            os.makedirs(os.path.join(self.DATA_OUTPUT_DIR, "valid"))
            os.makedirs(os.path.join(self.DATA_OUTPUT_DIR, "test"))

        file_set_type_list = ["train", "valid", "test"]
        if self.valid_model:
            file_set_type_list = ["test"]
        for file_set_type in file_set_type_list:
            print("produce data will store in: ", os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type)))
            # 在比赛模式下，会在 classfication_data\train 和 classfication_data\valid 生成 predicate_out.txt 文件
            # 把原始数据中的关系写入，一行一个或多个关系
            if file_set_type in ["train", "valid"] or not self.competition_mode:
                predicate_out_f = open(
                    os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type), "predicate_out.txt"), "w",
                    encoding='utf-8')
            # 存放原始文本
            text_f = open(os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type), "text.txt"), "w",
                          encoding='utf-8')
            # 存放用bert tokener 分字后的文本文件，包含[UNK]
            token_in_f = open(os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type), "token_in.txt"), "w",
                              encoding='utf-8')
            # 存放用bert tokener 分字后的文本文件，不包含[UNK]
            token_in_not_UNK_f = open(
                os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type), "token_in_not_UNK.txt"), "w",
                encoding='utf-8')

            def predicate_to_predicate_file(spo_list):
                predicate_list = [spo['predicate'] for spo in spo_list]
                predicate_list_str = " ".join(predicate_list)
                predicate_out_f.write(predicate_list_str + "\n")

            if file_set_type == "train":
                path_to_raw_data_file = "train_data.json"
            elif file_set_type == "valid":
                path_to_raw_data_file = "dev_data.json"
            else:
                if self.competition_mode is True:
                    path_to_raw_data_file = "test1_data_postag.json"
                else:
                    path_to_raw_data_file = "dev_data.json"

            # 读取原始数据中的json文件
            with open(os.path.join(self.DATA_INPUT_DIR, path_to_raw_data_file), 'r', encoding='utf-8') as f:
                count_numbers = 0
                while True:
                    line = f.readline()
                    if line:
                        count_numbers += 1
                        r = json.loads(line)
                        if (not self.competition_mode) or file_set_type in ["train", "valid"]:
                            spo_list = r["spo_list"]
                        else:
                            spo_list = []
                        text = r["text"]
                        text_tokened = self.bert_tokenizer.tokenize(text)
                        text_tokened_not_UNK = self.bert_tokenizer.tokenize_not_UNK(text)

                        if (not self.competition_mode) or file_set_type in ["train", "valid"]:
                            predicate_to_predicate_file(spo_list)
                        text_f.write(text + "\n")
                        token_in_f.write(" ".join(text_tokened) + "\n")
                        token_in_not_UNK_f.write(" ".join(text_tokened_not_UNK) + "\n")
                    else:
                        break
            print("all numbers", count_numbers)
            print("\n")
            text_f.close()
            token_in_f.close()
            token_in_not_UNK_f.close()


if __name__ == "__main__":
    Competition_Mode = True
    Valid_Mode = False
    model_data = Model_data_preparation(competition_mode=Competition_Mode, valid_model=Valid_Mode)
    model_data.separate_raw_data_and_token_labeling()
