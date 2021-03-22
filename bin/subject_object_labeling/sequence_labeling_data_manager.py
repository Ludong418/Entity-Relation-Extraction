import os
import json
from bert import tokenization
from config_ import model_config

print("if not have raw data, please dowload data from http://lic2019.ccf.org.cn/kg !")


class Model_data_preparation(object):
    def __init__(self, vocab_file_path="vocab.txt", do_lower_case=True):
        # BERT 自带WordPiece分词工具，对于中文都是分成单字
        self.bert_tokenizer = tokenization.FullTokenizer(vocab_file=self.get_vocab_file_path(vocab_file_path),
                                                         do_lower_case=do_lower_case)
        self.DATA_INPUT_DIR = self.get_data_input_dir()
        self.DATA_OUTPUT_DIR = self.get_data_output_dir()
        print("数据输入路径：", self.DATA_INPUT_DIR)
        print("数据输出路径：", self.DATA_OUTPUT_DIR)

    def get_data_input_dir(self):
        return model_config.RAW_DATA_DIR

    def get_vocab_file_path(self, vocab_file_path):
        return os.path.join(model_config.MODEL_DIR, vocab_file_path)

    def get_data_output_dir(self):
        return model_config.SEQUENCE_DATA_OUTPUT_DIR

    # 序列标注对齐：由原始数据文件生成标注好的序列文件
    def subject_object_labeling(self, spo_list, text):
        def _spo_list_to_spo_predicate_dict(spo_list):
            """
            以关系为 key 分组spo_list
            :param spo_list: [{"predicate": "国籍", "object_type": "国家", "subject_type": "人物",
             "object": "赞比亚", "subject": "Chanda Mushili"}]
            :return:{'国籍':[('Chanda Mushili', '赞比亚')]}
            """
            spo_predicate_dict = dict()
            for spo_item in spo_list:
                predicate = spo_item["predicate"]
                subject = spo_item["subject"]
                object = spo_item["object"]
                spo_predicate_dict.setdefault(predicate, []).append((subject, object))
            return spo_predicate_dict

        # 在列表 k 中确定列表 q 的位置
        def _index_q_list_in_k_list(q_list, k_list):
            """Known q_list in k_list, find index(first time) of q_list in k_list"""
            q_list_length = len(q_list)
            k_list_length = len(k_list)
            for idx in range(k_list_length - q_list_length + 1):
                t = [q == k for q, k in zip(q_list, k_list[idx: idx + q_list_length])]
                # print(idx, t)
                if all(t):
                    # print(idx)
                    idx_start = idx
                    return idx_start

        # 给主体和客体表上BIO分割式类型标签
        def _labeling_type(subject_object, so_type):
            tokener_error_flag = False
            # 对实体分字
            so_tokened = self.bert_tokenizer.tokenize(subject_object)
            so_tokened_length = len(so_tokened)
            # 确定实体的开始位置
            idx_start = _index_q_list_in_k_list(q_list=so_tokened, k_list=text_tokened)
            # 如果没有找实体位置，把实体分字结果和句子分字结果写入错误日志文件当中
            if idx_start is None:
                tokener_error_flag = True
                '''
                实体: "1981年"  原句: "●1981年2月27日，中国人口学会成立"
                so_tokened ['1981', '年']  text_tokened ['●', '##19', '##81', '年', '2', '月', '27', '日', '，', '中', '国', '人', '口', '学', '会', '成', '立']
                so_tokened 无法在 text_tokened 找到！原因是bert_tokenizer.tokenize 分词增添 “##” 所致！
                '''
                self.bert_tokener_error_log_f.write(subject_object + " @@ " + text + "\n")
                self.bert_tokener_error_log_f.write(str(so_tokened) + " @@ " + str(text_tokened) + "\n")
            # 给实体开始处标 B 其它位置标 I
            else:
                labeling_list[idx_start] = "B-" + so_type
                if so_tokened_length == 2:
                    labeling_list[idx_start + 1] = "I-" + so_type
                elif so_tokened_length >= 3:
                    labeling_list[idx_start + 1: idx_start + so_tokened_length] =\
                        ["I-" + so_type] * (so_tokened_length - 1)
            return tokener_error_flag

        text_tokened = self.bert_tokenizer.tokenize(text)
        text_tokened_not_UNK = self.bert_tokenizer.tokenize_not_UNK(text)

        spo_predicate_dict = _spo_list_to_spo_predicate_dict(spo_list)
        for predicate, spo_list_form in spo_predicate_dict.items():
            tokener_error_flag = False
            labeling_list = ["O"] * len(text_tokened)
            for (spo_subject, spo_object) in spo_list_form:
                flag_A = _labeling_type(spo_subject, "SUB")
                flag_B = _labeling_type(spo_object, "OBJ")
                if flag_A or flag_B:
                    tokener_error_flag = True

            # 给被bert_tokenizer.tokenize 拆分的词语打上特殊标签[##WordPiece]
            for idx, token in enumerate(text_tokened):
                """标注被 bert_tokenizer.tokenize 拆分的词语"""
                if token.startswith("##"):
                    labeling_list[idx] = "[##WordPiece]"
            # 如果没有分字出错就写入文件当中
            if not tokener_error_flag:
                self.token_label_and_one_prdicate_out_f.write(" ".join(labeling_list)+"\t"+predicate+"\n")
                self.text_f.write(text + "\n")
                self.token_in_f.write(" ".join(text_tokened)+"\t"+predicate+"\n")
                self.token_in_not_UNK_f.write(" ".join(text_tokened_not_UNK) + "\n")

    # 处理原始数据
    def separate_raw_data_and_token_labeling(self):
        if not os.path.exists(self.DATA_OUTPUT_DIR):
            os.makedirs(os.path.join(self.DATA_OUTPUT_DIR, "train"))
            os.makedirs(os.path.join(self.DATA_OUTPUT_DIR, "valid"))
            os.makedirs(os.path.join(self.DATA_OUTPUT_DIR, "test"))

        for file_set_type in ["train", "valid"]:
            print(os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type)))
            self.token_label_and_one_prdicate_out_f = \
                open(os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type),
                                  "token_label_and_one_prdicate_out.txt"), "w", encoding='utf-8')
            self.bert_tokener_error_log_f = \
                open(os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type),
                                  "bert_tokener_error_log.txt"), "w", encoding='utf-8')

            self.text_f = open(os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type), "text.txt"), "w",
                               encoding='utf-8')
            self.token_in_f = open(os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type), "token_in.txt"),
                                   "w", encoding='utf-8')
            self.token_in_not_UNK_f = open(os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type),
                                                        "token_in_not_UNK.txt"), "w", encoding='utf-8')

            path_to_raw_data_file = None
            if file_set_type == "train":
                path_to_raw_data_file = "train_data.json"
            elif file_set_type == "valid":
                path_to_raw_data_file = "dev_data.json"

            with open(os.path.join(self.DATA_INPUT_DIR, path_to_raw_data_file), 'r', encoding='utf-8') as f:
                count_numbers = 0
                while True:
                    line = f.readline()
                    if line:
                        count_numbers += 1
                        r = json.loads(line)
                        spo_list = r["spo_list"]
                        text = r["text"]
                        self.subject_object_labeling(spo_list=spo_list, text=text)
                    else:
                        break

            print("all numbers", count_numbers)
            self.text_f.close()
            self.token_in_f.close()
            self.token_in_not_UNK_f.close()
            self.token_label_and_one_prdicate_out_f.close()
            self.bert_tokener_error_log_f.close()


if __name__ == "__main__":
    model_data = Model_data_preparation()
    model_data.separate_raw_data_and_token_labeling()

