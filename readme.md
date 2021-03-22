# Entity-Relation-Extraction
本项目fork了2019语言与智能技术竞赛信息抽取（实体与关系抽取）任务解决方案冠军的代码，我在此基础上重新训练了模型（多分类和序列标注模型），并且增加了预测代码以及docker的运行方法。一些介绍可以查看本人写的[博客](https://blog.csdn.net/weixin_40570579/article/details/104900083).

## 环境
python 3.6.3  
tensorflow 1.9.0  
docker  

## 数据
2019语言与智能技术竞赛信息抽取提供的数据，见 *raw_data* 目录下，特别说明数据并不是全部的训练的数据，我没有找到全部的数据，如果有哪位朋友找到，可以联系我，谢谢了。

## 目录结构
bert  
bin  
   |----- predicate_classifiction # 关系多分类预处理代码  
   |----- subject_object_labeling # 序列标注预处理代码  
classfication_data # 分类模型的输入  
           |----- train  
           |----- valid  
           |----- test  
config_  
doc  
output  
    |----- predicate_classification_model # 多分类模型  
    |----- sequence_labeling_model # 序列标注模型  
    |----- predicate.tar # 模型的docker镜像  
pretrained_model  
    |----- chinese_L-12_H-768_A-12  
    |----- roberta_zh_l12  
raw_data  
SKE_2019_tokened_labeling # 序列标注的模型  
           |----- train  
           |----- valid  
           |----- test  
run.py # flask服务  
run_predicate_classification.py # 训练分类模型  
run_sequnce_labeling.py # 训练标注模型  

## 模型下载
#### 关系多分类
链接：https://pan.baidu.com/s/1wol3etcWCJVjEqAYG3KhWA   
提取码：zu7y 

#### 序列标注
链接：https://pan.baidu.com/s/1YziRU289E3LGzOSLmtIR3Q   
提取码：8whl

## 运行
```python
# 关系分类预处理
python predicate_data_manager.py
# 序列标注预处理
python sequence_labeling_data_manager.py
# 关系分类训练
python predicate_classification_infer.py
# 序列标注训练
python run_sequnce_labeling.py
```

## 效果
### 关系多分类模型评估
|  global_step   | eval_token_label_precision  |
|  ----  | ----  |
| 8128  | 0.648 |
### 序列标注模型评估
|  global_step   | eval_token_label_precision  | eval_token_label_recall | eval_token_label_f |
|  ----  | ----  | --- | --- |
| 9000  | 0.920 |   0.947 | 0.933 |

## 参考
[原作者github](https://github.com/yuanxiaosc/Entity-Relation-Extraction)