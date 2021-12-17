# Chinese named entity recognization (bert/roberta/macbert/bert_wwm with Keras)

## Project Structure

```
./
├── DataProcess
│   ├── __pycache__
│   ├── convert2bio.py
│   ├── convert_jsonl.py
│   ├── handle_numbers.py
│   ├── load_data.py
│   └── statistic.py
├── README.md
├── __pycache__
├── chinese_L-12_H-768_A-12                                    BERT权重
│   ├── bert_config.json
│   ├── bert_model.ckpt.data-00000-of-00001
│   ├── bert_model.ckpt.index
│   ├── bert_model.ckpt.meta
│   └── vocab.txt
├── chinese_bert_wwm                                           BERT_wwm权重
│   ├── bert_config.json
│   ├── bert_model.ckpt.data-00000-of-00001
│   ├── bert_model.ckpt.index
│   ├── bert_model.ckpt.meta
│   └── vocab.txt
├── chinese_macbert_base                                       macBERT权重
│   ├── chinese_macbert_base.ckpt.data-00000-of-00001
│   ├── chinese_macbert_base.ckpt.index
│   ├── chinese_macbert_base.ckpt.meta
│   ├── macbert_base_config.json
│   └── vocab.txt
├── chinese_roberta_wwm_ext_L-12_H-768_A-12                    roberta权重
│   ├── bert_config.json
│   ├── bert_model.ckpt.data-00000-of-00001
│   ├── bert_model.ckpt.index
│   ├── bert_model.ckpt.meta
│   └── vocab.txt
├── config                                                     
│   ├── __pycache__
│   ├── config.py                                              配置文件
│   └── pulmonary_label2id.json                                label id
├── data                                                       数据集
│   ├── pulmonary.test
│   ├── pulmonary.train
│   └── sict_train.txt
├── environment.yaml                                           conda环境配置文件
├── evaluate.py
├── generator_train.py
├── keras_bert                                                 keras_bert（可pip下）
├── keras_contrib                                              keras_contrib（可pip下）
├── log                                                        训练nohup日志
│   ├── chinese_L-12_H-768_A-12.out
│   ├── chinese_macbert_base.out
│   ├── chinese_roberta_wwm_ext_L-12_H-768_A-12.out
│   └── electra_180g_base.out
├── model.py                                                   模型构建文件
├── models                                                     保存的模型权重
│   ├── pulmonary_chinese_L-12_H-768_A-12_ner.h5
│   ├── pulmonary_chinese_bert_wwm_ner.h5
│   ├── pulmonary_chinese_macbert_base_ner.h5
│   └── pulmonary_chinese_roberta_wwm_ext_L-12_H-768_A-12_ner.h5
├── predict.py                                                 预测
├── report                                                     模型实体F1评估报告
│   ├── pulmonary_chinese_L-12_H-768_A-12_evaluate.txt
│   ├── pulmonary_chinese_L-12_H-768_A-12_predict.json
│   ├── pulmonary_chinese_bert_wwm_evaluate.txt
│   ├── pulmonary_chinese_bert_wwm_predict.json
│   ├── pulmonary_chinese_macbert_base_evaluate.txt
│   ├── pulmonary_chinese_macbert_base_predict.json
│   ├── pulmonary_chinese_roberta_wwm_ext_L-12_H-768_A-12_evaluate.txt
│   └── pulmonary_chinese_roberta_wwm_ext_L-12_H-768_A-12_predict.json
├── requirements.txt                                           pip环境
├── test.py                                                    
├── train.py                                                   训练
└── utils                                                      
    ├── FGM.py                                                 FGM对抗
    ├── __pycache__
    └── path.py                                                所有路径

56 directories, 193 files
```

## Dataset

三甲医院肺结节数据集，20000+字，BIO格式，形如：

```
中	B-ORG
共	I-ORG
中	I-ORG
央	I-ORG
致	O
中	B-ORG
国	I-ORG
致	I-ORG
公	I-ORG
党	I-ORG
十	I-ORG
一	I-ORG
大	I-ORG
的	O
贺	O
词	O
```
ATTENTION: 在处理自己数据集的时候需要注意：
- 字与标签之间用空格（"\ "）隔开
- 其中句子与句子之间使用空行隔开

## Steps

1. 替换数据集
2. 使用DataProcess/load_data.py生成label2id.txt文件
3. 修改config/config.py中的MAX_SEQ_LEN（超过截断，少于填充，最好设置训练集、测试集中最长句子作为MAX_SEQ_LEN）
4. 下载权重，放到项目中
5. 修改public/path.py中的地址
6. 根据需要修改model.py模型结构
7. 修改config/config.py的参数
8. 训练前**debug看下input_train_labels,result_train对不对**，input_train_types全是0
9. 训练

## Model

[BERT](https://github.com/google-research/bert)

[roberta](https://github.com/ymcui/Chinese-BERT-wwm)

[macBERT](https://github.com/ymcui/MacBERT)

[BERT_wwm](https://github.com/ymcui/Chinese-BERT-wwm)

## Train

运行train.py

## Evaluate

运行evaluate/f1_score.py

BERT

```
           precision    recall  f1-score   support

     SIGN     0.6651    0.7354    0.6985       189
  ANATOMY     0.8333    0.8409    0.8371       220
 DIAMETER     1.0000    1.0000    1.0000        16
  DISEASE     0.4915    0.6744    0.5686        43
 QUANTITY     0.8837    0.9157    0.8994        83
TREATMENT     0.3571    0.5556    0.4348         9
  DENSITY     1.0000    1.0000    1.0000         8
    ORGAN     0.4500    0.6923    0.5455        13
LUNGFIELD     1.0000    0.5000    0.6667         6
    SHAPE     0.5714    0.5714    0.5714         7
   NATURE     1.0000    1.0000    1.0000         6
 BOUNDARY     1.0000    0.6250    0.7692         8
   MARGIN     0.8333    0.8333    0.8333         6
  TEXTURE     1.0000    0.8571    0.9231         7

micro avg     0.7436    0.7987    0.7702       621
macro avg     0.7610    0.7987    0.7760       621
```

roberta

```
           precision    recall  f1-score   support

  ANATOMY     0.8624    0.8545    0.8584       220
  DENSITY     0.8000    1.0000    0.8889         8
     SIGN     0.7347    0.7619    0.7481       189
 QUANTITY     0.8977    0.9518    0.9240        83
  DISEASE     0.5690    0.7674    0.6535        43
 DIAMETER     1.0000    1.0000    1.0000        16
TREATMENT     0.3333    0.5556    0.4167         9
 BOUNDARY     1.0000    0.6250    0.7692         8
LUNGFIELD     1.0000    0.6667    0.8000         6
   MARGIN     0.8333    0.8333    0.8333         6
  TEXTURE     1.0000    0.8571    0.9231         7
    SHAPE     0.5714    0.5714    0.5714         7
   NATURE     1.0000    1.0000    1.0000         6
    ORGAN     0.6250    0.7692    0.6897        13

micro avg     0.7880    0.8261    0.8066       621
macro avg     0.8005    0.8261    0.8104       621
```

macBERT

```
           precision    recall  f1-score   support

  ANATOMY     0.8773    0.8773    0.8773       220
     SIGN     0.6538    0.7196    0.6851       189
  DISEASE     0.5893    0.7674    0.6667        43
 QUANTITY     0.9070    0.9398    0.9231        83
    ORGAN     0.5882    0.7692    0.6667        13
  TEXTURE     1.0000    0.8571    0.9231         7
 DIAMETER     1.0000    1.0000    1.0000        16
TREATMENT     0.3750    0.6667    0.4800         9
LUNGFIELD     1.0000    0.5000    0.6667         6
    SHAPE     0.4286    0.4286    0.4286         7
   NATURE     1.0000    1.0000    1.0000         6
  DENSITY     1.0000    1.0000    1.0000         8
 BOUNDARY     1.0000    0.6250    0.7692         8
   MARGIN     0.8333    0.8333    0.8333         6

micro avg     0.7697    0.8180    0.7931       621
macro avg     0.7846    0.8180    0.7977       621
```

BERT_wwm

```
           precision    recall  f1-score   support

  DISEASE     0.5667    0.7907    0.6602        43
  ANATOMY     0.8676    0.8636    0.8656       220
 QUANTITY     0.8966    0.9398    0.9176        83
     SIGN     0.7358    0.7513    0.7435       189
LUNGFIELD     1.0000    0.6667    0.8000         6
TREATMENT     0.3571    0.5556    0.4348         9
 DIAMETER     0.9375    0.9375    0.9375        16
 BOUNDARY     1.0000    0.6250    0.7692         8
  TEXTURE     1.0000    0.8571    0.9231         7
   MARGIN     0.8333    0.8333    0.8333         6
    ORGAN     0.5882    0.7692    0.6667        13
  DENSITY     1.0000    1.0000    1.0000         8
   NATURE     1.0000    1.0000    1.0000         6
    SHAPE     0.5000    0.5714    0.5333         7

micro avg     0.7889    0.8245    0.8063       621
macro avg     0.8020    0.8245    0.8104       621
```

## Predict

运行predict/predict_bio.py