# -*- coding: utf-8 -*-
import json
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
from keras.preprocessing.sequence import pad_sequences
from keras_bert import Tokenizer
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy
from keras_bert import AdamWarmup, calc_train_steps

from utils.path import event_type, train_file_path, test_file_path, BASE_CONFIG_NAME, BASE_MODEL_DIR, BASE_CKPT_NAME
from config.config import MAX_SEQ_LEN, BATCH_SIZE, EPOCH
from DataProcess.load_data import read_data
from model import BertBilstmCRF
from utils.FGM import adversarial_training

# 读取label2id字典
with open("./config/{}_label2id.json".format(event_type), "r", encoding = "utf-8") as h:
    label_id_dict = json.loads(h.read())

id_label_dict = {v: k for k, v in label_id_dict.items()}

# 载入数据
config_path = './{}/{}'.format(BASE_MODEL_DIR, BASE_CONFIG_NAME)
checkpoint_path = './{}/{}'.format(BASE_MODEL_DIR, BASE_CKPT_NAME)
dict_path = './{}/vocab.txt'.format(BASE_MODEL_DIR)

token_dict = {}
with open(dict_path, 'r', encoding = 'utf-8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


tokenizer = OurTokenizer(token_dict)


# 预处理输入数据
def PreProcessInputData(text):
    word_labels = []
    seq_types = []
    for sequence in text:
        code = tokenizer.encode(first = sequence, max_len = MAX_SEQ_LEN)
        word_labels.append(code[0])
        seq_types.append(code[1])
    return word_labels, seq_types


# 预处理结果数据
def PreProcessOutputData(text):
    tags = []
    for line in text:
        tag = [0]
        for item in line:
            tag.append(int(label_id_dict[item.strip()]))
        tag.append(0)
        tags.append(tag)
    
    pad_tags = pad_sequences(tags, maxlen = MAX_SEQ_LEN, padding = "post", truncating = "post")
    result_tags = np.expand_dims(pad_tags, 2)
    return result_tags


if __name__ == '__main__':
    # 读取训练集和测试集数据
    input_train, result_train = read_data('./data/pulmonary.train')
    input_test, result_test = read_data('./data/pulmonary.test')
    for sent, tag in zip(input_train[:10], result_train[:10]):
        print(sent, tag)
    for sent, tag in zip(input_test[:10], result_test[:10]):
        print(sent, tag)
    
    # 训练集
    input_train_labels, input_train_types = PreProcessInputData(input_train)
    print(input_train_types[0])
    result_train = PreProcessOutputData(result_train)
    # 测试集
    input_test_labels, input_test_types = PreProcessInputData(input_test)
    result_test = PreProcessOutputData(result_test)
    # add warmup
    total_steps, warmup_steps = calc_train_steps(
        num_example = len(input_train),
        batch_size = BATCH_SIZE,
        epochs = EPOCH,
        warmup_proportion = 0.1,
    )
    optimizer = AdamWarmup(total_steps, warmup_steps, lr = 1e-4, min_lr = 1e-9)
    model = BertBilstmCRF(max_seq_length = MAX_SEQ_LEN, lstm_dim = 128).create_model()
    model.compile(
        optimizer = optimizer,
        loss = crf_loss,
        metrics = [crf_accuracy]
    )
    # 启用对抗训练FGM
    adversarial_training(model, 'Embedding-Token', 0.5)
    
    save_file_path = "./models/{}_{}_ner.h5".format(event_type, BASE_MODEL_DIR)
    callbacks_list = [
        keras.callbacks.History(),
        keras.callbacks.EarlyStopping(monitor = 'loss', patience = 10, verbose = 2),  # 提前结束
        keras.callbacks.ModelCheckpoint(save_file_path, monitor = 'loss', verbose = 1, save_best_only = True,
                                        mode = 'min')
    ]
    
    history = model.fit(x = [input_train_labels, input_train_types],
                        y = result_train,
                        batch_size = BATCH_SIZE,
                        epochs = EPOCH,
                        # validation_data=[[input_test_labels, input_test_types], result_test],
                        verbose = 1,
                        shuffle = True,
                        callbacks = callbacks_list
                        )
    
    # 保存模型
    # model.save("./models/{}_{}_ner.h5".format(event_type, BASE_MODEL_DIR))
    
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['savefig.dpi'] = 360  # 图片像素
    plt.rcParams['figure.dpi'] = 360  # 分辨率
    
    # 绘制loss和acc图像
    plt.subplot(2, 1, 1)
    epochs = len(history.history['loss'])
    plt.plot(range(epochs), history.history['loss'], label = 'loss')
    # plt.plot(range(epochs), history.history['val_loss'], label='val_loss')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    epochs = len(history.history['crf_accuracy'])
    plt.plot(range(epochs), history.history['crf_accuracy'], label = 'crf_accuracy')
    # plt.plot(range(epochs), history.history['val_crf_accuracy'], label='val_crf_accuracy')
    plt.legend()
    plt.savefig("%s_loss_acc.png" % event_type)
