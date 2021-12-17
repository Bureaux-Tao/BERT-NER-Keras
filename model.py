# -*- coding: utf-8 -*-
import json
import keras
from keras.layers import *
from keras.models import Model
from keras_bert import load_trained_model_from_checkpoint
from keras_contrib.layers import CRF

from utils.path import event_type, BASE_MODEL_DIR, BASE_CONFIG_NAME, BASE_CKPT_NAME


# 创建BERT-BiLSTM-CRF模型
class BertBilstmCRF:
    def __init__(self, max_seq_length, lstm_dim):
        self.max_seq_length = max_seq_length
        self.lstmDim = lstm_dim
        self.label = self.load_label()
    
    # 抽取的标签
    def load_label(self):
        label_path = "./config/{}_label2id.json".format(event_type)
        with open(label_path, 'r', encoding = 'utf-8') as f_label:
            label = json.loads(f_label.read())
        
        return label
    
    # 模型
    def create_model(self):
        model_path = "./{}/".format(BASE_MODEL_DIR)
        print('')
        print('load bert Model start!')
        bert = load_trained_model_from_checkpoint(
            model_path + BASE_CONFIG_NAME,
            model_path + BASE_CKPT_NAME,
            seq_len = self.max_seq_length
        )
        # make bert layer trainable
        for layer in bert.layers:
            layer.trainable = True
        # x1 = Input(shape=(None,))
        # x2 = Input(shape=(None,))
        # bert_out = bert([x1, x2])
        print('load bert Model end!')
        # bert.output = Dropout(0.5)(bert.output)
        
        ###  ---1---
        lstm_out = Bidirectional(LSTM(int(self.lstmDim),
                                      return_sequences = True,
                                      dropout = 0.5,
                                      recurrent_dropout = 0.5))(bert.output)
        
        dense = TimeDistributed(Dense(len(self.label)))(lstm_out)
        dense = Dropout(0.2)(dense)
        crf_out = CRF(len(self.label), sparse_target = True)(dense)
        model = Model(bert.input, crf_out)
        model.summary()
        return model
        
        ###  ---2---
        # lstm_out = Bidirectional(LSTM(self.lstmDim,
        #                               return_sequences=True,
        #                               dropout=0.5,
        #                               recurrent_dropout=0.5,
        #                               kernel_regularizer=keras.regularizers.l2(0.01)
        #                               ))(bert.output)
        # lstm_out = Dropout(0.5)(lstm_out)
        # dense_out = Dense(len(self.label), kernel_regularizer=keras.regularizers.l2(0.01))(lstm_out)
        # dense_out = Dropout(0.5)(dense_out)
        # crf_out = CRF(len(self.label), sparse_target=True, kernel_regularizer=keras.regularizers.l2(0.01))(dense_out)
        # model = Model(bert.input, crf_out)
        # model.summary()
        # return model
        
        ### ---3---
        # lstm_out = Bidirectional(LSTM(self.lstmDim,
        #                               return_sequences=True,
        #                               dropout=0.5,
        #                               recurrent_dropout=0.5,
        #                               kernel_regularizer=keras.regularizers.l2(0.01)
        #                               ))(bert.output)
        # lstm_out = Dropout(0.5)(lstm_out)
        # crf_out = CRF(len(self.label), sparse_target=True, kernel_regularizer=keras.regularizers.l2(0.01))(lstm_out)
        # model = Model(bert.input, crf_out)
        # model.summary()
        # return model
