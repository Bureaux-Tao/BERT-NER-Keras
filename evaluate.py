# -*- coding: utf-8 -*-
# 利用seqeval模块对序列标注的结果进行评估
import json
import numpy as np
from keras.models import load_model
from keras_bert import get_custom_objects
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy
from seqeval.metrics import classification_report

from DataProcess.load_data import read_data
from utils.path import event_type, BASE_MODEL_DIR, test_file_path
from train import PreProcessInputData, id_label_dict

custom_objects = get_custom_objects()
for key, value in {'CRF': CRF, 'crf_loss': crf_loss, 'crf_accuracy': crf_accuracy}.items():
    custom_objects[key] = value

epoch = '21'
model = load_model("./models/{}_{}_ep{}.h5".format(event_type, BASE_MODEL_DIR, epoch), custom_objects = custom_objects)


# 对单句话进行预测
def predict_single_sentence(text):
    # 测试句子
    word_labels, seq_types = PreProcessInputData([text])
    # 模型预测
    predicted = model.predict([word_labels, seq_types])
    y = np.argmax(predicted[0], axis = 1)
    predict_tag = [id_label_dict[_] for _ in y]
    return predict_tag[1:-1]


if __name__ == '__main__':
    # 读取测试集数据
    input_test, result_test = read_data('./data/yidu_submit.txt' )
    for sent, tag in zip(input_test[:10], result_test[:10]):
        print(sent, tag)
    
    # 测试集
    i = 1
    true_tag_list = []
    pred_tag_list = []
    predict_samples = []
    for test_text, true_tag in zip(input_test, result_test):
        print("Predict %d samples" % i)
        pred_tag = predict_single_sentence(text = test_text)
        true_tag_list.append(true_tag)
        if len(true_tag) <= len(pred_tag):
            pred_tag_list.append(pred_tag[:len(true_tag)])
            predict_samples.append({"text": test_text, "tags": pred_tag[:len(true_tag)]})
        else:
            pred_tag_list.append(pred_tag + ["O"] * (len(true_tag) - len(pred_tag)))
            predict_samples.append({"text": test_text, "tags": pred_tag + ["O"] * (len(true_tag) - len(pred_tag))})
        i += 1
    
    result = classification_report(true_tag_list, pred_tag_list, digits = 4)
    print(result)
    
    with open("{}_{}_predict.json".format(event_type, BASE_MODEL_DIR), "w", encoding = "utf-8") as g:
        g.write(json.dumps(predict_samples, ensure_ascii = False, indent = 2))
    
    with open("{}_{}_evaluate.txt".format(event_type, BASE_MODEL_DIR), "w", encoding = "utf-8") as h:
        h.write(result)
