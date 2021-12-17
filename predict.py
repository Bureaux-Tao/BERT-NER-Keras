# -*- coding: utf-8 -*-
import re

import numpy as np
from pprint import pprint
from keras.models import load_model
from keras_bert import get_custom_objects
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy
from model_train import PreProcessInputData, id_label_dict


# 将BIO标签转化为方便阅读的json格式
def bio_to_json(string, tags):
    item = {"string": string, "entities": []}
    entity_name = ""
    entity_start = 0
    iCount = 0
    entity_tag = ""

    for c_idx in range(min(len(string), len(tags))):
        c, tag = string[c_idx], tags[c_idx]
        if c_idx < len(tags) - 1:
            tag_next = tags[c_idx + 1]
        else:
            tag_next = ''

        if tag[0] == 'B':
            entity_tag = tag[2:]
            entity_name = c
            entity_start = iCount
            if tag_next[2:] != entity_tag:
                item["entities"].append({"word": c, "start": iCount, "end": iCount + 1, "type": tag[2:]})
        elif tag[0] == "I":
            if tag[2:] != tags[c_idx - 1][2:] or tags[c_idx - 1][2:] == 'O':
                tags[c_idx] = 'O'
                pass
            else:
                entity_name = entity_name + c
                if tag_next[2:] != entity_tag:
                    item["entities"].append(
                        {"word": entity_name, "start": entity_start, "end": iCount + 1, "type": entity_tag})
                    entity_name = ''
        iCount += 1
    return item


# 加载训练好的模型
custom_objects = get_custom_objects()
for key, value in {'CRF': CRF, 'crf_loss': crf_loss, 'crf_accuracy': crf_accuracy}.items():
    custom_objects[key] = value
model = load_model("./models/pulmonary_chinese_L-12_H-768_A-12_ner.h5", custom_objects=custom_objects)

# 测试句子
text = \
    "与2018年-06-04旧片对比，结合MPR显示：纵膈各大血管结构清楚，血管间隙内未见明显肿大淋巴结。右横隔见数枚肿大淋巴结较前退缩，现显示不清（4:9）。左肺下叶后基底段见不规则结节灶较前稍缩小，现最大截面约1.1cm*0.9cm（7.15），边界尚清；右肺中下叶见散在数枚直径小于0.5cm的模糊小结节影与前大致相仿（7:18、30、36）；双肺尖见少许斑片、条索影较前无明显变化，余肺野未见明显实质性病变。双侧胸腔内未见明显胸水征。"
# "男患，64岁。2001年体检时胸部CT示右上肺结节病灶，直径约0.75cm。复查X线胸片或胸部CT病灶均无明显变化。2008年6月E查胸部CT示右上肺结节略增大，约1.46cm*1.10cm，边缘可见细小毛刺，邻近胸膜粘连凹陷；右肺中叶外侧段见片状模糊影；两肺散在多个类圆形小结节影；纵隔淋巴结显示。图2中右肺尖外带见小结节，斑片状高密度影，邻近胸膜凹陷性改变；右下肺见小结节密度增高影及斑片状，索条状高密度影。图3中右肺上叶、中叶多发斑片影，两下肺纤维索条影，右肺下叶后基底段结节影，邻近胸膜增厚粘连。图4中右肺上叶前段见一结节影，大小约1.25*1.10cm，边缘可见 细小毛刺，邻近胸膜粘连凹陷；右肺中叶外侧段见片状模糊影。右肺尖少量条索影，两肺散在多个类圆形小结节影。图5中右肺上叶前段见一影,大小约1.46*1.10cm，边缘可见细小毛刺，邻近胸膜粘连凹陷，右肺中叶外侧段见片状模糊影；两肺散在多个类圆形小结节影。"

for i in re.split("[。！？；]", text):
    word_labels, seq_types = PreProcessInputData([i])

    # 模型预测
    predicted = model.predict([word_labels, seq_types])
    y = np.argmax(predicted[0], axis=1)
    tag = [id_label_dict[_] for _ in y]

    # 输出预测结果
    result = bio_to_json(i, tag[1:-1])
    pprint(result)
    print('')
