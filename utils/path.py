origincal_csv_data = '../data/originalData.csv'
unlabeled_bio = '../data/unlabeled.txt'
number_handled_bio = '../data/unlabeled_numbered.txt'
jsonl = '../data/labdeled.jsonl'

event_type = "pulmonary"

train_file_path = "../data/%s.train" % event_type
test_file_path = "../data/%s.test" % event_type

all_path = '../data/all.txt'

# BASE_MODEL_DIR = "chinese_L-12_H-768_A-12"
# BASE_CONFIG_NAME = "bert_config.json"
# BASE_CKPT_NAME = "bert_model.ckpt"

BASE_MODEL_DIR = "chinese_roberta_wwm_ext_L-12_H-768_A-12"
BASE_CONFIG_NAME = "bert_config.json"
BASE_CKPT_NAME = "bert_model.ckpt"

# BASE_MODEL_DIR = "chinese_macbert_base"
# BASE_CONFIG_NAME = "macbert_base_config.json"
# BASE_CKPT_NAME = "chinese_macbert_base.ckpt"

# BASE_MODEL_DIR = "chinese_bert_wwm"
# BASE_CONFIG_NAME = "bert_config.json"
# BASE_CKPT_NAME = "bert_model.ckpt"
