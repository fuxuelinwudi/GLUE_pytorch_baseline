# coding:utf-8

import os
from datasets import load_dataset

data_path = 'data/MPRC1'
save_path = 'data/mprc'
os.makedirs(save_path, exist_ok=True)


train_path = os.path.join(data_path, 'msr_paraphrase_train.txt')
dev_path = os.path.join(data_path, 'msr_paraphrase_test.txt')

train_text_a, train_text_b, train_label, valid_text_a, valid_text_b, valid_label = [], [], [], [], [], []

with open(train_path, 'r', encoding='utf-8') as f:
    for line_id, line in enumerate(f):
        if line_id == 0:
            continue
        """
        Quality	#1 ID	#2 ID	#1 String	#2 String
        """
        Quality, ID1, ID2, String1, String2 = line.rstrip().split('\t')
        train_text_a.append(String1)
        train_text_b.append(String2)
        label = Quality
        train_label.append(label)

with open(dev_path, 'r', encoding='utf-8') as f:
    for line_id, line in enumerate(f):
        if line_id == 0:
            continue
        Quality, ID1, ID2, String1, String2 = line.rstrip().split('\t')
        valid_text_a.append(String1)
        valid_text_b.append(String2)
        label = Quality
        valid_label.append(label)


train_save_path = os.path.join(save_path, 'train.tsv')
with open(train_save_path, 'w', encoding='utf-8') as f:
    for i in range(len(train_text_a)):
        train_sent = train_text_a[i] + '\t' + train_text_b[i] + '\t' + str(train_label[i])
        f.write(train_sent + '\n')


valid_save_path = os.path.join(save_path, 'valid.tsv')
with open(valid_save_path, 'w', encoding='utf-8') as f:
    for i in range(len(valid_text_a)):
        valid_sent = valid_text_a[i] + '\t' + valid_text_b[i] + '\t' + str(valid_label[i])
        f.write(valid_sent + '\n')

