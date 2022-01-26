# coding:utf-8

import os

data_path = 'data/SST-21'
save_path = 'data/sst2'
os.makedirs(save_path, exist_ok=True)


train_path = os.path.join(data_path, 'train.tsv')
dev_path = os.path.join(data_path, 'dev.tsv')

train_text, train_label, valid_text, valid_label = [], [], [], []

with open(train_path, 'r', encoding='utf-8') as f:
    for line_id, line in enumerate(f):
        if line_id == 0:
            continue
        sentence, label = line.rstrip().split('\t')
        train_text.append(sentence)
        train_label.append(label)

with open(dev_path, 'r', encoding='utf-8') as f:
    for line_id, line in enumerate(f):
        if line_id == 0:
            continue
        sentence, label = line.rstrip().split('\t')
        valid_text.append(sentence)
        valid_label.append(label)


train_save_path = os.path.join(save_path, 'train.tsv')
with open(train_save_path, 'w', encoding='utf-8') as f:
    for i in range(len(train_text)):
        train_sent = train_text[i] + '\t' + str(train_label[i])
        f.write(train_sent + '\n')


valid_save_path = os.path.join(save_path, 'valid.tsv')
with open(valid_save_path, 'w', encoding='utf-8') as f:
    for i in range(len(valid_text)):
        valid_sent = valid_text[i] + '\t' + str(valid_label[i])
        f.write(valid_sent + '\n')

