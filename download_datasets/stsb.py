# coding:utf-8

import os

data_path = 'data/STS-B1'
save_path = 'data/sts-b'
os.makedirs(save_path, exist_ok=True)


train_path = os.path.join(data_path, 'train.tsv')
dev_path = os.path.join(data_path, 'dev.tsv')

train_text_a, train_text_b, train_label, valid_text_a, valid_text_b, valid_label = [], [], [], [], [], []

with open(train_path, 'r', encoding='utf-8') as f:
    for line_id, line in enumerate(f):
        if line_id == 0:
            continue
        index, genre, filename, year, old_index, source1, source2, sentence1, sentence2, score = \
            line.rstrip().split('\t')
        train_text_a.append(sentence1)
        train_text_b.append(sentence2)
        score = float(score)
        if 4.0 < score <= 5.0:
            label = 4
        elif 3.0 < score <= 4.0:
            label = 3
        elif 2.0 < score <= 3.0:
            label = 2
        elif 1.0 < score <= 2.0:
            label = 1
        else:
            label = 0
        train_label.append(label)

with open(dev_path, 'r', encoding='utf-8') as f:
    for line_id, line in enumerate(f):
        if line_id == 0:
            continue
        index, genre, filename, year, old_index, source1, source2, sentence1, sentence2, score = \
            line.rstrip().split('\t')
        valid_text_a.append(sentence1)
        valid_text_b.append(sentence2)
        score = float(score)
        if 4.0 < score <= 5.0:
            label = 4
        elif 3.0 < score <= 4.0:
            label = 3
        elif 2.0 < score <= 3.0:
            label = 2
        elif 1.0 < score <= 2.0:
            label = 1
        else:
            label = 0
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

