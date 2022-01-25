# coding:utf-8

import os
from datasets import load_dataset

data_path = 'data/MNLI1'
save_path = 'data/mnli'
os.makedirs(save_path, exist_ok=True)


train_path = os.path.join(data_path, 'train.tsv')
dev_matched_path = os.path.join(data_path, 'dev_matched.tsv')
dev_mismatched_path = os.path.join(data_path, 'dev_mismatched.tsv')

train_text_a, train_text_b, train_label, valid_matched_text_a, valid_matched_text_b, valid_matched_label, \
valid_mismatched_text_a, valid_mismatched_text_b, valid_mismatched_label = [], [], [], [], [], [], [], [], []

with open(train_path, 'r', encoding='utf-8') as f:
    for line_id, line in enumerate(f):
        if line_id == 0:
            continue
        index, promptID, pairID, genre, sentence1_binary_parse, sentence2_binary_parse, sentence1_parse, \
        sentence2_parse, sentence1, sentence2, label1, gold_label = line.rstrip().split('\t')
        train_text_a.append(sentence1)
        train_text_b.append(sentence2)
        if gold_label == 'entailment':
            label = 0
        elif gold_label == 'neutral':
            label = 1
        else:
            label = 2
        train_label.append(label)

with open(dev_matched_path, 'r', encoding='utf-8') as f:
    for line_id, line in enumerate(f):
        if line_id == 0:
            continue
        index, promptID, pairID, genre, sentence1_binary_parse, sentence2_binary_parse, sentence1_parse, \
        sentence2_parse, sentence1, sentence2, label1, label2, label3, label4, label5, gold_label = \
            line.rstrip().split('\t')
        valid_matched_text_a.append(sentence1)
        valid_matched_text_b.append(sentence2)
        if gold_label == 'entailment':
            label = 0
        elif gold_label == 'neutral':
            label = 1
        else:
            label = 2
        valid_matched_label.append(label)

with open(dev_mismatched_path, 'r', encoding='utf-8') as f:
    for line_id, line in enumerate(f):
        if line_id == 0:
            continue
        index, promptID, pairID, genre, sentence1_binary_parse, sentence2_binary_parse, sentence1_parse, \
        sentence2_parse, sentence1, sentence2, label1, label2, label3, label4, label5, gold_label = \
            line.rstrip().split('\t')
        valid_mismatched_text_a.append(sentence1)
        valid_mismatched_text_b.append(sentence2)
        if gold_label == 'entailment':
            label = 0
        elif gold_label == 'neutral':
            label = 1
        else:
            label = 2
        valid_mismatched_label.append(label)

train_save_path = os.path.join(save_path, 'train.tsv')
with open(train_save_path, 'w', encoding='utf-8') as f:
    for i in range(len(train_text_a)):
        train_sent = train_text_a[i] + '\t' + train_text_b[i] + '\t' + str(train_label[i])
        f.write(train_sent + '\n')


valid_matched_save_path = os.path.join(save_path, 'valid_matched.tsv')
with open(valid_matched_save_path, 'w', encoding='utf-8') as f:
    for i in range(len(valid_matched_text_a)):
        valid_sent = valid_matched_text_a[i] + '\t' + valid_matched_text_b[i] + '\t' +  str(valid_matched_label[i])
        f.write(valid_sent + '\n')


valid_mismatched_save_path = os.path.join(save_path, 'valid_mismatched.tsv')
with open(valid_mismatched_save_path, 'w', encoding='utf-8') as f:
    for i in range(len(valid_mismatched_text_a)):
        valid_sent = valid_mismatched_text_a[i] + '\t' + valid_mismatched_text_b[i] + '\t' + str(valid_mismatched_label[i])
        f.write(valid_sent + '\n')
