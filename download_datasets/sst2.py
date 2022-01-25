# coding:utf-8

import os
from datasets import load_dataset


sst_train = load_dataset('sst', split='train')
sst_test = load_dataset('sst', split='test')
sst_valid = load_dataset('sst', split='validation')

print(sst_train)
print(sst_test)
print(sst_valid)

train_datasets_length = 8544
test_datasets_length = 2210
valid_datasets_length = 1101

save_path = 'data/sst2'
os.makedirs(save_path, exist_ok=True)

train_text, train_label, test_text, test_label, valid_text, valid_label = [], [], [], [], [], []
for i in range(train_datasets_length):
    text = sst_train[i]['sentence']
    label = sst_train[i]['label']
    if label >= 0.5:
        label = 1
    else:
        label = 0
    train_text.append(text)
    train_label.append(label)

for i in range(test_datasets_length):
    text = sst_test[i]['sentence']
    label = sst_test[i]['label']
    if label >= 0.5:
        label = 1
    else:
        label = 0
    test_text.append(text)
    test_label.append(label)

for i in range(valid_datasets_length):
    text = sst_valid[i]['sentence']
    label = sst_valid[i]['label']
    if label >= 0.5:
        label = 1
    else:
        label = 0
    valid_text.append(text)
    valid_label.append(label)


train_save_path = os.path.join(save_path, 'train.tsv')
with open(train_save_path, 'w', encoding='utf-8') as f:
    for i in range(len(train_text)):
        train_sent = train_text[i] + '\t' + str(train_label[i])
        f.write(train_sent + '\n')

test_save_path = os.path.join(save_path, 'test.tsv')
with open(test_save_path, 'w', encoding='utf-8') as f:
    for i in range(len(test_text)):
        test_sent = test_text[i] + '\t' + str(test_label[i])
        f.write(test_sent + '\n')

valid_save_path = os.path.join(save_path, 'valid.tsv')
with open(valid_save_path, 'w', encoding='utf-8') as f:
    for i in range(len(valid_text)):
        valid_sent = valid_text[i] + '\t' + str(valid_label[i])
        f.write(valid_sent + '\n')
