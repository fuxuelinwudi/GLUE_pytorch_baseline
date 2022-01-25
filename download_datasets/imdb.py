# coding:utf-8

import os
from datasets import load_dataset


imdb_train = load_dataset('imdb', split='train')
imdb_test = load_dataset('imdb', split='test')

train_datasets_length = 25000
test_datasets_length = 25000

save_path = 'data/imdb'
os.makedirs(save_path, exist_ok=True)

train_text, train_label, test_text, test_label = [], [], [], []
for i in range(train_datasets_length):
    text = imdb_train[i]['text']
    label = imdb_train[i]['label']

    train_text.append(text)
    train_label.append(label)

for i in range(test_datasets_length):
    text = imdb_test[i]['text']
    label = imdb_test[i]['label']

    test_text.append(text)
    test_label.append(label)

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
