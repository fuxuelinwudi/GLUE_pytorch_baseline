# coding:utf-8

import gc
import csv
import sys
import warnings
import torch.nn as nn
from torch import multiprocessing
from collections import defaultdict
from argparse import ArgumentParser
from transformers import BertModel, BertPreTrainedModel, BertTokenizer

from src.bert.util.classifier_utils import *

sys.path.append('../../src')
multiprocessing.set_sharing_strategy('file_system')


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 2
        self.dropout = nn.Dropout(0.1)
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):

        encoder_out, pooled_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )

        pooled_out = self.dropout(pooled_out)

        logits = self.classifier(pooled_out)
        outputs = (logits,) + (pooled_out,)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


def read_data(args, tokenizer):
    train_inputs = defaultdict(list)
    with open(args.train_path, 'r', encoding='utf-8') as f:
        for line_id, line in enumerate(f):
            sentence_a, sentence_b, label = line.strip().split('\t')
            label = int(label)
            build_bert_inputs(train_inputs, label, sentence_a, tokenizer, sentence_b)

    dev_inputs = defaultdict(list)
    with open(args.dev_path, 'r', encoding='utf-8') as f:
        for line_id, line in enumerate(f):
            sentence_a, sentence_b, label = line.strip().split('\t')
            label = int(label)
            build_bert_inputs(dev_inputs, label, sentence_a, tokenizer, sentence_b)

    train_cache_pkl_path = os.path.join(args.data_cache_path, 'train.pkl')
    dev_cache_pkl_path = os.path.join(args.data_cache_path, 'dev.pkl')

    save_pickle(train_inputs, train_cache_pkl_path)
    save_pickle(dev_inputs, dev_cache_pkl_path)


def build_model_and_tokenizer(args):
    tokenizer = BertTokenizer.from_pretrained(args.model_path, do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained(args.model_path)
    model.to(args.device)

    return tokenizer, model


def train(args):
    tokenizer, model = build_model_and_tokenizer(args)

    if not os.path.exists(os.path.join(args.data_cache_path, 'train.pkl')):
        read_data(args, tokenizer)

    train_dataloader, dev_dataloader = load_data(args, tokenizer)

    total_steps = args.num_epochs * len(train_dataloader)

    optimizer, scheduler = build_optimizer(args, model, total_steps)

    total_loss, cur_avg_loss, global_steps = 0., 0., 0
    best_acc_score = 0.

    for epoch in range(1, args.num_epochs + 1):

        train_iterator = tqdm(train_dataloader, desc=f'Training epoch : {epoch}', total=len(train_dataloader))

        model.train()

        for batch in train_iterator:
            batch_cuda = batch2cuda(args, batch)
            loss, logits = model(**batch_cuda)[:2]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            total_loss += loss.item()
            cur_avg_loss += loss.item()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if (global_steps + 1) % args.logging_step == 0:

                epoch_avg_loss = cur_avg_loss / args.logging_step
                global_avg_loss = total_loss / (global_steps + 1)

                print(f"\n>> epoch - {epoch},  global steps - {global_steps + 1}, "
                      f"epoch avg loss - {epoch_avg_loss:.4f}, global avg loss - {global_avg_loss:.4f}.")

                metric = evaluation(args, model, dev_dataloader)
                acc, avg_val_loss = metric['acc'], metric['avg_val_loss']

                if acc > best_acc_score:
                    best_acc_score = acc
                    model_save_path = args.output_path
                    save_model(model, tokenizer, model_save_path)

                    print(f'\n>>>\n    best acc - {best_acc_score}, '
                          f'dev loss - {avg_val_loss} .')

                model.train()
                cur_avg_loss = 0.

            global_steps += 1

            train_iterator.set_postfix_str(f'running training loss: {loss.item():.4f}')

    save_model(model, tokenizer, args.output_path)

    print('\n>> evaluate at last .')
    best_model = BertForSequenceClassification.from_pretrained(args.output_path)
    best_model.to(args.device)

    metric = evaluation(args, best_model, dev_dataloader)
    acc, avg_val_loss = metric['acc'], metric['avg_val_loss']

    print(f'\n>>>\n    best acc - {best_acc_score}, dev loss - {avg_val_loss} .')
    os.makedirs(os.path.join(args.output_path, f'acc-{best_acc_score}'), exist_ok=True)

    del model, best_model, tokenizer, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()


def main(task_type):
    parser = ArgumentParser()

    parser.add_argument('--output_path', type=str,
                        default=f'../../user_data/bert/{task_type}/output_model/bert')
    parser.add_argument('--train_path', type=str,
                        default=f'../../download_datasets/data/{task_type}/train.tsv')
    parser.add_argument('--dev_path', type=str,
                        default=f'../../download_datasets/data/{task_type}/valid.tsv')
    parser.add_argument('--data_cache_path', type=str,
                        default=f'../../user_data/bert/process_data/pkl/{task_type}/')

    parser.add_argument('--model_path', type=str,
                        default=f'../../user_data/pretrain_model/bert-base')

    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_seq_len', type=int, default=128)

    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--eps', type=float, default=1e-8)

    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    parser.add_argument('--logging_step', type=int, default=3411)  # evaluate 10 times

    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--device', type=str, default='cuda')

    warnings.filterwarnings('ignore')
    args = parser.parse_args()

    path_list = [args.output_path, args.data_cache_path]
    make_dirs(path_list)

    seed_everything(args.seed)
    train(args)


if __name__ == '__main__':
    main('qqp')
