# coding:utf-8

import re
import os
import torch
import pickle
import random
import shutil
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from torch.optim.lr_scheduler import LambdaLR
from transformers import AdamW, BertTokenizer
from torch.utils.data import DataLoader, Dataset
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


seed_everything(2021)


def save_pickle(dic, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(dic, f)


def load_pickle(load_path):
    with open(load_path, 'rb') as f:
        message_dict = pickle.load(f)
    return message_dict


def save_model(model, tokenizer, saving_path):
    os.makedirs(saving_path, exist_ok=True)
    model_to_save = model

    output_model_file = os.path.join(saving_path, 'pytorch_model.bin')
    output_config_file = os.path.join(saving_path, 'config.json')

    torch.save(model_to_save.state_dict(), output_model_file)

    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(saving_path)


def sorted_checkpoints(args, best_model_checkpoint, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False):
    ordering_and_checkpoint_path = []

    glob_checkpoints = [str(x) for x in Path(args.output_path).glob(f"{checkpoint_prefix}-*")]

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    # Make sure we don't delete the best model.
    if best_model_checkpoint is not None:
        best_model_index = checkpoints_sorted.index(str(Path(best_model_checkpoint)))
        checkpoints_sorted[best_model_index], checkpoints_sorted[-1] = (
            checkpoints_sorted[-1],
            checkpoints_sorted[best_model_index],
        )
    return checkpoints_sorted


def rotate_checkpoints(args, best_model_checkpoint, use_mtime=False) -> None:
    if args.save_total_limit is None or args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = sorted_checkpoints(args, best_model_checkpoint, use_mtime=use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        shutil.rmtree(checkpoint)


def batch2cuda(args, batch):
    return {item: value.to(args.device) for item, value in list(batch.items())}


def build_bert_inputs(inputs, label, sentence, tokenizer, sentence_b=None):
    if sentence_b is not None:
        inputs_dict = tokenizer.encode_plus(sentence, sentence_b, add_special_tokens=True,
                                            return_token_type_ids=True, return_attention_mask=True)
    else:
        inputs_dict = tokenizer.encode_plus(sentence, add_special_tokens=True,
                                            return_token_type_ids=True, return_attention_mask=True)

    input_ids = inputs_dict['input_ids']
    token_type_ids = inputs_dict['token_type_ids']
    attention_mask = inputs_dict['attention_mask']

    inputs['input_ids'].append(input_ids)
    inputs['token_type_ids'].append(token_type_ids)
    inputs['attention_mask'].append(attention_mask)
    inputs['labels'].append(label)


class KGDataset(Dataset):
    def __init__(self, data_dict: dict, tokenizer: BertTokenizer):
        super(KGDataset, self).__init__()
        self.data_dict = data_dict
        self.tokenizer = tokenizer

    def __getitem__(self, index: int) -> tuple:
        data = (
            self.data_dict['input_ids'][index],
            self.data_dict['token_type_ids'][index],
            self.data_dict['attention_mask'][index],
            self.data_dict['labels'][index]
        )

        return data

    def __len__(self) -> int:
        return len(self.data_dict['input_ids'])


class Collator:
    def __init__(self, max_seq_len: int, tokenizer: BertTokenizer):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def pad_and_truncate(self, input_ids_list, token_type_ids_list, attention_mask_list, labels_list, max_seq_len):

        input_ids = torch.zeros((len(input_ids_list), max_seq_len), dtype=torch.long)
        token_type_ids = torch.zeros_like(input_ids)
        attention_mask = torch.zeros_like(input_ids)

        for i in range(len(input_ids_list)):
            seq_len = len(input_ids_list[i])

            # pad
            if seq_len <= max_seq_len:
                input_ids[i, :seq_len] = torch.tensor(input_ids_list[i], dtype=torch.long)
                token_type_ids[i, :seq_len] = torch.tensor(token_type_ids_list[i], dtype=torch.long)
                attention_mask[i, :seq_len] = torch.tensor(attention_mask_list[i], dtype=torch.long)

            # cut
            else:
                input_ids[i] = torch.tensor(input_ids_list[i][:max_seq_len - 1] + [self.tokenizer.sep_token_id],
                                            dtype=torch.long)
                token_type_ids[i] = torch.tensor(token_type_ids_list[i][:max_seq_len], dtype=torch.long)
                attention_mask[i] = torch.tensor(attention_mask_list[i][:max_seq_len], dtype=torch.long)

        labels = torch.tensor(labels_list, dtype=torch.long)

        return input_ids, token_type_ids, attention_mask, labels

    def __call__(self, examples: list) -> dict:
        input_ids_list, token_type_ids_list, attention_mask_list, labels_list = list(zip(*examples))

        cur_max_seq_len = max(len(input_id) for input_id in input_ids_list)
        max_seq_len = min(cur_max_seq_len, self.max_seq_len)

        input_ids, token_type_ids, attention_mask, labels = \
            self.pad_and_truncate(input_ids_list, token_type_ids_list, attention_mask_list, labels_list, max_seq_len)

        data_dict = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

        return data_dict


def load_data(args, tokenizer):
    train_cache_pkl_path = os.path.join(args.data_cache_path, 'train.pkl')
    dev_cache_pkl_path = os.path.join(args.data_cache_path, 'dev.pkl')

    with open(train_cache_pkl_path, 'rb') as f:
        train_data = pickle.load(f)

    with open(dev_cache_pkl_path, 'rb') as f:
        dev_data = pickle.load(f)

    collate_fn = Collator(args.max_seq_len, tokenizer)

    train_dataset = KGDataset(train_data, tokenizer)
    dev_dataset = KGDataset(dev_data, tokenizer)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=0, collate_fn=collate_fn, pin_memory=True)
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=8, shuffle=False,
                                num_workers=0, collate_fn=collate_fn, pin_memory=True)

    return train_dataloader, dev_dataloader


class WarmupLinearSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


def build_optimizer(args, model, train_steps):
    no_decay = ['bias', 'LayerNorm.weight']

    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.eps)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=train_steps * args.warmup_ratio, t_total=train_steps)

    return optimizer, scheduler


def evaluation(args, model, val_dataloader):
    model.eval()

    metric = {}
    preds, labels = [], []
    val_loss = 0.

    val_iterator = tqdm(val_dataloader, desc='Evaluation', total=len(val_dataloader))

    with torch.no_grad():
        for batch in val_iterator:
            batch_cuda = batch2cuda(args, batch)
            loss, logits = model(**batch_cuda)[:2]
            val_loss += loss.item()

            preds.extend([i for i in torch.argmax(torch.softmax(logits, -1), 1).cpu().numpy().tolist()])
            labels.extend([i for i in batch_cuda['labels'].cpu().numpy().tolist()])

    avg_val_loss = val_loss / len(val_dataloader)

    acc = accuracy_score(y_true=labels, y_pred=preds)
    avg_val_loss, acc = round(avg_val_loss, 4), round(acc, 4)

    metric['acc'], metric['avg_val_loss'] = acc, avg_val_loss

    return metric


def evaluation_f1(args, model, val_dataloader):
    model.eval()

    metric = {}
    preds, labels = [], []
    val_loss = 0.

    val_iterator = tqdm(val_dataloader, desc='Evaluation', total=len(val_dataloader))

    with torch.no_grad():
        for batch in val_iterator:
            batch_cuda = batch2cuda(args, batch)
            loss, logits = model(**batch_cuda)[:2]
            val_loss += loss.item()

            preds.extend([i for i in torch.argmax(torch.softmax(logits, -1), 1).cpu().numpy().tolist()])
            labels.extend([i for i in batch_cuda['labels'].cpu().numpy().tolist()])

    avg_val_loss = val_loss / len(val_dataloader)

    f1 = f1_score(y_true=labels, y_pred=preds)
    avg_val_loss, f1 = round(avg_val_loss, 4), round(f1, 4)
    f1 = round(f1, 4)

    metric['f1'], metric['avg_val_loss'] = f1, avg_val_loss
    metric['f1'] = f1

    return metric


def evaluation_mcc(args, model, val_dataloader):
    model.eval()

    metric = {}
    preds, labels = [], []
    val_loss = 0.

    val_iterator = tqdm(val_dataloader, desc='Evaluation', total=len(val_dataloader))

    with torch.no_grad():
        for batch in val_iterator:
            batch_cuda = batch2cuda(args, batch)
            loss, logits = model(**batch_cuda)[:2]
            val_loss += loss.item()

            preds.extend([i for i in torch.argmax(torch.softmax(logits, -1), 1).cpu().numpy().tolist()])
            labels.extend([i for i in batch_cuda['labels'].cpu().numpy().tolist()])

    avg_val_loss = val_loss / len(val_dataloader)

    mcc = matthews_corrcoef(y_true=labels, y_pred=preds)
    avg_val_loss, mcc = round(avg_val_loss, 4), round(mcc, 4)

    metric['mcc'], metric['avg_val_loss'] = mcc, avg_val_loss

    return metric


def evaluation_psc(args, model, val_dataloader):
    model.eval()

    metric = {}
    preds, labels = [], []
    val_loss = 0.

    val_iterator = tqdm(val_dataloader, desc='Evaluation', total=len(val_dataloader))

    with torch.no_grad():
        for batch in val_iterator:
            batch_cuda = batch2cuda(args, batch)
            loss, logits = model(**batch_cuda)[:2]
            val_loss += loss.item()

            preds.extend([i for i in torch.argmax(torch.softmax(logits, -1), 1).cpu().numpy().tolist()])
            labels.extend([i for i in batch_cuda['labels'].cpu().numpy().tolist()])

    avg_val_loss = val_loss / len(val_dataloader)

    pearson_corr = pearsonr(preds, labels)[0]
    # spearman_corr = spearmanr(preds, labels)[0]

    avg_val_loss, pc = round(avg_val_loss, 4), round(pearson_corr, 4)

    metric['pc'], metric['avg_val_loss'] = pc, avg_val_loss

    return metric


def make_dirs(path_list):
    for i in path_list:
        os.makedirs(os.path.dirname(i), exist_ok=True)
