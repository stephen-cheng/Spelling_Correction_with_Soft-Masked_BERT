#!usr/bin/env python
#-*- coding:utf-8 -*-

import os
import tqdm
import pandas as pd
import torch
import torch.nn as nn
from torch import autograd
from torch.optim import Adam
from transformers import BertTokenizer, BertModel, BertConfig
from optim_schedule import ScheduledOptim
from torch.utils.data import Dataset, DataLoader
from model import SoftMaskedBert
MAX_INPUT_LEN = 512
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.autograd.set_detect_anomaly(True)
torch.cuda.set_device(0)


class SoftMaskedBertModel():
    def __init__(self, bert, tokenizer, device, hidden=256, layer_n=1, lr=2e-5, gama=0.8, betas=(0.9, 0.999), weight_decay=0.01, warmup_steps=10000):
        self.device = device
        self.tokenizer = tokenizer
        self.model = SoftMaskedBert(bert, self.tokenizer, hidden, layer_n, self.device).to(self.device)

        if torch.cuda.device_count() > 1:
            print("Using %d GPUS for train" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=[0,1,2])

        optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(optim, hidden, n_warmup_steps=warmup_steps)
        self.criterion_c = nn.NLLLoss()
        self.criterion_d = nn.BCELoss()
        self.gama = gama
        self.log_freq = 100


    def inference(self, data_loader):
        self.model.eval()
        out_put = []
        data_loader = tqdm.tqdm(enumerate(data_loader),
                              desc="%s" % 'Inference:',
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")
        for i, data in data_loader:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            out, prob = self.model(data["input_ids"], data["input_mask"], data["segment_ids"]) #prob [batch_size, seq_len, 1]
            out_put.extend(out.argmax(dim=-1).cpu().numpy().tolist())
        return [''.join(self.tokenizer.convert_ids_to_tokens(x)) for x in out_put]


    def load(self, file_path):
        if not os.path.exists(file_path):
            return
        self.model = torch.load(file_path)
        self.model.to(self.device)

        return avg_loss / len(data_loader)


class BertDataset(Dataset):
    def __init__(self, tokenizer, dataset, max_len=512, pad_first=True, mode='test'):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_len = max_len
        self.data_size = len(dataset)
        self.pad_first = pad_first
        self.mode = mode

    def __len__(self):
        return self.data_size

    def __getitem__(self, item):
        item = self.dataset.iloc[item]
        input_ids = item['noisy']
        input_ids = ['[CLS]'] + list(input_ids)[:min(len(input_ids), self.max_len - 2)] + ['[SEP]']
        # convert to bert ids
        input_ids = self.tokenizer.convert_tokens_to_ids(input_ids)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        pad_len = self.max_len - len(input_ids)
        if self.pad_first:
            input_ids = [0] * pad_len + input_ids
            input_mask = [0] * pad_len + input_mask
            segment_ids = [0] * pad_len + segment_ids
        else:
            input_ids = input_ids + [0] * pad_len
            input_mask = input_mask + [0] * pad_len
            segment_ids = segment_ids + [0] * pad_len

        output = {
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids,
        }

        return {key: torch.tensor(value) for key, value in output.items()}


if __name__ == '__main__':

    dataset = pd.read_csv('dataset/processed_test.csv')
    dataset.dropna(inplace=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert = BertModel.from_pretrained('bert-base-uncased')
    test = BertDataset(tokenizer, dataset, max_len=200)
    test = DataLoader(test, batch_size=8, num_workers=2)
    model = SoftMaskedBertModel(bert, tokenizer, device)
    model.load('checkpoints/best_model_{}ford.pt'.format(0))
    for i in model.inference(test):
        print(i)
        print('\n')
