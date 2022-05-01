#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


class KEDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512, is_test=False, target_names=None):
        # target_names - список наименований категорий в one-hot представлении целевой переменной. Формат элементов - str.
        self.df = df
        self.target_names = target_names
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        text  = row["text"]
        if not self.is_test: 
            label = row[self.target_names]
            true_class = row['category_id']

        tokenized_text = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True)

        res = {
            "input_ids": torch.Tensor(tokenized_text["input_ids"]).long(),
            "token_type_ids": torch.Tensor(tokenized_text["token_type_ids"]).long(),
            "attention_mask": torch.Tensor(tokenized_text["attention_mask"]).long(),
            }
        if not self.is_test: res["label"] = torch.Tensor(label).float()
        if not self.is_test: res["true_class"] = true_class
        
        return res
    

class BertClasifyByCLS(nn.Module):
    def __init__(self, bert, emd_size=264, num_classes=2):
        super(BertClasifyByCLS, self).__init__()
        self.bert = bert
        self.num_classes = num_classes
        self.drop_out = nn.Dropout(0.1)
        self.clsf = nn.Linear(emd_size, self.num_classes)

    def forward(self, input_ids_text,
                      token_type,
                      attention_mask_text):
        bert_text_out = self.bert(input_ids_text,
                                  token_type,
                                  attention_mask_text)[0][:,0]
        clsf_input = self.drop_out(bert_text_out)
        return self.clsf(clsf_input)

