#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import networkx as nx
import random, os

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import f1_score

from tqdm import tqdm


def seed_everything(seed: int):   
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_path(G, pred):
    #по предсказанной категории возвращает всю ветку до корня
    # G - граф со структурой категорий
    # pred - предсказанная категория, название категории в формате int
    # возвращает всю ветку категорий от предсказанной категории до корня
    node = pred
    path = [node]
    while list(G.predecessors(node)):
        path.append(list(G.predecessors(node))[0])
        node = list(G.predecessors(node))[0]
    return set(path)

def get_hF(G, y_true, y_pred):
    # G - граф со структурой категорий
    # y_true - истинные метки классов,  названия категории в формате int
    # y_pred - предсказанные метки классов,  названия категории в формате int
    # возвращает рассчитанное значение иерархического f1.
    assert len(y_true) == len(y_pred)
    P_T = 0
    P = 0
    T = 0
    for i in range(len(y_true)):
        P_i = get_path(G, y_pred[i]) 
        T_i = get_path(G, y_true[i])
        P_T += len(P_i.intersection(T_i))
        P += len(P_i)
        T += len(T_i)
    hP = P_T / P
    hR = P_T / T
    hF = 2 * hP * hR / (hP + hR)
    return hF

def train_epoch(model, loss_function, optimizer, train_loader, val_loader,
                num_epoch=3, scheduler=None, device='cuda', 
                needed_features=['input_ids', 'token_type_ids', 'attention_mask']):
    model.train()
    running_loss = 0.0
    num_batches = 0
    step = 0
    for batch in tqdm(train_loader):
        label = batch["label"].to(device)

        seed_everything(42)
        model_out = model(*[batch[fname].to(device) for fname in needed_features])

        batch_loss = loss_function(model_out, label)
        batch_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0
        )

        optimizer.step()
        optimizer.zero_grad()

        if scheduler:
            scheduler.step()

        running_loss += batch_loss.detach().cpu().numpy()
        num_batches += 1
    epoch_loss = running_loss / num_batches
    print("train loss: ", epoch_loss)


def val_epoch(model, loss_function, val_loader, device='cpu', target_names = None,
              needed_features=['input_ids', 'token_type_ids', 'attention_mask']):
    # target_names - список наименований категорий в one-hot представлении целевой переменной. Формат элементов - str.
    model.eval()
    running_loss = 0.0
    num_batches = 0
    model_prob = None
    true_label = None

    for batch in val_loader:
        label = batch["label"].to(device)
        true_class = batch["true_class"]
        model_out = model(*[batch[fname].to(device) for fname in needed_features])
        
        batch_prob = torch.softmax(model_out, dim=1).detach().cpu().numpy()
        batch_loss = loss_function(model_out, label)

        if model_prob is None:
            model_prob = batch_prob
            true_label = label.detach().cpu().numpy()
            true_class_model = true_class.detach().cpu().numpy()
        else:
            model_prob = np.vstack((model_prob, batch_prob))
            true_label = np.vstack((true_label, label.detach().cpu().numpy()))
            true_class_model = np.hstack((true_class_model, true_class.detach().cpu().numpy()))

        running_loss += batch_loss.detach().cpu().numpy()
        num_batches += 1
        
    epoch_loss = running_loss/num_batches
    print("val loss: ", epoch_loss)

    pred_idx = np.argmax(model_prob, axis=1)
    pred_label = np.array([int(target_names[i]) for i in pred_idx])  
    epoch_f1 = f1_score(true_class_model, pred_label, average='micro')
    
    return epoch_f1
    

def infer(model, val_loader, device='cpu',
          needed_features=['input_ids', 'token_type_ids', 'attention_mask']):
    model.eval()
    running_loss = 0.0
    num_batches = 0
    model_prob = None
    true_label = None
    for batch in tqdm(val_loader):
        model_out = model(*[batch[fname].to(device) for fname in needed_features])
        batch_prob = torch.softmax(model_out, dim=1).detach().cpu().numpy()
        
        if model_prob is None:
            model_prob = batch_prob
        else:
            model_prob = np.vstack((model_prob, batch_prob))

    return model_prob


def infer_single(model, val_loader, device='cpu',
                 needed_features=['input_ids', 'token_type_ids', 'attention_mask']):
    model.eval()
    running_loss = 0.0
    num_batches = 0
    model_prob = None
    true_label = None
    for batch in tqdm(val_loader):
        model_out = model(*[batch[fname].to(device) for fname in needed_features])["logits"]
        batch_prob = torch.softmax(model_out, dim=1).detach().cpu().numpy()
        
        if model_prob is None:
            model_prob = batch_prob
        else:
            model_prob = np.vstack((model_prob, batch_prob))

    return model_prob


def train_model(model, loss_function, optimizer, train_loader, val_loader, target_names = None,
                num_epoch=3, scheduler=None, device='cpu', name_to_save='model.pt',
                best_val_score=0.0, needed_features=['input_ids', 'token_type_ids', 'attention_mask']):
    # target_names - список наименований категорий в one-hot представлении целевой переменной. Формат элементов - str.
    for epoch in range(num_epoch):
        train_epoch(model, loss_function, optimizer, train_loader, val_loader, 
                    num_epoch, scheduler, device, needed_features)
        
        val_score = val_epoch(model, loss_function, val_loader, device, target_names,
                    needed_features)
        print(f"val f1_score after {epoch+1} epoch: {val_score}")

        if val_score > best_val_score:
            best_val_score = val_score
            print("new best score! Saving model...")
            torch.save(model, name_to_save)
        print("*"*100)

