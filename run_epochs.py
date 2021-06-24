import torch
import random
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from utils import on_device

def epoch_train(networks,optimizers,num_models,num_preds,trainloader):

    criterion = nn.CrossEntropyLoss()
    for j in range(num_models):
        networks[j]=networks[j].train()

    for data in trainloader:

        assgn_ls = []

        for _ in range(num_models):
            assgn_ls.append([])

        inputs, labels = data

        sample_size = inputs.size()[0]

        inputs,labels = on_device(inputs,labels,num_models)

        for op in optimizers:
            op.zero_grad()

        logits_list = [networks[j](inputs[j]) for j in range(num_models)]
        
        for b in range(sample_size):

            with torch.no_grad():

                loss_ls = [criterion(torch.unsqueeze(logits_list[j][b,:],dim=0), torch.unsqueeze(labels[j][b],dim=0)) for j in range(num_models)]
            
            _, min_index_ls = torch.topk(-(torch.tensor(loss_ls)),num_preds)

            for index in min_index_ls:
          
                assgn_ls[index].append(b)

        for m,assgn in enumerate(assgn_ls):
            if(len(assgn)!=0):

                if(len(assgn)>1):
                    loss = criterion(logits_list[m][assgn,:], labels[m][assgn])
                else:
                    
                    loss = criterion(torch.unsqueeze(logits_list[m][assgn[0],:],dim=0), torch.unsqueeze(labels[m][assgn[0]],dim=0))
                
                loss.backward()
                optimizers[m].step()




def epoch_val(networks,num_models,testloader):

    criterion = nn.CrossEntropyLoss()
    for j in range(num_models):
        networks[j]=networks[j].eval()

    correct = 0
    total = 0
    val_loss = 0.0
    

    with torch.no_grad():
        for data in testloader:

            inputs, labels = data

            inputs = inputs.cuda()
            labels = labels.cuda()

            logits_list = [networks[j](inputs) for j in range(num_models)]
            loss_ls = [criterion(logits_list[j],labels) for j in range(num_models)]
            preds_ls = [torch.max(logits_list[j].data, 1)[1].item() for j in range(num_models)]


            value = torch.min(torch.tensor(loss_ls))

            val_loss = val_loss + value.item()

            if(labels.item() in preds_ls):
                correct = correct + 1

            total += labels.size(0)

        return [(100 * (correct / total)), (val_loss/total)]