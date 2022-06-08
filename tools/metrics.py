""" Eval metrics and related

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from numpy.random import random
from sklearn.metrics import recall_score,precision_score,f1_score


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


jks=0
jk2=0
gh=1
def accuracy(output, target, ops, topk=(1,),):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))


    y_pre=pred[0].cpu().numpy().tolist()
    targetssr=target.cpu().numpy().tolist()
    #y1=pred[1]
    #y_true=torch.zeros(96,dtype=torch.int32)
    #prec, rec, f1, _ = precision_recall_fscore_support(targetssr, y_pre, average="binary")

    # print(prec)
    # print(rec)
    # print(f1)
    # print(prec2)
    # print(rec2)
    # print(f12)
    # print("cv")
    # res = []
    # global jks
    # for k in topk:
    #     correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
    #     cnm1=correct[:k]
    #     cnm2=cnm1.reshape(-1)
    #     cnm3=cnm2.float()
    #     res.append(correct_k.mul_(100.0 / batch_size))
    
    #print("cv")

    # fox=96*(ops-1)
    # res = []
    # global jks
    # for k in topk:
    #     correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
    #     cnm1=correct[:k]
    #     cnm2=cnm1.reshape(-1)
    #     cnm3=cnm2.float()
    #     res.append(correct_k.mul_(100.0 / batch_size))
    #     v=1
    #     if k <=1:
    #         print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    #         for j in cnm3:
    #             if j == 0:
    #                 # print(v+fox)
    #                 # print("**********")
    #                 fox2=v+fox
    #                 if fox2>390:
    #                     print(fox2+3883)
    #                 else:
    #                     if fox2==312:
    #                         print(v)
    #                         print(fox)
    #                         print("NOw i get you")
    #                     print(fox2)
    #                 jks=jks+1
    #             v += 1
    #         print("$$$$$$$$$$$$$$$$$$$$$$$$$$")
    #
    # print(res)
    # print("all mistakes are ",jks)
    global jk2
    global gh
    fox=[]
    
    cv=0

    for j in output:

        if  jk2>=384 :
            if  cv>4:
                gh=0
            if  gh==0:

                fox.append(j[1].cpu().numpy().tolist())
            if  gh==1:

                 fox.append(j[0].cpu().numpy().tolist())

        else:
            fox.append(j[0].cpu().numpy().tolist())
        cv+=1

    jk2 += 96
    acc1,acc5=[correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]
    return acc1,acc5,targetssr,y_pre,fox

def get_num():
    return jks