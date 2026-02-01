from SELFRec import SELFRec
from util.conf import ModelConf

import torch
import numpy as np
import random

def set_seed():
    torch.manual_seed(1203)
    np.random.seed(2048)

if __name__ == '__main__':
    # Register your model here
    graph_baselines = ['LightGCN','DirectAU','MF','GraphAU']
    ssl_graph_models = ['SGL', 'SimGCL', 'SEPT', 'MHCN', 'BUIR', 'SelfCF', 'SSL4Rec', 'XSimGCL', 'NCL', 'MixGCF', 'LightGCL','DGCF','DirectAU1', 'SFA','Test','ICL', 'uCTRL','CPTPP','pop', 'PAAC', 'prototype','swav','spec','byol','NCL','AUPlus','share','swav2','swav3', 'DDAU', 'Construction','pretrain']
    sequential_baselines= ['SASRec']
    ssl_sequential_models = ['CL4SRec','DuoRec','BERT4Rec']
    torch.manual_seed(1203)
    np.random.seed(1203)
    random.seed(1203)
    torch.cuda.manual_seed_all(1203)
    torch.cuda.manual_seed(1203)
    print('=' * 80)
    print('   SELFRec: A library for self-supervised recommendasition.   ')
    print('=' * 80)
    print('Graph-Based Baseline Models:')
    print('   '.join(graph_baselines))
    print('-' * 100)
    print('Self-Supervised  Graph-Based Models:')
    print('   '.join(ssl_graph_models))
    print('=' * 80)
    print('Sequential Baseline Models:')
    print('   '.join(sequential_baselines))
    print('-' * 100)
    print('Self-Supervised Sequential Models:')
    print('   '.join(ssl_sequential_models))
    print('=' * 80)
    model = input('Please enter the model you want to run:')
    import time

    # set_seed()
    s = time.time()
    if model in graph_baselines or model in ssl_graph_models or model in sequential_baselines or model in ssl_sequential_models:
        conf = ModelConf('./conf/' + model + '.conf')
    else:
        print('Wrong model name!')
        exit(-1)
    rec = SELFRec(conf)
    rec.execute()
    e = time.time()

    print("Running time: %f s" % (e - s))