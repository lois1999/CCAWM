# Training script for experiments with SCAN dataset

import os
import json

import torch
import torch.nn as nn
import torch.optim as optim

from data import build_scan
from models.transformer import *
from test_optimize import test


def train(d_model, nhead, num_encoder_layers, num_decoder_layers, 
          dim_feedforward, dropout, learning_rate, num_epochs, num_runs=5):

    for run in range(num_runs):
        assert type(d_model) == int
        assert type(nhead) == int
        assert d_model % nhead == 0
        assert type(num_encoder_layers) == int
        assert type(num_decoder_layers) == int 
        assert type(dim_feedforward) == int
        assert type(num_epochs) == int 

        # CUDA
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        # Data
        SRC, TRG, train_data, dev_data, test_data = build_scan('simple_scan', # split
                                                            64, # batch size 
                                                            device)

        # vocab
        src_vocab_size = len(SRC.vocab.stoi)
        trg_vocab_size = len(TRG.vocab.stoi)
        pad_idx = SRC.vocab[SRC.pad_token]
        assert TRG.vocab[TRG.pad_token] == pad_idx

        # Model
        model = Transformer(src_vocab_size, trg_vocab_size, d_model,
                                nhead, num_encoder_layers,
                                num_decoder_layers, dim_feedforward,
                                dropout, pad_idx, device)
        #model.load_state_dict(torch.load(args.load_weights_from))
        model = model.to(device)
        model.train()

        # Loss function
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
        loss_fn = loss_fn.to(device)

        # Optimizer
        params = model.parameters()
        optimizer = optim.Adam(params, lr=learning_rate)

        # Setup things to record
        loss_data = [] # records losses
        train_accs = [] # records train accuracy
        dev_accs = [] # records development accuracy
        test_accs = [] # records test accuracy
        best_dev_acc = float('-inf') # best dev accuracy (for doing early stopping)

        # Training loop:
        accs = []
        
        for epoch in range(num_epochs):
            for iter,batch in enumerate(train_data):
                optimizer.zero_grad()
                out, attn_wts = model(batch.src,batch.trg)
                loss = loss_fn(out.view(-1,trg_vocab_size),(batch.trg).roll(-1, 0).view(-1))
                loss.backward()
                optimizer.step()
                # Record loss
                if iter % 20 == 0: # record loss every
                    loss_datapoint = loss.data.item()
                    print('Run:', run,
                        'Epoch:', epoch,
                        'Iter:', iter,
                        'Loss:', loss_datapoint)
                    loss_data.append(loss_datapoint)

            # Checkpoint
            if epoch % 1 == 0: # checkpoint every
                # Checkpoint on train data
                print("Checking training accuracy...")
                train_acc = test(train_data, model, pad_idx)
                print("Training accuracy is ", train_acc)
                train_accs.append(train_acc)

                # Checkpoint on development data
                print("Checking development accuracy...")
                dev_acc = test(dev_data, model, pad_idx)
                print("Development accuracy is ", dev_acc)
                dev_accs.append(dev_acc)

                # Save model weights
                if run == 0: #first run only
                    # changed by LOIS
                    if dev_acc[1] > best_dev_acc: # use dev to decide to save (loss or acc[1])
                        best_dev_acc = dev_acc[1]

        accs.append(dev_acc[1])
    return np.mean(accs)

def train_ints( d_model, nhead, num_encoder_layers, num_decoder_layers, 
            dim_feedforward, dropout, learning_rate, num_epochs, num_runs=5):
    
    nhead = int(nhead)
    d_model = int(d_model)

    if d_model % nhead != 0 or d_model % 2 != 0:
        for i in range(10):
            d_model += 2
            nhead += 1

            if d_model % nhead == 0 and d_model % 2 == 0:
                num_encoder_layers = int(num_encoder_layers)
                num_decoder_layers = int(num_decoder_layers)
                dim_feedforward = int(dim_feedforward)
                num_epochs = int(num_epochs)

                return train(d_model, nhead, num_encoder_layers, num_decoder_layers, 
                dim_feedforward, dropout, learning_rate, num_epochs, num_runs=5)

        return 0

    else:
        num_encoder_layers = int(num_encoder_layers)
        num_decoder_layers = int(num_decoder_layers)
        dim_feedforward = int(dim_feedforward)
        num_epochs = int(num_epochs)

        return train(d_model, nhead, num_encoder_layers, num_decoder_layers, 
                    dim_feedforward, dropout, learning_rate, num_epochs, num_runs=5)