# Function for testing models

import numpy as np
from sklearn.compose import TransformedTargetRegressor
import torch

def test(data, model, pad_idx):
    model.eval()
    with torch.no_grad():
        all_correct_trials = [] # list of booleans indicating whether correct
        all_correct_words = []
        for batch in data:
            out, attn_wts = model(batch.src, batch.trg)
            preds = torch.argmax(out,dim=2).roll(1,0)
            
            for i, item in enumerate(batch.trg):
                for j, word in enumerate(item):
                    word_target = word
                    if word_target == 1: # dont take into account padding
                        continue
                    else:
                        word_prediction = preds[i][j]
                        correct = (word_target == word_prediction)
                        all_correct_words.append(correct)

            correct_pred = preds == batch.trg
            correct_pred = correct_pred.cpu().numpy()
            mask = batch.trg == pad_idx # mask out padding
            mask = mask.cpu().numpy()
            correct = np.logical_or(mask,correct_pred)
            # CHANGED TO 1 by LOIS: SHOULD COUNT ALL TRUES AND NOT ALL FALSES 
            correct = correct.all(1).tolist()
            all_correct_trials += correct
    
    accuracy = np.mean(all_correct_trials)
    overlap = np.mean(all_correct_words)
    model.train()

    return accuracy, overlap
