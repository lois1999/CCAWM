# Utilities for dealing with SCAN dataset

import os
# CHANGE MADE BY LOIS: REMOVED LEGACY
from torchtext import data, datasets
from torchtext.data import Field, BucketIterator
from torchtext.datasets import TranslationDataset


def build_scan(split, batch_size, device):
    # Get paths and filenames of each partition of split
    if split == 'simple_scan':
        path = 'data/scan/simple/'
    elif split == 'addjump_scan':
        path = 'data/scan/addjump/'
    elif split == 'simple_nacs':
        path = 'data/nacs/simple/'
    elif split == 'addjump_nacs':
        path = 'data/nacs/addjump/'
    elif split == 'addleft_scan':
        path = 'data/scan/addleft/'
    elif split == 'addleft_nacs':
        path = 'data/nacs/addleft/'
    elif split == 'length_scan':
        path = 'data/scan/length/'
    elif split == 'length_nacs':
        path = 'data/nacs/length/'
    elif split == 'addleft_removedleft_scan':
        path = 'data/scan/addleft_removedleft/'
    elif split == 'addleft_removedleft_nacs':
        path = 'data/nacs/addleft_removedleft/'
    elif split == 'addx_scan':
        path = 'data/scan/addx/'
    elif split == 'addx_nacs':
        path = 'data/nacs/addx/'
    elif split == 'churny_scan':
        path = 'data/scan/churny/'
    elif split == 'churny_nacs':
        path = 'data/nacs/churny/'
    elif split == 'split1_scan':
        path = 'data/scan/split1/'
    elif split == 'split2_scan':
        path = 'data/scan/split2/'
    elif split == 'jumpo_scan':
        path = 'data/scan/jumpo/'
    elif split == 'uniquesuffix_scan':
        path = 'data/scan/uniquesuffix/'
    elif split == 'split1_nacs':
        path = 'data/nacs/split1/'
    elif split == 'split2_nacs':
        path = 'data/nacs/split2/'
    elif split == 'jumpo_nacs':
        path = 'data/nacs/jumpo/'
    elif split == 'uniquesuffix_nacs':
        path = 'data/nacs/uniquesuffix/'
    elif split == 'verbarg_nacs':
        path = 'data/nacs/verbarg/'
    elif split == 'verbarg_scan':
        path = 'data/scan/verbarg/'
    elif split == 'simplesplit1_scan':
        path = 'data/scan/simplesplit1/'
    else:
        assert split not in ['simple_scan','addjump_scan',
        'simple_nacs','addjump_nacs', 'addleft_scan', 'length_nacs', 'length_scan', 'split1_scan', 'split2_scan', 'jumpo_scan', 'uniquesuffix_scan', 'split1_nacs', 'split2_nacs', 'jumpo_nacs', 'uniquesuffix_nacs', 'verbarg_scan', 'verbarg_nacs', 'simplesplit1_scan'], "Unknown split"
    train_path = os.path.join(path,'train')
    dev_path = os.path.join(path,'dev')
    test_path = os.path.join(path,'test')
    exts = ('.SRC','.TRG')

    # Fields for source (SRC) and target (TRG) sequences
    SRC = Field(init_token='<sos>',eos_token='<eos>')
    TRG = Field(init_token='<sos>',eos_token='<eos>')
    fields = (SRC,TRG)

    # Build datasets
    train_ = TranslationDataset(train_path,exts,fields)
    dev_ = TranslationDataset(dev_path,exts,fields)
    test_ = TranslationDataset(test_path,exts,fields)

    # Build vocabs: fields ensure same vocab used for each partition
    SRC.build_vocab(train_)
    TRG.build_vocab(train_)

    # BucketIterator ensures similar sequence lengths to minimize padding
    train, dev, test = BucketIterator.splits((train_, dev_, test_),
        batch_size = batch_size, device = device)

    return SRC, TRG, train, dev, test
