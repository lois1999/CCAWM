# Main script for gathering args, running train

import argparse
from collections import defaultdict
import itertools
from numba import jit, cuda

from train import train

# Needed for parameter search
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Needed for parameter search
def makeGrid(pars_dict):  
    keys=pars_dict.keys()
    combinations=itertools.product(*pars_dict.values())
    ds=[dict(zip(keys,cc)) for cc in combinations]
    return ds

parser = argparse.ArgumentParser()
# Training data
parser.add_argument('--split',
                    choices = ['simple_scan','addjump_scan','simple_nacs','addjump_nacs',
                     'addleft_scan', 'addleft_nacs', 'length_scan', 'length_nacs'
                     , 'addleft_removedleft_scan', 'addleft_removedleft_nacs'
                     , 'addx_scan', 'addx_nacs', 'churny_scan', 'churny_nacs', 'split1_scan', 
                     'split2_scan', 'jumpo_scan', 'uniquesuffix_scan', 'split1_nacs', 
                     'split2_nacs', 'jumpo_nacs', 'uniquesuffix_nacs'],
                    help='SCAN split to use for training and testing')
parser.add_argument('--num_runs', type=int, default=10,
                    help='Number of runs to do.')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Samples per batch')
parser.add_argument('--num_epochs', type=int, default=2,
                    help='Number of training epochs')

# Models
parser.add_argument('--model_type', choices=['transformer'],
                    default='transformer', help='Type of seq2seq model to use.')
parser.add_argument('--d_model', type=int, default=12,
                    help='Dimension of inputs/outputs in transformer')
parser.add_argument('--nhead', type=int, default=2,
                    help='Number of heads in transformer multihead attention')
parser.add_argument('--num_encoder_layers', type=int, default=2,
                    help='Number of layers in transformer encoder')
parser.add_argument('--num_decoder_layers', type=int, default=2,
                    help='Number of layers in transformer decoder')
parser.add_argument('--dim_feedforward', type=int, default=20,
                    help='Dimension of feedforward layers in transformer')
parser.add_argument('--dropout', type=float, default=0.08,
                    help='Dropout rate')
parser.add_argument('--load_weights_from', default=None,
                    help='Path to saved weights')

# Optimization
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Fixed learning rate for Adam optimizer')

# Output options
parser.add_argument('--results_dir', default='transformer',
                    help='Results subdirectory to save results')
parser.add_argument('--out_data_file', default='train_results.json',
                    help='Name of output data file with training loss data')
parser.add_argument('--checkpoint_path',default=None,
                    help='Path to output saved weights.')
parser.add_argument('--checkpoint_every', type=int, default=1,
                    help='Epochs before evaluating model and saving weights')
parser.add_argument('--record_loss_every', type=int, default=20,
                    help='iters before printing and recording loss')

@jit(target_backend ="cuda")      
def main(args):
    for run in range(args.num_runs):
        return train(run,args)

if __name__ == '__main__':

    params = {"--dim_feedforward": [32, 64, 128, 256],
    "--dropout": [0.01, 0.04, 0.08, 0.2, 0.3],
    "--learning_rate": [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]}

    params2 = {"--batch_size": [64, 128, 256], 
    "--num_encoder_layers": [2, 4, 8], 
    "--num_decoder_layers": [2, 4, 8],
    "--dim_feedforward": [15, 20, 25, 30, 35],
    "--learning_rate": [0.001, 0.003, 0.005]}

    grid = makeGrid(params)
    print(len(grid))
    accs = defaultdict()

    for search in grid:
        print(search)
        args = Namespace(batch_size = 64,
        d_model = 12, 
        nhead = 2, 
        num_encoder_layers = 2, 
        num_decoder_layers = 2,
        dim_feedforward = search["--dim_feedforward"],
        dropout = search["--dropout"],
        learning_rate = search["--learning_rate"],
        num_runs = 1,
        split = "simple_scan", 
        model_type = "transformer",
        load_weights_from = None,
        num_epochs = 40,
        record_loss_every = 20, 
        checkpoint_every = 1,
        results_dir = "transformer",
        out_data_file = "simple_scan_search",
        checkpoint_path = None)
        acc = main(args)
        accs[args] = acc
    print(accs)
    #args = parser.parse_args()

    #main(args)
