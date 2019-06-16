
import argparse

import numpy as np

from bilm.training import train, load_options_latest_checkpoint, load_vocab
from bilm.data import BidirectionalLMDataset
import os


def main(args):
    max_token_length = int(args.max_token_length)

    # load the vocab
    # vocab = load_vocab(args.vocab_file, 50) 
    print(args.stroke_vocab_file)
    vocab = load_vocab(args.vocab_file, 
                       args.stroke_vocab_file, # Winfred stroke_vocab
                       max_token_length) # Winfred stroke_vocab

    # define the options
    batch_size = 128  # batch size for each GPU
    n_gpus = 1

    # number of tokens in training data (this for 1B Word Benchmark)
    n_train_tokens = 10731134 # 768648884

    # options = {
    #  'bidirectional': True,

    #  'char_cnn': {'activation': 'relu',
    #   'embedding': {'dim': 16},
    #   'filters': [[1, 32],
    #    [2, 32],
    #    [3, 64],
    #    [4, 128],
    #    [5, 256],
    #    [6, 512],
    #    [7, 1024]],
    #   'max_characters_per_token': max_token_length,
    #   'n_characters': 266, # 原261 + 筆畫5
    #   'n_highway': 2}, # 2
    
    #  'dropout': 0.1,
    
    #  'lstm': {
    #   'cell_clip': 3,
    #   'dim': 4096,
    #   'n_layers': 2,
    #   'proj_clip': 3,
    #   'projection_dim': 512,
    #   'use_skip_connections': True},
    
    #  'all_clip_norm_val': 10.0,
    
    #  'n_epochs': 1,
    #  'n_train_tokens': n_train_tokens,
    #  'batch_size': batch_size,
    #  'n_tokens_vocab': vocab.size,
    #  'unroll_steps': 20,
    #  'n_negative_samples_batch': 8192,
    # }
    
    # Add by Winfred
    option_file = os.path.join(args.save_dir, "options.json")
    with open(option_file, "r") as f:
        options = json.load(f)
    # End

    prefix = args.train_prefix
    data = BidirectionalLMDataset(prefix, vocab, test=False,
                                      shuffle_on_load=False, # True
                                      do_record=args.do_record,       # Add by Winfred
                                      records_path=args.records_path, # Add by Winfred
                                      vocab_file=args.vocab_file)     # Add by Winfred

    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir
    train(options, data, n_gpus, tf_save_dir, tf_log_dir,
          restart_ckpt_file=args.restart_ckpt_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', help='Vocabulary file')
    parser.add_argument('--train_prefix', help='Prefix for train files')
    parser.add_argument('--stroke_vocab_file', help='')
    parser.add_argument('--max_token_length', help='')
    parser.add_argument('--restart_ckpt_file', help='')
    parser.add_argument('--do_record', help='')
    parser.add_argument('--records_path', help='')

    args = parser.parse_args()
    main(args)

