"""This file processes arguments and configs"""

import argparse
import logging
import sys

import torch


###############################
# Read arguments from CLI
###############################
parser = argparse.ArgumentParser()

parser.add_argument("--working_place", choices=['local', 'remote', 'local2'],
                    help="working environment", default='remote')
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_proc", type=int, default=4, help="number of processes")
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--bert_model", type=str, default="bert-base-uncased",
                    help="default pretrain BERT model")
parser.add_argument("--task", type=str, help="Select task to do")
parser.add_argument("--n_shards", type=int, help="Number of chunks to split from large dataset",
                    default=8)
parser.add_argument("--device", type=str, default="default", choices=["default", "cpu", "cuda"],
                    help="Select device to run")
parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
parser.add_argument("--w_decay", type=float, default=0, help="Weight decay")

# args = parser.parse_args()
args, _ = parser.parse_known_args()


###############################
# Add several necessary argument
###############################
## args = working_place
if args.working_place == "local":
    args.init_path  = "/Users/hoangle/Projects/VinAI"
elif args.working_place == "remote":
    args.init_path  = "/home/ubuntu"
elif args.working_place == "local2":
    args.init_path = "/home/tommy/Projects/VinAI"
else:
    print("Err: Invalid init path. Exiting..")
    sys.exit(1)


## args = device
if args.device == "default":
    args.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    args.device = torch.device(args.device)

## other args
args.multi_gpus = torch.cuda.device_count() > 0


###############################
# Config logging
###############################
logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%b-%d-%Y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

###############################
# Config path of backup files
###############################
PATH    = {
    'bert_model'    : f"{args.init_path}/_pretrained/BERT/{args.bert_model}",


    ## Paths associated with Data Reading
    'dataset_para'  : "./backup/dataset_paras_[SPLIT]/"
}
