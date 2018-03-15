import os
import argparse
from solver import Solver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=20)
    
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--max_vocab", type=int, default=-1)
    
    parser.add_argument("--ckpt_dir", type=str, default="checkpoint")
    parser.add_argument("--ckpt_name", type=str, default="sst")
    parser.add_argument("--print_every", type=int, default=1)
    
    parser.add_argument("--result_dir", type=str, default="result")

    args = parser.parse_args()
    solver = Solver(args)
    solver.fit()

if __name__ == "__main__":
    main()
