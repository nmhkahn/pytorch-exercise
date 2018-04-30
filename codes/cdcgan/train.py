import os
import argparse
from solver import Solver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--z_dim", type=int, default=100)
    
    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument("--result_dir", type=str, default="./result")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoint")
    parser.add_argument("--data_root", type=str, default="./data")

    args = parser.parse_args()
    solver = Solver(args)
    solver.fit()

if __name__ == "__main__":
    main()
