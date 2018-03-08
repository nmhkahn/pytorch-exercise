import os
import argparse
from solver import Solver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=1000)
    
    parser.add_argument("--ckpt_dir", type=str, default="checkpoint")
    parser.add_argument("--ckpt_name", type=str, default="sr")
    parser.add_argument("--print_every", type=int, default=50)
    
    parser.add_argument("--result_dir", type=str, default="result")
    parser.add_argument("--train_dir", type=str, default="data/flower/train")
    parser.add_argument("--test_dir", type=str, default="data/flower/test")

    args = parser.parse_args()
    solver = Solver(args)
    solver.fit()

if __name__ == "__main__":
    main()
