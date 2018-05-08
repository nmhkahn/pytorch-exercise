import os
import argparse
from solver import Solver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=50)
    
    parser.add_argument("--max_vocab", type=int, default=10000)
    parser.add_argument("--embed_dim", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--attn_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=1)

    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--image_size", type=int, default=224)
    
    parser.add_argument("--ckpt_dir", type=str, default="checkpoint")
    parser.add_argument("--ckpt_name", type=str, default="caption")
    parser.add_argument("--print_every", type=int, default=1000) # 1k step not epoch
    
    args = parser.parse_args()
    solver = Solver(args)
    solver.fit()

if __name__ == "__main__":
    main()
