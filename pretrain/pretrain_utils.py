import argparse


def get_args():
    parser = argparse.ArgumentParser(description="GAT")
    # train model
    parser.add_argument('--device', type=int, default=0, help ='0,1,2,3')
    parser.add_argument('--learning_rate', type=float, default=2e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--drop_rate', type=float, default=0.3, help='Dropout')
    # pretrain model 
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature.")
    parser.add_argument("--num_layers", type=int, default=1, help="number of hidden layers")
    parser.add_argument("--num_hidden", type=int, default=128, help="number of hidden units")
    parser.add_argument("--num_features", type=int, default=1024)
    parser.add_argument("--der", type=float, default=0.4)
    parser.add_argument("--dnr", type=float, default=0.4)

    args = parser.parse_args()
    return args
