import argparse
import torch

class EarlyStopping:
    def __init__(self, logdir, patience=7, verbose=False, delta=0):
        """
        :param patience: How long to wait after last time validation loss improved.
        :param verbose: If True, prints a message for each validation loss improvement. 
        :param delta: Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.logdir = logdir
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model, p_model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, p_model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, p_model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, p_model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.module.state_dict(), self.logdir+'model_checkpoint.pt')
        # torch.save(p_model.module.state_dict(), self.logdir+'p_model_checkpoint.pt')
        self.val_loss_min = val_loss


def get_args():
    parser = argparse.ArgumentParser(description="GAT")
    # train model
    parser.add_argument('--device', type=int, default=0, help ='0,1,2,3')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--drop_rate', type=float, default=0.3, help='Dropout')
    # pretrain model 
    parser.add_argument("--num_layers", type=int, default=1, help="number of hidden layers")
    parser.add_argument("--num_hidden", type=int, default=128, help="number of hidden units")
    parser.add_argument("--num_features", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--dataset", type=str, default='Twibot-reply')

    args = parser.parse_args()
    return args
