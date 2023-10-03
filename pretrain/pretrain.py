import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
from cogdl.data.batch import batch_graphs
import random
import os
from tqdm import tqdm
from grace_model import Grace
import argparse
# import wandb
from tensorboardX import SummaryWriter

from model import Model, WeightedLoss
from data import SubgraphDataset
from pretrain_utils import get_args

from sklearn.metrics import precision_recall_fscore_support, auc
import sklearn.metrics
import warnings
warnings.filterwarnings("ignore",  category=UserWarning)
# os.environ["CUDA_VISIBLE_DEVICES"] = '4, 5, 6, 7'
# os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2, 3, 4, 5, 6, 7'

def set_seed(seed):  
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def drawer(value, epoch, label):
    if label == 'train_loss':
        writer.add_scalar('train/train_loss', value, epoch)
    elif label == 'pre_train_loss':
        writer.add_scalar('train/pre_train_loss', value, epoch)
    elif label == 'all_train_loss':
        writer.add_scalar('train/all_train_loss', value, epoch)
    elif label == 'eval_loss':
        writer.add_scalar('eval/eval_loss', value, epoch)
    elif label == 'eval_pre_loss':
        writer.add_scalar('eval/eval_pre_loss', value, epoch)
    elif label == 'all_eval_loss':
        writer.add_scalar('eval/all_eval_loss', value, epoch)
    elif label == 'test_auc':
        writer.add_scalar('test/auc', value, epoch)
    elif label == 'test_prec':
        writer.add_scalar('test/prec', value, epoch)
    elif label == 'test_rec':
        writer.add_scalar('test/rec', value, epoch)
    elif label == 'test_f1':
        writer.add_scalar('test/f1', value, epoch)

def train(model, p_model, optimizer, epoch, train_loader, device):

    p_model.train()
    running_p_loss = 0
    print('Epoch '+str(epoch)+':')
    for data_batch, graph_batch in tqdm(train_loader, ncols=80):

        graph_batch = graph_batch.to(args.device)

        p_loss = p_model(graph_batch, graph_batch.x)

        p_loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        running_p_loss += p_loss.item()

    if dist.get_rank() == 0:
        print("Epoch {} complete, pre-train loss: {} ".format(epoch, running_p_loss/len(train_loader)))
    return running_p_loss/len(train_loader)


def evaluate(model, p_model, epoch, eval_loader, device):

    p_model.eval()
    running_p_loss = 0
    print('Epoch '+str(epoch)+':')
    with torch.no_grad():
        for data_batch, graph_batch in tqdm(eval_loader, ncols=80):

            graph_batch = graph_batch.to(args.device)

            p_loss = p_model(graph_batch, graph_batch.x)

            running_p_loss += p_loss.item()

        if dist.get_rank() == 0:
            print("Evaluation {} complete, pre-train loss: {}".format(epoch, running_p_loss/len(eval_loader)))
        return running_p_loss/len(eval_loader)


def custom_collate_fn(batches):
    samples = []
    graphs = []
    for batch in batches:
        sample, graph = batch

        samples.append(torch.tensor(sample))
        graphs.append(graph)

    return torch.stack(samples), batch_graphs(graphs)


if __name__=='__main__':
    args = get_args()

    device = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(device)
    dist.init_process_group(backend='nccl')  

    # # use wandb
    # if dist.get_rank() == 0:
    #     wandb.init()
    # # use sweep
    # hps = [None]
    # if dist.get_rank() == 0:
    #     args = wandb.config
    #     hps = [dict(args)]
    # dist.broadcast_object_list(hps, src=0)
    # args = argparse.Namespace(**hps[0])

    args.device = device
    
    seed = 0
    args.seed = seed
    print(args)
    set_seed(args.seed)

    model = None
    
    p_model = Grace(in_dim=args.num_features, 
                    hid_dim=args.num_hidden,
                    out_dim=args.num_hidden,
                    act_fn='relu',
                    temp=args.temperature,
                    num_layers=args.num_layers,
                    drop_edge_rate=args.der,
                    drop_node_rate=args.dnr
                    )
    p_model = p_model.to(device)
    p_model = DDP(p_model, device_ids=[device], output_device=device)

    optimizer = torch.optim.Adam(p_model.parameters(), lr=args.learning_rate)

    log_dir = 'weibo_model/seed={}_lr={}_bs={}_dpr={}_temp={}_layers={}_der={}_dnr={}/'.format(args.seed, args.learning_rate, args.batch_size, args.drop_rate, args.temperature, args.num_layers, args.der, args.dnr)

    if dist.get_rank() == 0:
        writer = SummaryWriter(log_dir=log_dir)

    best_auc, best_prec, best_rec, best_f1 = 0, 0, 0, 0

    train_data_set = torch.load('../dataset/weibo/train_new.pt')
    eval_data_set = torch.load('../dataset/weibo/eval_new.pt')
    test_data_set = torch.load('../dataset/weibo/test_new.pt')

    train_data_set = torch.cat((train_data_set, eval_data_set), dim=0)

    train_set = SubgraphDataset(train_data_set)
    eval_set = SubgraphDataset(test_data_set)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, seed=seed)
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_set, seed=seed)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4, shuffle=False, sampler=train_sampler, collate_fn=custom_collate_fn)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, num_workers=4, shuffle=False, sampler=eval_sampler, collate_fn=custom_collate_fn)
    t_loss = 20
    for i in range(args.epochs):

        train_sampler.set_epoch(i)
        eval_sampler.set_epoch(i)
        t_p_loss = train(model, p_model, optimizer, i, train_loader, device)
        e_p_loss = evaluate(model, p_model, i, eval_loader, device)
        if dist.get_rank() == 0:
            drawer(t_p_loss, i, 'pre_train_loss')
            drawer(e_p_loss, i, 'pre_eval_loss')
            # wandb.log({ 
            #         "pre_train_loss": t_p_loss,
            #         "pre_eval_loss": e_p_loss
            #         })
            if t_loss > e_p_loss:
                t_loss = e_p_loss
                torch.save(p_model.module.state_dict(), log_dir+'pretrained_model.pt')
    # wandb.finish()

# torchrun --nproc_per_node 7 --master_port=29500 pretrain.py
# wandb sweep --project=www --name=pretrain_new pretrain.yaml