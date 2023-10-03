import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_linear_schedule_with_warmup

import numpy as np
# import wandb
from cogdl.data.batch import batch_graphs
import random
import os
from tqdm import tqdm
from grace_model import Grace
from aug import aug
from tensorboardX import SummaryWriter

from model import Model, WeightedLoss
from data import SubgraphDataset
from finetune_utils import get_args, EarlyStopping
import argparse

from sklearn.metrics import precision_recall_fscore_support, auc
import sklearn.metrics
import warnings
warnings.filterwarnings("ignore",  category=UserWarning)
# os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
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

def train(model, p_model, optimizer, lr_scheduler, epoch, train_loader, device):
    model.train()
    p_model.train()
    running_t_loss = 0
    print('Epoch '+str(epoch)+':')
    for data_batch, graph_batch in tqdm(train_loader, ncols=80):

        graph_batch = graph_batch.to(args.device)

        p_feature = p_model.module.encoder(graph_batch, graph_batch.x)
        label = data_batch[:, 3].unsqueeze(1).float()

        sample = torch.stack((data_batch[:, 0], data_batch[:, 1], data_batch[:, 2]), dim=1)
        sample = sample.to(device)
        label = label.to(device)

        scores = model(sample, graph_batch, p_feature)

        t_loss = loss_fn(scores, label)        
        t_loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        running_t_loss += t_loss.item()

    if dist.get_rank() == 0:
        print("Epoch {} complete, train loss: {} ".format(epoch, running_t_loss/len(train_loader)))
    return running_t_loss/len(train_loader)
        

def evaluate(model, p_model, epoch, eval_loader, device):

    model.eval()
    p_model.eval()
    running_t_loss = 0
    print('Epoch '+str(epoch)+':')
    with torch.no_grad():
        for data_batch, graph_batch in tqdm(eval_loader, ncols=80):
            graph_batch = graph_batch.to(args.device)

            p_feature = p_model.module.encoder(graph_batch, graph_batch.x)
            label = data_batch[:, 3].unsqueeze(1).float()
            sample = torch.stack((data_batch[:, 0], data_batch[:, 1], data_batch[:, 2]), dim=1)
            sample = sample.to(device)
            label = label.to(device)

            scores = model(sample, graph_batch, p_feature)
            t_loss = loss_fn(scores, label)        

            running_t_loss += t_loss.item()

        if dist.get_rank() == 0:
            print("Evaluation {} complete, train loss: {} ".format(epoch, running_t_loss/len(eval_loader)))
        return running_t_loss/len(eval_loader)


def test(model, p_model, epoch, test_loader, device):
    model.eval()
    p_model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for data_batch, graph_batch in tqdm(test_loader, ncols=80):
            graph_batch = graph_batch.to(args.device)

            p_feature = p_model.module.encoder(graph_batch, graph_batch.x)
            label = data_batch[:, 3].unsqueeze(1).float()

            sample = torch.stack((data_batch[:, 0], data_batch[:, 1], data_batch[:, 2]), dim=1)
            sample = sample.to(device)
            label = label.to(device)

            scores = model(sample, graph_batch, p_feature)  

            graph_batch = graph_batch.to(args.device)

            y_true += data_batch[:, 3].data.tolist()
            
            y_pred += scores.squeeze(1).data.tolist()
    for i in range(len(y_true)):
        if y_pred[i] >= 0.5:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
            
    FPR, TPR, thresholds = sklearn.metrics.roc_curve(y_true, y_pred)
    Auc = auc(FPR, TPR)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    print("Test complete, AUC: {}, Precision: {}, Recall: {}, F1: {},".format(Auc, prec, rec, f1))
    return Auc, prec, rec, f1


def custom_collate_fn(batches):
    samples = []
    graphs = []
    for batch in batches:
        sample, graph = batch
        # 在这里添加预处理步骤，例如图像调整大小
        samples.append(torch.tensor(sample))
        graphs.append(graph)
    # 在这里返回处理后的数据集
    return torch.stack(samples), batch_graphs(graphs)


if __name__=='__main__':
    args = get_args()

    device = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(device)
    dist.init_process_group(backend='nccl')  

    # # use wandb
    # if dist.get_rank() == 0:
    #     wandb.init()
    # # # use sweep
    # hps = [None]
    # if dist.get_rank() == 0:
    #     args = wandb.config
    #     hps = [dict(args)]
    # dist.broadcast_object_list(hps, src=0)
    # args = argparse.Namespace(**hps[0])

    args.device = device
    
    seed = 0
    for seed in range(3):
        args.seed = seed
        print(args)
        set_seed(args.seed)

        model = Model(args=args)
        model = model.to(device)
        model = DDP(model, device_ids=[device], output_device=device)
        # loss_fn = WeightedLoss()
        loss_fn = torch.nn.BCELoss()
        
        p_model = Grace(
                        in_dim=args.num_features, 
                        hid_dim=args.num_hidden,
                        out_dim=args.num_hidden,
                        act_fn='relu',
                        temp=1,
                        num_layers=args.num_layers,
                        drop_edge_rate=0,
                        drop_node_rate=0
                        )
        
        p_model.load_state_dict(torch.load("/data/wangbojie/www/pretrainV2/gat_30/seed=0_lr=0.002_bs=64_dpr=0.3_temp=0.3_layers=1_der=0.4_dnr=0.2/pretrained_model.pt"))
        p_model = p_model.to(device)
        p_model = DDP(p_model, device_ids=[device], output_device=device)

        optimizer = torch.optim.Adam([{'params':model.parameters()}, {'params':p_model.parameters()}], lr=args.learning_rate)

        log_dir = 'reply/wu={}_seed={}_lr={}_bs={}_dpr={}_layers={}_/'.format(args.warmup, args.seed, args.learning_rate, args.batch_size, args.drop_rate, args.num_layers)

        if dist.get_rank() == 0:
            writer = SummaryWriter(log_dir=log_dir)

        best_auc, best_prec, best_rec, best_f1 = 0, 0, 0, 0

        train_data_set = torch.load('../dataset/'+args.dataset+'/train_new.pt')
        eval_data_set = torch.load('../dataset/'+args.dataset+'/eval_new.pt')
        test_data_set = torch.load('../dataset/'+args.dataset+'/test_new.pt')

        train_set = SubgraphDataset(train_data_set, args.dataset)
        eval_set = SubgraphDataset(eval_data_set, args.dataset)
        test_set = SubgraphDataset(test_data_set, args.dataset)

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, seed=seed)
        eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_set, seed=seed)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4, shuffle=False, sampler=train_sampler, collate_fn=custom_collate_fn)
        eval_loader = DataLoader(eval_set, batch_size=args.batch_size, num_workers=4, shuffle=False, sampler=eval_sampler, collate_fn=custom_collate_fn)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=4, collate_fn=custom_collate_fn)

        t_total = len(train_loader)*args.epochs  # Necessary to take into account Gradient accumulation
        num_warmup_steps = t_total//args.warmup # The number of steps for the warmup phase.
        lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)
        # lr_scheduler = None
        # early_stopping = EarlyStopping(logdir=log_dir, patience=5, verbose=True)
        for i in range(args.epochs):

            train_sampler.set_epoch(i)
            eval_sampler.set_epoch(i)

            t_t_loss = train(model, p_model, optimizer, lr_scheduler, i, train_loader, device)
            e_t_loss = evaluate(model, p_model, i, eval_loader, device)

            if dist.get_rank() == 0:
                Auc, prec, rec, f1 = test(model, p_model, i, test_loader, device)
                drawer(t_t_loss, i, 'train_loss')
                drawer(e_t_loss, i, 'eval_loss')
                drawer(Auc, i, 'test_auc')
                drawer(prec, i, 'test_prec')
                drawer(rec, i, 'test_rec')
                drawer(f1, i, 'test_f1')
                # wandb.log({ 
                #         "train_loss": t_t_loss,
                #         "eval_loss": e_t_loss,                 
                #         'test_auc': Auc, 
                #         "test_prec": prec, 
                #         "test_rec": rec,
                #         "test_f1": f1
                #         })
                if best_auc <= Auc and best_f1 <= f1:
                    best_auc = Auc
                    best_prec = prec
                    best_rec = rec
                    best_f1 = f1
                    f = open(log_dir+'record.txt', 'w')
                    f.write('AUC: {}\nPrecision: {}\nRecall: {}\nF1: {}'.format(best_auc, best_prec, best_rec, best_f1))
                    f.close()
                # early_stopping(e_t_loss, model, p_model)
                # if early_stopping.early_stop:
                #     print("Early stopping")
                #     break
    # wandb.finish()

# torchrun --nproc_per_node 4 --master_port=29500 finetune.py
# wandb sweep --project=www --name=finetune_new finetune.yaml