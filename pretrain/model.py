import torch.nn as nn
import torch
import pickle


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.device = args.device
        pretrained_tweet_embedding = torch.load('../dataset/weibo/tweet_embedding_m.pt')
        self.tweet_embedding = nn.Embedding(pretrained_tweet_embedding.size(0), pretrained_tweet_embedding.size(1))
        self.tweet_embedding.weight = nn.Parameter(pretrained_tweet_embedding)
        self.tweet_embedding.weight.requires_grad = False

        pretrained_description_embedding = torch.load('../dataset/weibo/node_features_m.pt')
        self.description_embedding = nn.Embedding(pretrained_description_embedding.size(0), pretrained_description_embedding.size(1))
        self.description_embedding.weight = nn.Parameter(pretrained_description_embedding)
        self.description_embedding.weight.requires_grad = False

        # self.degree_dict = pickle.load(open('../dataset/degree_dict.pickle', 'rb'))
        # self.degree_embedding = nn.Embedding(695363, args.num_hidden)


        # pretrained_graph_embedding = torch.tensor(torch.load('dataset/graph_embed.pt')).float()
        # self.graph_embedding = nn.Embedding(pretrained_graph_embedding.size(0), pretrained_graph_embedding.size(1))
        # self.graph_embedding.weight = nn.Parameter(pretrained_graph_embedding)
        # self.graph_embedding.weight.requires_grad = False

        self.t_linear = nn.Linear(1024, args.num_hidden)

        self.mlp = nn.Sequential(
            nn.Linear(args.num_hidden*3, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(args.drop_rate),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        

    def forward(self, sample, graph, p_feature):

        u_feat = torch.tensor([]).to(self.device)
        v_feat = torch.tensor([]).to(self.device)

        for i in range(min(128, len(sample))):
            u_idx = torch.nonzero(torch.eq(graph.batch, torch.tensor(i)))[0]
            v_idx = u_idx+1
            # u_feat = torch.cat((u_feat, p_feature[u_idx]+self.degree_embedding(self.degree_dict[int(sample[i][0])].to(self.device))))
            # v_feat = torch.cat((v_feat, p_feature[v_idx]+self.degree_embedding(self.degree_dict[int(sample[i][2])].to(self.device))))
            u_feat = torch.cat((u_feat, p_feature[u_idx]))
            v_feat = torch.cat((v_feat, p_feature[v_idx]))

        input = torch.cat((u_feat+self.t_linear(self.description_embedding(sample[:, 0])), 
                           self.t_linear(self.tweet_embedding(sample[:, 1])), 
                           v_feat+self.t_linear(self.description_embedding(sample[:, 2]))), 
                           dim=1)

        return torch.sigmoid(self.mlp(input))

    
class WeightedLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.55, reduction='mean'):
        super(WeightedLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = predict # sigmoide获取概率
        #在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = - self.alpha * target * torch.log(pt) - (1 - self.alpha) * (1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss
