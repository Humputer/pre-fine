import pickle
from torch.utils.data import Dataset
import random
import cogdl
import torch


class SubgraphDataset(Dataset):

    def __init__(self, data_set):
        self.data_set = data_set
        
        self.edges = torch.load('../dataset/weibo/edges.pt')
        self.features = torch.tensor(torch.load('../dataset/weibo/user_tweet_feature_50_m.pt')).float()
        self.graph = cogdl.data.Graph(edge_index=self.edges.t(), x=self.features)
        
        self.map_list = pickle.loads(open('../dataset/weibo/subgraph_30.pickle', 'rb').read())

    def __len__(self):
        return len(self.data_set)
    
    def __getitem__(self, idx):
        return self.data_set[idx], self.graph.subgraph(
                self.map_list[
                    (int(self.data_set[idx][0]), int(self.data_set[idx][2]))
                    ]
                )
