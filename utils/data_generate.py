import torch


def getData(n_feature,n_sample,bais):
    torch.manual_seed(42)
    m= 2.0 #torch.tensor([[3.0],[4.0],[5.0],[3.5]])
    x_train=torch.rand(n_sample,n_feature)*10

    y_train= x_train * m + bais + torch.rand(n_sample,1)*0.5

    return x_train,y_train