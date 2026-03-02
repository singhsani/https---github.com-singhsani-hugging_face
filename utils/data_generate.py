import torch


def getData():
    torch.manual_seed(42)
    n_feature=4
    n_sample= 1000
    bais=0.2
    m=torch.tensor([[3.0],[4.0],[5.0],[3.5]])
    x_train=torch.rand(n_sample,n_feature)*10

    y_train= x_train @ m + bais + torch.rand(n_sample,1)*0.5

    return x_train,y_train