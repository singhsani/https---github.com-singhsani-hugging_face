import pandas as pd
import torch

def data():
    df=pd.read_excel("D:\python\dataSet\converted_data.xlsx")
    df.to_excel("data.xlsx", index=False)
    X = df.iloc[:, 1].values
    Y = df.iloc[:, 2].values
    X = torch.tensor(X, dtype=torch.float32).view(-1,1)
    Y = torch.tensor(Y, dtype=torch.float32).view(-1,1)

    print(X.shape)  # should be (30,1)
    print(Y.shape)  # should be (30,1)
    #print("x data",X)
    print("y data",Y)
    return X,Y
