#
import torch
import torch.nn as nn
import torch.optim as optim
from utils.data_generate import getData

def learningData():

    # input features and data Sample
    n_feature= int(input("Enter features : "))
    n_sample= int(input ("Enter sample data : "))
    bais= float(input("Enter Bais: "))
    x_train, y_train = getData(n_feature,n_sample,bais)
    model = nn.Linear(n_feature,1)
    #print(x_train)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001)
    prediction = model(x_train)
    loss = criterion(prediction, y_train)
   # print(prediction,loss)

    for epoch in range(n_sample):
        prediction = model(x_train)
        loss = criterion(prediction, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}")

    print("Learned Weights:")
    print(model.weight.data)

    print("Learned Bias:")
    print(model.bias.data)
    
    return model
