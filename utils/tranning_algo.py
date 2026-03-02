#
import torch
import torch.nn as nn
import torch.optim as optim
from utils.data_generate import getData

def learningData():
    x_train, y_train = getData()
    model = nn.Linear(4,1)
    #print(x_train)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001)
    prediction = model(x_train)
    loss = criterion(prediction, y_train)
   # print(prediction,loss)

    for epoch in range(5000):
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

