import torch
import torch.nn as nn
import torch.optim as optim

def train_single_model(x_train, y_train, epochs=1000, lr=0.0001):

    n_feature = x_train.shape[1]
    
    model = nn.Linear(n_feature, 1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        prediction = model(x_train)
        loss = criterion(prediction, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")

    print("Learned Weight:", model.weight.data)
    print("Learned Bias:", model.bias.data)

    return model


def train_multi_output_model(x_train, y_train, output_size, epochs=1000, lr=0.0001):

    n_feature = x_train.shape[1]
    
    model = nn.Linear(n_feature, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        prediction = model(x_train)
        loss = criterion(prediction, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model