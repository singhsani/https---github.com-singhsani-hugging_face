import torch

def test_by_user_data(model,X):
    with torch.no_grad():
        x_test = X
        predict = model(x_test)
    return predict
