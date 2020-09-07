import torch
import torch.nn as nn

x = torch.tensor([[1,2,3],[4,5,6],[7,8,9]], dtype=torch.float).requires_grad_()
y = torch.tensor([[2.8191, 4.3610, 3.7487],[3.8279, 12.9571, 10.9117],[4.8367, 21.5532, 18.0747]])

w1 = torch.tensor([0.5, 0.7, 0.9]).requires_grad_()
w2 = torch.tensor([0.125, 0.825, 0.951]).requires_grad_()
w3 = torch.tensor([-0.1, -0.3, 0.5]).requires_grad_()

lr = 1e-5

for t in range(5):
    #w = torch.stack((w1, w2, w3))
    #w.squeeze_(-1)
    #y_pred = x.mm(w)
    for _ in range(1):
        w = torch.stack((w1,w2,w3))
        w.squeeze_(-1)
        y_pred = x.mm(w)
        loss = (y_pred - y).pow(2).sum()
        loss.backward()
    with torch.no_grad():
        w1 -= lr * w1.grad
        print("w1 gradient is {}".format(w1.grad))
        w2 -= lr * w2.grad
        w3 -= lr * w3.grad
        w1.grad.zero_()
        w2.grad.zero_()
        w3.grad.zero_()
    
