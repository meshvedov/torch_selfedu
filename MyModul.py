import torch
import torch.optim as optim
import torch.nn.functional as F
from random import randint


class MyModule(torch.nn.Module):
    def __init__(self, input_dim, num_hidden, output_dim):
        super().__init__()
        self.layer1 = torch.nn.Linear(input_dim, num_hidden)
        self.layer2 = torch.nn.Linear(num_hidden, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = F.tanh(x)
        x = self.layer2(x)
        x = F.tanh(x)
        return x

model = MyModule(3,2,1)
print(list(model.parameters()))

x_train = torch.FloatTensor([(-1,-1,-1,), (-1, -1, 1), (-1,1,-1), (-1,1,1), (1,-1,-1), (1,-1,1), (1,1,-1), (1,1,1)])
y_train = torch.FloatTensor([-1, 1, -1, 1, -1, 1, -1, -1])
total = len(y_train)

optimizer = optim.RMSprop(params=model.parameters(), lr=.01)
loss_func = torch.nn.MSELoss()

model.train()
for _ in range(1000):
    k = randint(0, total - 1)
    y = model(x_train[k])
    loss = loss_func(y, y_train[k])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()