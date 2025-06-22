# str1 ='  你好  吗\n'
# print(str1.strip())
import torch
import torch.nn as nn
# a-->[2, 2]-->dim=0
a = torch.tensor([[2.0, 1.0],
                  [2.0, 4.0]])
# e^2 = 7.389, e^1 = 2.718 --> 10.1
# l1 = nn.LogSoftmax(dim=-1)
# result = l1(a)
# print(result)

# nn.CrossEntropyLoss()  = nn.NLLLoss() + nn.LogSoftmax(dim=-1)
b = a.unsqueeze(dim=1)
print(b.shape)