import torch.nn as nn
import torch
# # embedding = nn.Embedding(10, 3)
# # input = torch.LongTensor([[1,2,4,5],
# #                           [4,3,2,9]])
# # output = embedding(input)
# # print(output)
# # print(f'output--》{output.shape}')
# # embedding = nn.Embedding(10, 3, padding_idx=1)
# # input = torch.tensor([[1, 4, 3, 1, 0, 0, 0]], dtype=torch.long)
# # output = embedding(input)
# # print(output)
# # a = torch.arange(0, 60,).unsqueeze(1)
# # print(a)
# # b = torch.randn(10, 4, 5)
# # print(b.size(0))
# # c = a*b
# # print(c.shape)
# import numpy as np
# # a = np.arange(100)
# # print(a)
# #
# # print(np.triu([[1, 1, 1, 1, 1],
# #                [2, 2, 2, 2, 2],
# #                [3, 3, 3, 3, 3],
# #                [4, 4, 4, 4, 4],
# #                [5, 5, 5, 5, 5]], k=1))
# #
# # print(np.triu([[1, 1, 1, 1, 1],
# #                [2, 2, 2, 2, 2],
# #                [3, 3, 3, 3, 3],
# #                [4, 4, 4, 4, 4],
# #                [5, 5, 5, 5, 5]], k=0))
# #
# # print(np.triu([[1, 1, 1, 1, 1],
# #                [2, 2, 2, 2, 2],
# #                [3, 3, 3, 3, 3],
# #                [4, 4, 4, 4, 4],
# #                [5, 5, 5, 5, 5]], k=-1))
#
# # a = torch.randn(3, 4)
# # print(a)
# # b = np.triu(m=np.ones((3, 4)), k=1).astype('uint8')
# # mask = torch.from_numpy(1-b)
# # print(mask)
# #
# # d = a.masked_fill(mask==0, 1e-9)
# # print(f'd,{d}')
# import copy
# scores = torch.tensor([[1, 2, 0],
#                        [2, 4, 0],
#                        [0, 0, 0]], dtype=torch.float)
# mask = torch.tensor([[1, 2, 0],
#                     [2, 4, 0],
#                     [0, 0, 0]], dtype=torch.int)
#
# mask[mask!=0] = 1
# #
# print('mask', mask)
# # # print(a)
# c = scores.masked_fill(mask==0, -1e9)
# print(c)
# #
# # a = torch.randn(2, 3)
# # print(a)
# # print(a.transpose(0,1))
# # print(a.transpose(1,0))

# def fun(x):
#     assert x==3
#     print('你真棒')
#
# fun(3)
# a,b,c = [1, 2, 3]
# print(a)
# print(b)
# print(c)

# a = (1, 2, 3, 7)
# b = [5, 4, 6]
# print(list(zip(a, b)))



# for x, y in zip(a, b):
#     print(x, y)

# query = torch.randn(2, 8, 4, 64)
# key = torch.randn(2, 8, 64, 4)
# scores = torch.matmul(query, key)
# print(scores.shape)

# a = torch.tensor([[[3, 4, 3],
#                   [3, 0, 7]]], dtype=torch.float)
#
# c = torch.tensor([2, 2, 2])
# print(a*c)

# c = a.mean(dim=1, keepdim=True)
# print(c)
# d = torch.mean(a, dim=1, keepdim=True)
# print(d)
# # e = torch.std(a, dim=1, keepdim=True)
# # print(e)
#
# sublayer = lambda x: x+x+x+3
#
# def forward(x, sublayer):
#     return sublayer(x)
#
# result = forward(x=2, sublayer=sublayer)
# print(result)
# torch.manual_seed(1)
# class SequentialModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # self.linear1 = nn.Linear(5, 6)
#         # self.linear2 = nn.Linear(6, 2)
#         self.a = nn.Sequential(nn.Linear(5, 6),
#                                nn.Linear(6, 2))
#
#
#     def forward(self, x):
#         # x1 = self.linear1(x)
#         # x2 = self.linear2(x1)
#         x2 = self.a(x)
#         return x2
#
# if __name__ == '__main__':
#     my_sm = SequentialModel()
#     x = torch.randn(2, 4, 5)
#     print(my_sm(x))

model = nn.Transformer()
print(model)


