# import torch
# # # m，n  和 n, k -->【m, k】
# # a = torch.tensor([[2, 4, 0],
# #                   [1, 2, 1]])
# # b = torch.tensor([[2, 1],
# #                   [0, 2],
# #                   [2, 4]])
# # c = a @ b
# # print('c',c)
# # # 矩阵运算方法：torch.matmul() # 万能
# # d = torch.matmul(a, b)
# # print('d', d)
# # # 矩阵运算方法：torch.mm() # 只支持二维矩阵运算
# # e = torch.mm(a, b)
# # print('e', e)
#
# #  矩阵运算方法：torch.bmm() # 只支持三维矩阵运算
#
# x1 = torch.randn(2, 3, 4)
# # print(x1)
# x2 = torch.randn(2, 4, 5)
# # print(x2)
#
# # y1 = torch.matmul(x1, x2)
# # # print('y1', y1)
# # print('y1', y1.shape)
#
# y2 = torch.bmm(x1, x2)
# # print('y2', y2)
# print('y2', y2.shape)
#
#
#
# import re
# s = " I Love? 我 爱；you!  "
# s1 = s.lower().strip()
# print(s1)
# s2 = re.sub(r"([?.!])", r' \1', s1)
# print(s2)
# s3 = re.sub(r"[^a-zA-Z.!?]+", ' ', s2)
# print(s3)
def fun():
    return 1, 2
a = fun()
print(a)
