# # str1 ='  你好  吗\n'
# # print(str1.strip())
# # import torch
# # import torch.nn as nn
# # # a-->[2, 2]-->dim=0
# # a = torch.tensor([[2.0, 1.0],
# #                   [2.0, 4.0]])
# # # e^2 = 7.389, e^1 = 2.718 --> 10.1
# # # l1 = nn.LogSoftmax(dim=-1)
# # # result = l1(a)
# # # print(result)
# #
# # # nn.CrossEntropyLoss()  = nn.NLLLoss() + nn.LogSoftmax(dim=-1)
# # b = a.unsqueeze(dim=1)
# # print(b.shape)
# import json
# # dict1 = {"name":"lg", 'age':20, "job":"teacher"}
#
# # with open('a.json', mode='w', encoding='utf-8') as fw:
# #     str1 = json.dumps(dict1)
# #     print(type(str1))
# #     fw.write(str1)
#
#
#
# with open('./a.json', mode='r', encoding='utf-8') as fr:
#     line = fr.readline()
# #
# # dict2 = eval(line)
# # print(dict2)
# # print(type(dict2))
# dict1 = json.loads(line)
# print(dict1)
# print(type(dict1))
# # print(dict1["total_loss_list"])
# # print(dict1["all_time"])
# # print(dict1["total_acc_list"])
import torch
a = torch.tensor([[1.2, 2.3, 3.4, 4.5, 7.8, 9.2],
                 [0.5, 0.2, 9.2, 2.9, 3.4, 5.6]])
topv, topi = torch.topk(a, k=1, dim=-1)
print(f'topv-->{topv}')
print(f'topi-->{topi}')