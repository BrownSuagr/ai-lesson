# print('你好', end=' ')
# print('不好', end=' ')

# pytorch框架下保存模型：两种方式：

'''
第一种方式：(最常用)
只保存了模型的参数
torch.save(model.state_dict(), "model.pth")
# 加载过程
model = Model()
model.load_state_dict(torch.load("model.pth"))
'''

'''
第二种方式：（占据内存太大）
不仅保存模型的参数，还保存模型的架构
torch.save(model, "model.pth")
加载过程：
model = torch.load("model.pth")
'''



# import random
#
# print(random.randint(0, 1))

from transformers import BertModel
model = BertModel.from_pretrained("/Users/ligang/PycharmProjects/NLP/NLPBase/GZ_AI/dm06_transformers/model/bert-base-chinese")

print(model)












