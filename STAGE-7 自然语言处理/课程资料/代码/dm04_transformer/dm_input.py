# coding:utf-8
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np


# 定义词嵌入层
class MyEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        # vocab_size-->编码器输入数据所有单词的总个数(需要进行embedding词的个数)
        self.vocab_size = vocab_size
        # d_model-->词嵌入维度
        self.d_model = d_model
        # 定义词嵌入层
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # 在这里乘以根号下d_model的原因：
        # 1.因为词嵌入向量后续会和position位置信息相加，为了防止position数值过大，对齐产生影响，所以增大数据
        # 2.让数据符合标准正太分布
        return self.embed(x) * math.sqrt(self.d_model)


# 定义位置编码器
class PositionEncoding(nn.Module):
    def __init__(self, d_model, dropout_p, max_len=600):
        super().__init__()
        # d_model:词嵌入维度；dropout_p:随机失活的系数；max_len：最大句子长度
        # 定义随机失活层
        self.dropout = nn.Dropout(p=dropout_p)
        # 定义位置编码矩阵（一开始是全零） # [60, 512]
        pe = torch.zeros(max_len, d_model)
        # 定义位置矩阵position-->[60, 1]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 定义中间变化矩阵
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0))/d_model)
        # position和div_term进行相乘，得到sin(x)x--》频率 my_vec=[60, 256]
        my_vec = position*div_term
        # print(f'my_vec--》{my_vec.shape}')
        # 对PE的奇数位置进行sin函数表示，偶数位用cos函数
        pe[:, 0::2] = torch.sin(my_vec)
        pe[:, 1::2] = torch.cos(my_vec)
        # pe进行升维-->【1,60，512】
        pe = pe.unsqueeze(0)
        # 将pe注册到模型的缓存区，可以让模型加载其对应的参数，但是不更新
        self.register_buffer('pe', pe)
        # print('11', self.pe.requires_grad)

    def forward(self, x):
        # x-->代表词嵌入之后的结果：【2，4，512】
        # self.pe-->【1，60，512】
        # 返回的结果是包含了位置编码信息的词向量
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

def test_positionEncoding():
    # 1.得到embedding的结果
    my_embed = MyEmbedding(vocab_size=1000, d_model=512)
    x = torch.tensor([[100, 2, 421, 508],
                      [491, 999, 1, 221]], dtype=torch.long)
    embed_x = my_embed(x)
    print(f'embedding之后的结果-->{embed_x.shape}')
    # 2.对embedding的结果加上位置编码信息
    my_position = PositionEncoding(d_model=512, dropout_p=0.1)
    position_x = my_position(embed_x)
    print(f'embedding+positiobn的结果-->{position_x.shape}')
    return position_x

# 定义函数：绘图：位置编码信息

def show_pe():
     my_position = PositionEncoding(d_model=20, dropout_p=0)
     embed_x = torch.zeros(1, 100, 20)
     position_x = my_position(embed_x)
     # 画图
     plt.figure()
     plt.plot(np.arange(100), position_x[0, :, 4:8])
     plt.legend(["dim_%s" % i for i in [4, 5, 6, 7]])
     plt.show()




if __name__ == '__main__':
    # test_positionEncoding()
    show_pe()