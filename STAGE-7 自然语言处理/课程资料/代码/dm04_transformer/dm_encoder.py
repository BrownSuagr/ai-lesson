import copy
import math
import torch.nn.functional as F
import numpy as np
import torch

from dm_input import *

# 定义下三角矩阵（掩码张量）
def sub_mask(size):
    # 生成上三角矩阵（只有0和1两种值）
    up_vector = np.triu(m=np.ones((1, size, size)), k=1).astype('uint8')
    # 变成下三角
    return torch.from_numpy(1-up_vector)

# 定义函数：计算注意力：按照第三种注意力计算规则来实现的，

def attention(query, key, value, mask=None, dropout=None):
    # query, key, value分别来自输入部分，如果是自注意力：query=key=value;如果不是自注意力：query!=key=value
    # query, key, value形状一致：[2, 4, 512]
    # mask代表是否对注意力分数进行掩码，如果用到编码器第一层，那就是padding——mask,防止补齐的元素对注意力产生影响
    # 如果用到解码器第一层,sentence-mask,解码的时候，防止未来信息被提前利用
    # dropout随机失活的对象，防止过拟合
    # 1. 获取d_k,代表词嵌入维度
    d_k = query.size(-1)
    # 2. 计算注意力的分数:[2, 4, 512]*[2,512, 4]--->scores-->[2, 4, 4]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # print(f'scores-->{scores.shape}')
    # 3. 如果进行掩码，我们的掩码张量需要作用到注意力分数上
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)
    # 4. 对注意力分数进行softmax归一化，得到注意力权重

    p_atten = F.softmax(scores, dim=-1)
    # print(f'p_atten--》{p_atten}')
    # 5. 为了防止过拟合，我们需要对p_atten进行dropout
    if  dropout is not None:
        p_atten = dropout(p_atten)

    # 6. 计算出注意力:[2,4,4]*[2,4,512]-->【2,4, 512】
    atten1 = torch.matmul(p_atten, value)
    return atten1, p_atten

def test_attention():
    # 调用函数获取：输入部分
    position_x = test_positionEncoding()
    # 准备数据
    query = key = value = position_x
    atten1, p_atten = attention(query, key, value)
    mask = torch.zeros(2, 4, 4)
    print('注意力结果', atten1.shape)
    print('注意力权重', p_atten.shape)
    atten2, p_atten2 = attention(query, key, value, mask=mask)
    print('da注意力结果', atten2.shape)
    print('da注意力权重', p_atten2.shape)

def clones(module, N):
    # 克隆N个神经网络的某一层
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# 定义多头注意力
class MutiHeadAttention(nn.Module):
    def __init__(self, head, embed_dim, dropout_p=0.1):
        super().__init__()
        # head代表指定多少个头；embed_dim代表词嵌入维度；dropout_p代表随机失活的系数
        # 1.判断词嵌入维度是否能够整除head
        assert embed_dim % head == 0
        # 2. 获得每个头的词嵌入表示维度:embed_dim=512,head=8,--> d_k=64
        self.d_k = embed_dim // head
        # 3. 指定head的属性
        self.head = head
        # 4. 定义4个全连接层
        self.linears = clones(nn.Linear(embed_dim, embed_dim), 4)
        # 5.定义atten的属性
        self.atten = None
        # 6. 定义Dropout层
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, query, key, value, mask=None):
        # query;key;value都来自输入部分，如果是自注意力：他们相等，否则不等
        # 按照课堂上的例子 ：query, key, value形状都是【2，4，512】
        # mask由三维变成四维-->[8,4,4]-->[1, 8, 4, 4]
        if mask is not None:
            mask = mask.unsqueeze(0)

        # 获取batch_size
        batch_size = query.size(0)
        # 获取进行多头注意力运算的Q\K\V
        # model(x)-->【2，4，512】
        # model(x).view(batch_size, -1, self.head, self.d_k)-->【2，4，8，64】
        # model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2) -->【2，8，4，64】
        # 转置的原因：习惯性【seq_len, embed_dim】-->这样做，能够特征更加充分
        query, key, value = [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
                             for model, x in zip(self.linears, (query, key, value))]
        # query, key, value的形状都是--》[2,8,4,64]
        # 计算注意力：需要调用attention的方法
        # x--》[2,8,4,64]; atten_weight-->[2,8,4,4]
        x, atten_weight = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 需要对上一步计算的注意力结果进行融合 新的x-->[2,4,512]
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head*self.d_k)
        return self.linears[-1](x)


def test_MutiheadAtten():
    my_atten = MutiHeadAttention(head=8, embed_dim=512)
    position_x = test_positionEncoding()
    query = key = value = position_x

    mask = torch.zeros(8, 4, 4) #-->【head, seq_len, seq_len】
    result = my_atten(query, key, value, mask)
    print(f'最终多头自注意力机制的结果：{result.shape}')
    print(result)


# 定义前馈全连接层
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_p=0.1):
        super().__init__()
        # d_model代表词嵌入维度；d_ff代表中间表示维度
        # 定义第一个全连接层
        self.linear1 = nn.Linear(d_model, d_ff)
        # 定义第二个全连接层
        self.linear2 = nn.Linear(d_ff, d_model)
        # 定义dropout层
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


def test_ff():
    # 获得输入部分：wordEmbedding+positionEncoding
    position_x = test_positionEncoding()
    # 获得多头注意力的结果
    my_atten = MutiHeadAttention(head=8, embed_dim=512)
    query = key = value = position_x
    mask = torch.zeros(8, 4, 4)  # -->【head, seq_len, seq_len】
    atten_result = my_atten(query, key, value, mask)
    print(f'最终多头自注意力机制的结果：{atten_result.shape}')
    # 获取前馈全连接层的结果
    my_ff = FeedForward(d_model=512, d_ff=1024)
    output = my_ff(atten_result)
    print(f'前馈全连接层获得的结果--》output:{output.size()}')

# 定义规范化层
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        # features--》代表词嵌入维度
        # 定义系数a，会随着模型的训练进行参数的更改
        self.a = nn.Parameter(torch.ones(features))
        # 定义偏置b
        self.b = nn.Parameter(torch.zeros(features))

        # 定义常量eps:防止分母为0
        self.eps = eps

    def forward(self, x):
        # x-->【2，4，512】
        # 1. 求出均值
        x_mean = torch.mean(x, dim=-1, keepdim=True)
        # 2. 求出标准差
        x_std = torch.std(x, dim=-1, keepdim=True)
        # 3. 计算规范化值
        y = self.a * (x - x_mean) / (x_std + self.eps) + self.b
        return y


def test_layerNorm():
    # 获得输入部分：wordEmbedding+positionEncoding
    position_x = test_positionEncoding()
    # 获得多头注意力的结果
    my_atten = MutiHeadAttention(head=8, embed_dim=512)
    query = key = value = position_x
    mask = torch.zeros(8, 4, 4)  # -->【head, seq_len, seq_len】
    atten_result = my_atten(query, key, value, mask)
    print(f'最终多头自注意力机制的结果：{atten_result.shape}')
    # 获取前馈全连接层的结果
    my_ff = FeedForward(d_model=512, d_ff=1024)
    ff_output = my_ff(atten_result)
    print(f'前馈全连接层获得的结果--》ff_output:{ff_output.size()}')
    # 这里是假设ff_output就是规范化层的输入
    # 实例化规范化层对象
    my_layernorm = LayerNorm(features=512)
    result = my_layernorm(ff_output)
    print(f'规范化后的结果--》result:{result.size()}')
    print(result)


# 定义子层连接结构
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout_p=0.1):
        # size:代表词嵌入维度
        super().__init__()
        # 实例化标准化层
        self.norm = LayerNorm(size)
        # 定义随机失活层
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x, sublayer):
        # x代表原始的输入，sublayer代表函数的对象
        # 方式1
        # result = x + self.dropout(self.norm(sublayer(x)))
        # 方式2
        result = x + self.dropout(sublayer(self.norm(x)))
        return result

def test_Sublayer():
    # 1.准备数据
    x = torch.tensor([[2, 4, 10, 20], [4, 8, 32, 20]], dtype=torch.long)
    # 2. 将x送入embedding层：
    vocab_size = 1000 # 假设一共有1000个单词
    d_model = 512 # 假设词嵌入维度是512
    my_embed = MyEmbedding(vocab_size, d_model)
    x_embed = my_embed(x)
    print(f'词嵌入之后的结果--》{x_embed.shape}')
    # 3.需要将x_embed送入PE位置编码器，添加位置信息
    dropout_p = 0.1 # 随机失活的系数
    my_pe = PositionEncoding(d_model, dropout_p)
    x_pe = my_pe(x_embed)
    print(f'经过位置编码之后的结果--》{x_pe.shape}')
    # 4. 实例化多头注意力层（自注意力）
    head = 8
    embed_dim = d_model
    self_atten = MutiHeadAttention(head, embed_dim)
    mask = torch.zeros(8, 4, 4)
    # 定义一个匿名函数：返回的就是多头自注意力的结果
    sublayer = lambda x: self_atten(x, x, x, mask=mask)
    # 5. 实现子层连接结构的应用
    my_sub_connect = SublayerConnection(size=512)
    result = my_sub_connect(x_pe, sublayer)
    print(f'经过第一个子层连接结构的结果--》{result.shape}')

# 定义编码器层
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout_p=0.1):
        super().__init__()
        # self_attn：代表实例化多头自注意力对象
        self.self_attn = self_attn
        # feed_forward:代表实例化前馈全连接层的对象
        self.feed_forward = feed_forward
        # size:词嵌入维度
        self.size = size
        # 定义两层子层连接结构
        self.sublayers = clones(SublayerConnection(size, dropout_p), 2)

    def forward(self, x, mask):
        # 经过第一个子层连接结构;self.sublayers[0]代表第一个子层连接结构的对象
        x1 = self.sublayers[0](x=x, sublayer=lambda x: self.self_attn(x, x, x, mask))
        # 经过第二个子层连接结构
        x2 = self.sublayers[1](x=x1, sublayer=self.feed_forward)
        return x2

def test_encoderlayer():
    # 1.准备数据
    x = torch.tensor([[2, 4, 10, 20], [4, 8, 32, 20]], dtype=torch.long)
    # 2. 将x送入embedding层：
    vocab_size = 1000 # 假设一共有1000个单词
    d_model = 512 # 假设词嵌入维度是512
    my_embed = MyEmbedding(vocab_size, d_model)
    x_embed = my_embed(x)
    print(f'词嵌入之后的结果--》{x_embed.shape}')
    # 3.需要将x_embed送入PE位置编码器，添加位置信息
    dropout_p = 0.1 # 随机失活的系数
    my_pe = PositionEncoding(d_model, dropout_p)
    x_pe = my_pe(x_embed)
    print(f'经过位置编码之后的结果--》{x_pe.shape}')
    mask = torch.zeros(8, 4, 4)
    # 4. 实例化多头注意力层（自注意力）
    head = 8
    embed_dim = d_model
    self_atten = MutiHeadAttention(head, embed_dim)
    # 5. 实例化前馈全连接层
    feed_forward = FeedForward(d_model=512, d_ff=1024)
    # 6.实例化编码器层
    my_encoderlayer = EncoderLayer(size=512, self_attn=self_atten, feed_forward=feed_forward)
    output = my_encoderlayer(x_pe, mask)
    print(f'一个编码器层得到的结果---》{output.shape}')

# 定义编码器
class Encoder(nn.Module):
    def __init__(self, layer, N):
        # layer代表：编码器层的对象；N代表由N个编码器层
        super().__init__()
        # 克隆N个编码器层
        self.layers = clones(layer, N)
        # 定义规范化层：作用在最后一个编码器层的输出结果上
        self.norm = LayerNorm(features=layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

def test_encoder():
    # 1.准备数据
    x = torch.tensor([[2, 4, 10, 20], [4, 8, 32, 20]], dtype=torch.long)
    # 2. 将x送入embedding层：
    vocab_size = 1000 # 假设一共有1000个单词
    d_model = 512 # 假设词嵌入维度是512
    my_embed = MyEmbedding(vocab_size, d_model)
    x_embed = my_embed(x)
    print(f'词嵌入之后的结果--》{x_embed.shape}')
    # 3.需要将x_embed送入PE位置编码器，添加位置信息
    dropout_p = 0.1 # 随机失活的系数
    my_pe = PositionEncoding(d_model, dropout_p)
    x_pe = my_pe(x_embed)
    print(f'经过位置编码之后的结果--》{x_pe.shape}')
    mask = torch.zeros(8, 4, 4)
    # 4. 实例化多头注意力层（自注意力）
    head = 8
    embed_dim = d_model
    self_atten = MutiHeadAttention(head, embed_dim)
    # 5. 实例化前馈全连接层
    feed_forward = FeedForward(d_model=512, d_ff=1024)
    # 6.实例化编码器层
    my_encoderlayer = EncoderLayer(size=512, self_attn=self_atten, feed_forward=feed_forward)

    # 7.实例化编码器
    my_encoder = Encoder(layer=my_encoderlayer, N=6)
    encoder_output = my_encoder(x_pe, mask)
    # print(f'encoder_output--》{encoder_output}')
    print(f'encoder_output编码器输出结果--》{encoder_output.shape}')
    return encoder_output


if __name__ == '__main__':
    # test_attention()
    # test_MutiheadAtten()
    # test_ff()
    # test_layerNorm()
    # test_Sublayer()
    # test_encoderlayer()
    test_encoder()