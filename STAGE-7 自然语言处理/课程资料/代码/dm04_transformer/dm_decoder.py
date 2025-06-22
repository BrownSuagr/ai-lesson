import torch

from dm_input import *
from dm_encoder import *

# 定义解码器层
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        # size：代表词嵌入维度
        self.size = size
        # self_attn：代表多头自注意力机制的对象：Q=K=V
        self.self_attn = self_attn
        # src_attn：代表多头注意力机制的对象：Q!=K=V
        self.src_attn = src_attn
        # feed_forward：代表前馈全连接层对象
        self.feed_forward = feed_forward
        # 定义三个子层连接对象：使用clones(module, N)方法
        self.sublayers = clones(SublayerConnection(size, dropout), 3)

    def forward(self, y, memory, source_mask, target_mask):
        # y: 代表来自解码器的输入
        # memory：代表编码器的输出结果
        # source_mask：代表解码器第二个子层，进行注意力计算的掩码：对padding mask
        # target_mask：代表解码器第一个子层，进行自注意力计算的掩码：sentences mask
        # 1. 数据经过第一个子层连接结构
        y1 = self.sublayers[0](x=y, sublayer=lambda x: self.self_attn(x, x, x, target_mask))
        # 2. 数据经过第二个子层连接结构
        y2 = self.sublayers[1](x=y1, sublayer=lambda x: self.src_attn(x, memory, memory, source_mask))
        # 3. 数据经过第三个子层连接结构
        y3 = self.sublayers[2](x=y2, sublayer=self.feed_forward)
        return y3

# 测试解码器层
def test_decoderlayer():

    size = 512
    # 多头注意力机制的对象
    head = 8
    embed_dim = 512
    # 实例化多头自注意力机制对象
    self_attn = MutiHeadAttention(head, embed_dim)
    # 实例化多头注意力机制对象
    src_attn = MutiHeadAttention(head, embed_dim)
    # 实例化前馈全连接层对象
    d_model = 512
    d_ff = 1024
    feed_forward = FeedForward(d_model, d_ff)
    # 定义随机失活的系数
    dropout = 0.1
    # 实例化DecoderLayer对象
    my_decoderlayer = DecoderLayer(size, self_attn, src_attn, feed_forward, dropout)
    # 定义解码器的输入y
    y = torch.tensor([[20, 40, 1, 5], [8, 90, 18, 24]], dtype=torch.long)
    # 需要将y经过embedding+positionEncoding
    vocab_size = 1000
    # 得到embedding的结果
    my_embed = MyEmbedding(vocab_size, d_model)
    embed_y = my_embed(y)
    # 将embed_y进行位置编码
    my_position = PositionEncoding(d_model, dropout)
    position_y = my_position(embed_y)
    # 获得编码器的输入结果
    encoder_output = test_encoder()
    # 假设source_mask, target_mask相等都是0（实际上不一样，只不过这里是举例而已）
    source_mask = target_mask = torch.zeros(8, 4, 4)
    output = my_decoderlayer(y=position_y, memory=encoder_output, source_mask=source_mask, target_mask=target_mask)
    print(f'解码器层的输出--》{output.shape}')
    print(f'解码器层的输出--》{output}')


# 定义解码器
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        # layer：代表解码器层对象；N代表多少个解码器层
        # 定义N个解码器层
        self.layers = clones(layer, N)
        # 定义规范化层
        self.norm = LayerNorm(layer.size)

    def forward(self, y, memory, source_mask, target_mask):
        # y: 代表来自解码器的输入
        # memory：代表编码器的输出结果
        # source_mask：代表解码器第二个子层，进行注意力计算的掩码：对padding mask
        # target_mask：代表解码器第一个子层，进行自注意力计算的掩码：sentences mask
        # 循环N次，得到最终的编码器输出结果（最后输出需要经过norm）
        for layer in self.layers:
            y = layer(y, memory, source_mask, target_mask)
        return self.norm(y)

# 测试解码器
def test_decoder():
    size = 512
    # 多头注意力机制的对象
    head = 8
    embed_dim = 512
    d_model = 512
    d_ff = 1024
    # 定义随机失活的系数
    dropout = 0.1
    # 定义解码器的输入y
    y = torch.tensor([[20, 40, 1, 5], [8, 90, 18, 24]], dtype=torch.long)
    # 需要将y经过embedding+positionEncoding
    vocab_size = 1000
    # 得到embedding的结果
    my_embed = MyEmbedding(vocab_size, d_model)
    embed_y = my_embed(y)
    # 将embed_y进行位置编码
    my_position = PositionEncoding(d_model, dropout)
    # 得到解码器的输入部分
    position_y = my_position(embed_y)
    # 实例化多头自注意力机制对象
    self_attn = MutiHeadAttention(head, embed_dim)
    # 实例化多头注意力机制对象
    src_attn = MutiHeadAttention(head, embed_dim)
    # 实例化前馈全连接层对象
    feed_forward = FeedForward(d_model, d_ff)
    # 实例化DecoderLayer对象
    my_decoderlayer = DecoderLayer(size, self_attn, src_attn, feed_forward, dropout)
    # 获得编码器的输入结果
    encoder_output = test_encoder() # memory
    # 假设source_mask, target_mask相等都是0（实际上不一样，只不过这里是举例而已）
    source_mask = target_mask = torch.zeros(8, 4, 4)

    # 实例化解码器对象
    my_decoder= Decoder(layer=my_decoderlayer, N=6)
    decoder_output = my_decoder(position_y, encoder_output, source_mask, target_mask)
    print(f'解码器的输出-->{decoder_output.shape}')
    print(f'解码器的输出-->{decoder_output}')

    return decoder_output
if __name__ == '__main__':
    # test_decoderlayer()
    test_decoder()