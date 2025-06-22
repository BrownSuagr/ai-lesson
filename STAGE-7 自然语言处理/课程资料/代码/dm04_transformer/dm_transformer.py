import torch

from dm_output import *
# 定义EncoderDecoder架构（transformer）
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        # encoder：代表编码器的对象
        self.encoder = encoder
        # decoder：代表解码器的对象
        self.decoder = decoder
        # src_embed：代表编码器输入部分的（词嵌入层+位置编码器层）对象————>embedding
        self.src_embed = src_embed
        # tgt_embed：代表解码器输入部分的（词嵌入层+位置编码器层）对象-----》embedding
        self.tgt_embed = tgt_embed
        # generator：代表输出部分：输出层对象
        self.generator = generator

    def forward(self, source, target, source_mask, target_mask):
        # source代表编码器原始的输入--》shape-->[2,4]--代表：2个样本4个单词
        # target代表解码器原始的输入--》shape-->[2,4]--代表：2个样本4个单词
        # source_mask 代表padding_mask; target_mask代表sentences_mask
        # 第一步：将原始的输入source，送入（词嵌入层+位置编码器层）对象得到--》word_Embedding；编码器输入部分
        # src_embed_x-->[2, 4, 512]
        src_embed_x = self.src_embed(source)
        # 第二步：将编码器的输入：src_embed_x,送入编码器，得到编码器的输出结果;encoder_output--》[2,4,512]
        encoder_output = self.encoder(src_embed_x, source_mask)
        # 第三步：获取解码器的输出
        # 3.1 将原始的输入target，送入（词嵌入层+位置编码器层）对象得到--》word_Embedding；解码器输入部分
        #  tgt_embed_x--》[2,4,512]
        tgt_embed_x = self.tgt_embed(target)
        # 3.2 获取解码器结果
        # decoder_output--》[2,4,512]
        decoder_output = self.decoder(tgt_embed_x, encoder_output, source_mask, target_mask)
        # 第四步：将解码器的输出结果送入输出层，得到最后模型的输出
        # result-->[2,4,1000]
        result = self.generator(decoder_output)
        return result

# 测试一下EncoderDecoder
def make_model(source_vocab, target_vocab, N=6, d_model=512, d_ff=2048, head=8, dropout=0.1):
    # 实例化EncoderDecoder
    # 实例化编码器输入的embed词嵌入层对象
    encoder_embed = MyEmbedding(vocab_size=source_vocab, d_model=d_model)
    # 实例化解码器输入的embed词嵌入层对象
    decoder_embed = MyEmbedding(vocab_size=target_vocab, d_model=d_model)
    # 实例化位置编码器对象
    pe = PositionEncoding(d_model=d_model, dropout_p=dropout)
    # 实例化多头注意力机制的对象
    atten = MutiHeadAttention(head=head, embed_dim=d_model)
    # 实例化前馈全连接层对象
    ff = FeedForward(d_model=d_model, d_ff=d_ff)

    c = copy.deepcopy
    # 获得第一个参数：encoder编码的对象
    encoder = Encoder(layer=EncoderLayer(d_model, c(atten), c(ff), dropout), N=N)
    # 获得第二个参数：decoder解码器对象
    decoder = Decoder(layer=DecoderLayer(d_model, c(atten), c(atten), c(ff), dropout), N=N)
    # 获得第三个参数：src_embed:编码器输入部分
    src_embed = nn.Sequential(encoder_embed, c(pe))
    # 获得第四个参数：tgt_embed:解码器输入部分
    tgt_embed = nn.Sequential(decoder_embed, c(pe))
    # 获得第五个参数：generator：输出部分
    generator = Generator(d_model=d_model, vocab_size=target_vocab)

    my_transformer = EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator)

    for p in my_transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return my_transformer
if __name__ == '__main__':
    model = make_model(source_vocab=1000, target_vocab=2000)
    print(model)
    source = torch.tensor([[1, 10, 20, 30], [3, 5, 90, 7]],dtype=torch.long)
    target = torch.tensor([[4, 100, 2, 30], [3, 51, 90, 6]],dtype=torch.long)
    # 这里只是测试，实际不相等
    source_mask=target_mask=torch.zeros(8, 4, 4)
    result = model(source, target, source_mask, target_mask)
    print(f'最后transformer模型的输出result--》{result.shape}')

