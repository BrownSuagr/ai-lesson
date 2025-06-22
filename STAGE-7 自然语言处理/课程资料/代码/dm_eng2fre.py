# coding:utf-8
# 用于正则表达式
import re
# 用于构建网络结构和函数的torch工具包
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# torch中预定义的优化方法工具包
import torch.optim as optim
import time
# 用于随机生成数据
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# 定义设备：是否需要GPU训练
# windows或者linux的写法
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# M1及其以上的电脑
# device = 'mps'
# print(f'device--》{device}')
# 定义开始字符：特殊token
SOS_token = 0
# 定义结束字符：特殊token
EOS_token = 1
# 定义句子最大长度（包含标点符号）
MAX_LENGTH = 10
# 定义数据集的路径
data_path = './data/eng-fra-v2.txt'

# 定义文本清洗函数
def normalString(s):
    # s:代表输入的是一个字符串
    # 第一步：将字符串里面的大写编程小写，去除两边的空白
    s1 = s.lower().strip()
    # 第二步：对字符串里面的标点符号.?! 前面都加上空格，来和原始的单词分割开
    s2 = re.sub(r"([.!?])", r" \1", s1)
    # 第二步：除了字符串里面的a-zA-Z.?!其他字符都用空格来替代
    s3 = re.sub(r'[^a-zA-Z.!?]+', r' ', s2)
    return s3

# 定义函数实现文本数据清洗+构建词典
def get_data():
    # 1.读取数据到内存
    with open(data_path, 'r', encoding='utf-8')as fr:
        lines = fr.readlines()
    # print(f'lines--》{lines[:2]}')
    # 2.得到数据对my_pairs-->[[英文，法文], [英文，法文], ...]
    my_pairs = [[normalString(s) for s in line.split('\t')] for line in lines]
    # print(f'my_pairs-->{my_pairs[:4]}')
    # print(f'my_pairs0-->{my_pairs[8000][0]}')
    # print(f'my_pairs1-->{my_pairs[8000][1]}')
    # 3.构建英文字典和法文字典
    english_word2index = {"SOS": 0, "EOS": 1}
    english_word_n = 2
    french_word2index = {"SOS": 0, "EOS": 1}
    french_word_n = 2
    for pair in my_pairs:
        # 获取英文词典
        for word in pair[0].split(' '):
            if word not in english_word2index:
            # if word not in english_word2index.keys():
                english_word2index[word] = english_word_n
                english_word_n += 1
                # 等价于下面
                # english_word2index[word] = len(english_word2index)
        # 获取法文字典
        for word in pair[1].split(' '):
            if word not in french_word2index:
                french_word2index[word] = french_word_n
                french_word_n += 1
    # print(f'english_word2index--》{english_word2index}')
    # print(f'french_word2index--》{french_word2index}')

    # 4。获取字典：english_index2word, french_index2word
    english_index2word = {v: k for k, v in english_word2index.items()}
    french_index2word = {v: k for k, v in french_word2index.items()}
    # print(f'english_index2word--》{english_index2word}')
    # print(f'french_index2word--》{french_index2word}')

    return english_word2index, english_index2word, english_word_n, french_word2index, french_index2word, french_word_n, my_pairs


english_word2index, english_index2word, english_word_n, french_word2index,\
    french_index2word, french_word_n, my_pairs = get_data()

# 定义dataset
class MyPairsDataset(Dataset):
    def __init__(self, my_pairs):
        # 样本
        self.my_pairs = my_pairs
        # 样本的数量
        self.sample_len = len(my_pairs)

    def __len__(self):
        return self.sample_len

    def __getitem__(self, index):
        # index异常值处理
        index = min(max(index, 0), self.sample_len-1)
        # 根据索引取出对应英文和法文
        x = self.my_pairs[index][0] # 英文
        # print(f'x---》{x}')
        y = self.my_pairs[index][1] # 法文
        # print(f'y---》{y}')
        # 将x和y都进行数字化表示
        x1 = [english_word2index[word] for word in x.split(' ')]
        x1.append(EOS_token)
        tensor_x = torch.tensor(x1, dtype=torch.long, device=device)

        y1 = [french_word2index[word] for word in y.split(" ")]
        y1.append(EOS_token)
        tensor_y = torch.tensor(y1, dtype=torch.long, device=device)
        return tensor_x, tensor_y

def test_dataset():
    print(english_word2index)
    print(french_word2index)
    my_dataset = MyPairsDataset(my_pairs)
    print(len(my_dataset))
    # tensor_x, tensor_y = my_dataset[0]
    # print(f'tensor_x--》{tensor_x}')
    # print(f'tensor_y--》{tensor_y}')
    # 实例化dataloader
    my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
    for x, y in my_dataloader:
        print(f'x--->{x.shape}')
        print(f'y--->{y.shape}')
        break


# 定义编码器模型
class EncoderGRU(nn.Module):
    def __init__(self, english_vocab_size, hidden_size):
        super().__init__()
        # english_vocab_size：英文单词的总个数
        self.vocab_size = english_vocab_size
        # hidden_size:代表词嵌入维度:假设为256
        self.hidden_size = hidden_size
        # 定义Embedding层
        self.embed = nn.Embedding(self.vocab_size, self.hidden_size)
        # 定义GRU层
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)

    def forward(self, input, hidden):
        # input--》[bacth_size, seq_len]-->【1，6】
        # hidden-->[1, 1, 256]
        # 将input通过embedding层,得到每个词的嵌入结果
        # input1--》[1,6,256]
        input1 = self.embed(input)
        # 将input1和hidden送入模型
        output, hn = self.gru(input1, hidden)
        return output, hn

    def inithidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# 测试编码器
def test_encoder():
    my_dataset = MyPairsDataset(my_pairs)
    print(len(my_dataset))
    # 实例化dataloader
    my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
    # 实例化模型
    encoder_gru = EncoderGRU(english_vocab_size=english_word_n, hidden_size=256)
    encoder_gru = encoder_gru.to(device=device)

    for x, y in my_dataloader:
        h0 = encoder_gru.inithidden()
        output, hn = encoder_gru(x, h0)
        print(f'output--》{output.shape}')
        print(f'output--》{output.device}')
        break


# 定义不带attention的解码器
class DecoderGRU(nn.Module):
    def __init__(self, french_vocab_size, hidden_size):
        super().__init__()
        # french_vocab_size代表法文单词的总个数:4345
        self.vocab_size = french_vocab_size
        # hidden_size 代表词嵌入的维度
        self.hidden_size = hidden_size
        # 定义Embedding层
        self.embed = nn.Embedding(self.vocab_size, self.hidden_size)
        # 定义gru层
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        # 定义输出层out
        self.out = nn.Linear(self.hidden_size, self.vocab_size)
        # 定义logsoftmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        #  input-->shape[1, 1]  ,一个字符一个字符去解码
        # 将input进行Embedding;input1=[1,1,256]
        input1 = self.embed(input)
        # 对embedding之后的结果，进行relu激活，防止过拟合
        input1 = F.relu(input1)
        # 将input1和hidden送入gru模型:output--》[1,1,256]
        output, hidden = self.gru(input1, hidden)
        # 需要对output送入输出层:result-->[1, 4345]
        result = self.softmax(self.out(output[0]))
        return result, hidden

    def inithidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# 测试不带attention的解码器
def test_decoder():
    # 实例化数据源对象
    my_dataset = MyPairsDataset(my_pairs)
    print(len(my_dataset))
    # 实例化dataloader
    my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
    # 实例化编码器模型
    encoder_gru = EncoderGRU(english_vocab_size=english_word_n, hidden_size=256)
    encoder_gru = encoder_gru.to(device=device)
    # 实例化解码器模型
    decoder_gru = DecoderGRU(french_vocab_size=french_word_n, hidden_size=256)
    decoder_gru = decoder_gru.to(device=device)
    # 遍历数据
    for i, (x, y) in enumerate(my_dataloader):
        print(f'x-->{x}')
        print(f'y-->{y}')
        # 将x送入编码器
        h0 = encoder_gru.inithidden()
        encoder_output, hidden = encoder_gru(x, h0)
        # 解码：一个字符一个字符去解码
        for j in range(y.shape[1]):
            temp = y[0][j].view(1, -1)
            output, hidden = decoder_gru(temp, hidden)
            print(f'output--》{output.shape}')
        break

# 定义带attention的解码器
class AttenDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super().__init__()
        # vocab_size：代表法文单词的总个数:4345
        self.vocab_size = vocab_size
        # hidden_size:代表词嵌入的维度
        self.hidden_size = hidden_size
        # dropout_p：随即失活的系数
        self.dropout_p = dropout_p
        # max_length：最大句子长度
        self.max_length = max_length
        # 定义Embedding层
        self.embed = nn.Embedding(self.vocab_size, self.hidden_size)
        # 定义dropout层
        self.dropout = nn.Dropout(p=self.dropout_p)
        # 定义第一个全连接层：得到注意力的权重分数:
        self.attn = nn.Linear(self.hidden_size*2, self.max_length)
        # 定义第二个全连接层：让注意力按照指定维度输出
        self.attn_combine = nn.Linear(self.hidden_size*2, self.hidden_size)
        # 定义GRU层
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        # 定义out输出层
        self.out = nn.Linear(self.hidden_size, self.vocab_size)
        # 定义softmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden, encoder_output):
        # input--》query--》Q--》[1,1]-->[[8]]
        # hidden-->key-->K-->[1,1,256]
        # encoder_output-->value--》V--》[max_len,256]-->[10,256]
        # 1.将input经过embed-->embeded-->[1,1,256]
        embedded = self.embed(input)
        # 2. 将embeded的结果经过dropout，防止过拟合;不改变形状
        embedded = self.dropout(embedded)
        # 3. 按照注意力计算步骤完成注意力的计算
        # 3.1 按照注意力计算步骤的第一步：将Q、K、V按照第一计算规则来计算
        # 将embedded=Q和K=hidden拼接，然后经过attn(线性层)变化，得到权重分数
        # embedded[0]-->[1,256]; hidden[0]-->[1,256],cat之后【1，512】;atten_weight->[1,10]
        atten_weight = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]),dim=-1)),dim=-1)
        # 将atten_weight和V=encoder_output进行相乘，别忘了升维度：attn_applied--》【1,1,256】
        attn_applied = torch.bmm(atten_weight.unsqueeze(dim=0),encoder_output.unsqueeze(dim=0))
        # 3.2  因为第一步是拼接，所以这里第二步：需要将Q和attn_applied结果再次拼接：cat_tensor--》【1,512】
        cat_tensor = torch.cat((embedded[0], attn_applied[0]),dim=-1)
        # 3.3 第三步：经过线性变化，按照指定维度输出:atten = [1,1,256]
        atten = self.attn_combine(cat_tensor.unsqueeze(dim=0))
        # 4. 将注意力的结果进行relu函数激活，为了防止过拟合;不改变形状
        atten = F.relu(atten)
        # 5.将atten和hidden共同送入GRU模型:gru_output[1,1,256]; hidden[1,1,256]
        gru_output, hidden = self.gru(atten, hidden)
        # 6.经过输出层:result = [1,4345]
        result = self.softmax(self.out(gru_output[0]))
        return result, hidden, atten_weight

# 测试带attenion的解码器

def test_attentionDecoder():
    # 实例化数据源对象
    my_dataset = MyPairsDataset(my_pairs)
    print(len(my_dataset))
    # 实例化dataloader
    my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
    # 实例化编码器模型
    encoder_gru = EncoderGRU(english_vocab_size=english_word_n, hidden_size=256)
    encoder_gru = encoder_gru.to(device=device)
    # 实例化带attenion的解码器
    attention_decoder_gru = AttenDecoder(vocab_size=french_word_n, hidden_size=256)
    attention_decoder_gru = attention_decoder_gru.to(device=device)
    # 迭代数据
    for i, (x, y) in enumerate(my_dataloader):
        print(f'x-->{x.shape}')
        print(f'y-->{y.shape}')
        # 将x送入编码器
        h0 = encoder_gru.inithidden()
        output, hidden = encoder_gru(x, h0)
        print(f'output-->{output.shape}')
        print(f'hidden-->{hidden.shape}')
        # 准备解码器的输入
        encoder_c = torch.zeros(MAX_LENGTH, encoder_gru.hidden_size, device=device)
        print(f'encoder_c--》{encoder_c}')
        # 将真实长度的编码结果赋值
        for idx in range(output.shape[1]):
            encoder_c[idx] = output[0][idx]
        print(f'encoder_c1--》{encoder_c}')

        # 解码：一个字符一个字符去解码
        for idx in range(y.shape[1]):
            temp = y[0][idx].view(1, -1)
            decoder_output, hidden, atten_weight = attention_decoder_gru(temp, hidden, encoder_c)
            print(f'decoder_output--》{decoder_output.shape}')

        break

epochs = 2
mylr = 1e-4
teacher_forcing_ratio = 0.5
# 定义内部训练函数
def Train_Iters(x, y, encoder, atten_decoder, encoder_adam, decoder_adam, crossentropy):
    # 1.将x送入编码器，得到编码结果；x==>[1,6];得到编码器的结果
    h0 = encoder.inithidden()
    encoder_output, encoder_hidden = encoder(x, h0)
    # print(f'encoder_output-->{encoder_output.shape}')
    # print(f'encoder_hidden-->{encoder_hidden.shape}')
    # 2.准备解码器的参数
    # 2.1 准备第一个参数：中间语意张量C: encoder_outputs
    encoder_outputs_c = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)
    # print(f'encoder_outputs_c--》{encoder_outputs_c}')
    # 将真实的x编码器后的结果赋值
    for idx in range(x.shape[1]):
        encoder_outputs_c[idx] = encoder_output[0, idx]
    # print(f'encoder_outputs_c--》{encoder_outputs_c}')
    # 2.2 准备第二个参数：解码器第一个时间步的hidden，直接利用的是编码器最后一个时间步的隐藏层结果
    decoder_hidden  = encoder_hidden
    # 2.3 准备第三个参数：一开始的输入是一个特殊字符：SOS_TOKEN
    input_y = torch.tensor([[SOS_token]], device=device)
    # 3.定义初始化损失值
    my_loss = 0.0
    # 4. 定义真实翻译的句子长度
    y_len = y.shape[1]

    # 5.是否使用teacher_forcing
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    # 6.进行解码
    if use_teacher_forcing:
        for idx in range(y_len):
            # 模型预测结果
            output_y, decoder_hidden, atten_weight = atten_decoder(input_y, decoder_hidden, encoder_outputs_c)
            # print(f'output_y-——》{output_y.shape}')
            # 当前时间步真实的标签值
            target_y = y[0][idx].view(1)
            # print(f'target_y--》{target_y}')
            # 计算损失
            my_loss = my_loss + crossentropy(output_y, target_y)
            # 更新input_y:使用真实的值
            input_y = y[0][idx].view(1, -1)

    else:
        for idx in range(y_len):
            # 模型预测结果
            output_y, decoder_hidden, atten_weight = atten_decoder(input_y, decoder_hidden, encoder_outputs_c)
            # print(f'output_y-——》{output_y.shape}')
            # 当前时间步真实的标签值
            target_y = y[0][idx].view(1)
            # print(f'target_y--》{target_y}')
            # 计算损失
            my_loss = my_loss + crossentropy(output_y, target_y)
            # 得出当前时间步模型预测结果
            topv, topi = torch.topk(output_y, k=1)
            input_y = topi.detach()

    # 梯度清零
    encoder_adam.zero_grad()
    decoder_adam.zero_grad()
    # 反向传播
    my_loss.backward()
    # 梯度更新
    encoder_adam.step()
    decoder_adam.step()

    return my_loss.item() / y_len # 该样本的平均损失
# 定义训练函数
def train_seq2seq():
    # 1.实例化数据源对象
    my_dataset = MyPairsDataset(my_pairs)
    # 2.实例化dataloader
    my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
    # 3. 实例化编码器
    encoder = EncoderGRU(english_vocab_size=english_word_n, hidden_size=256)
    encoder = encoder.to(device=device)
    # 4. 实例化解码器（带attention）
    atten_decoder = AttenDecoder(vocab_size=french_word_n, hidden_size=256)
    atten_decoder = atten_decoder.to(device=device)
    # 5. 实例化优化器
    encoder_adam = optim.Adam(encoder.parameters(), lr=mylr)
    decoder_adam = optim.Adam(atten_decoder.parameters(), lr=mylr)
    # 6. 实例化损失函数的对象
    crossentropy = nn.NLLLoss()
    # 7. 定义打印日志的参数
    plot_loss_list = []
    # 8. 开始训练（外部循环）
    for epoch_idx in range(1, epochs+1):
        # 定义变量
        print_loss_total = 0.0
        plot_loss_total = 0.0
        start_time = time.time()
        # 开始内部迭代数据
        for item, (x, y) in enumerate(tqdm(my_dataloader), start=1):
            # print(f'x-->{x}')
            # print(f'y-->{y}')
            # 调用内部函数
            my_loss = Train_Iters(x, y, encoder, atten_decoder, encoder_adam, decoder_adam, crossentropy)
            # print(f'my_loss---》{my_loss}')
            print_loss_total = print_loss_total + my_loss
            plot_loss_total = plot_loss_total + my_loss
            # 每隔1000步，打印下训练日志
            if item % 1000 == 0:
                avg_loss = print_loss_total / 1000
                print_loss_total = 0.0
                use_time = time.time() - start_time
                print('当前轮次:%d,损失值是：%.3f, 使用的时间：%d'% (epoch_idx, avg_loss, use_time))
            # 每隔100步，保存平均损失：画图
            if item % 100 == 0:
                avg_plot_loss = plot_loss_total / 100
                plot_loss_list.append(avg_plot_loss)
                # 如果画图报错：
                # plot_loss_list.append(avg_plot_loss.cpu().detach().numpy())
                plot_loss_total = 0.0
        # 每轮都要保存模型
        torch.save(encoder.state_dict(), './save_model/encoder_%d.pth'%(epoch_idx))
        torch.save(atten_decoder.state_dict(), './save_model/atten_decoder_%d.pth'%(epoch_idx))

    # 9.画损失曲线图
    plt.figure()
    plt.plot(plot_loss_list)
    plt.savefig('./data/eng2fre_loss.png')
    plt.show()

#模型预测/评估函数
def seq2seq2_Evaluate(x, my_encoder, my_decoder):
    # print(f'x-->{x}')
    with torch.no_grad():
        # 1. 将x送入编码器，得到编码之后的结果
        h0 = my_encoder.inithidden()
        encoder_output, encoder_hidden = my_encoder(x, h0)
        # print(f'encoder_output-->{encoder_output.shape}')
        # 2. 准备解码器的参数
        # 2.1 中间语意张量C，encoder_output_c
        encoder_output_c = torch.zeros(MAX_LENGTH, my_encoder.hidden_size, device=device)
        for idx in range(x.shape[1]):
            encoder_output_c[idx] = encoder_output[0, idx]
        # 2.2 将编码器最后一个单词的隐藏层输出结果，作为解码器的第一个时间步隐藏层输入
        decoder_hidden = encoder_hidden
        # 2.3 定义解码器第一个时间步的输入：开始字符SOS
        input_y = torch.tensor([[SOS_token]], device=device)
        # 3. 准备空列表：存储解码出来的法文单词
        decoder_list = []
        # 4. 准备全零的二维矩阵：将存储每一个时间步得到的注意力权重
        decoder_attention = torch.zeros(MAX_LENGTH, MAX_LENGTH)
        # 5. 开始解码
        for idx in range(MAX_LENGTH):
            output_y, decoder_hidden, atten_weight = my_decoder(input_y, decoder_hidden, encoder_output_c)
            # print(f'output_y--》{output_y.shape}')
            # print(f'atten_weight--》{atten_weight.shape}')
            # 取出output_y中4345个概率值对应最大的值以及对应的索引
            _, topi = torch.topk(output_y, k=1)
            # print(f'topi--》{topi}')
            decoder_attention[idx] = atten_weight
            # 判断预测的结果是否是终止符号：EOS
            if topi.item() == EOS_token:
                decoder_list.append('<EOS>')
                break
            else:
                decoder_list.append(french_index2word[topi.item()])
            input_y = topi.detach()

        return decoder_list, decoder_attention[:idx+1]
# 测试模型预测函数
def test_seq2seqEvaluate():
    # 1. 加载训练好的编码器模型
    my_encoder = EncoderGRU(english_vocab_size=english_word_n, hidden_size=256)
    my_encoder.load_state_dict(torch.load('./save_model/encoder_2.pth'))
    # my_encoder.load_state_dict(torch.load('./save_model/my_encoderrnn_2.pth',map_location='cpu'),  strict=False)
    my_encoder = my_encoder.to(device=device)
    # 2.加载训练好的解码器模型
    my_decoder = AttenDecoder(vocab_size=french_word_n, hidden_size=256)
    my_decoder.load_state_dict(torch.load('./save_model/atten_decoder_2.pth'))
    # my_decoder.load_state_dict(torch.load('./save_model/my_attndecoderrnn_2.pth', map_location='cpu'),  strict=False)
    my_decoder = my_decoder.to(device=device)
    # 3. 准备预测数据
    my_samplepairs = [
        ['i m impressed with your french .', 'je suis impressionne par votre francais .'],
        ['i m more than a friend .', 'je suis plus qu une amie .'],
        ['she is beautiful like her mother .', 'elle est belle comme sa mere .']]
    # 4. 循环数据，对每一个数据进行预测，对比真实的结果
    for i, pair in enumerate(my_samplepairs):
        x = pair[0]
        y = pair[1]
        # 对x也就是英文文本进行处理，将所有的英文单词编程数字
        temp_x = [english_word2index[word] for word in x.split(" ")]
        temp_x.append(EOS_token)
        tensor_x = torch.tensor([temp_x], dtype=torch.long, device=device)
        decoder_list, atten_weight = seq2seq2_Evaluate(tensor_x, my_encoder, my_decoder)
        # print(f'decoder_list--》{decoder_list}')
        predict = ' '.join(decoder_list)
        print(f'x-->{x}')
        print(f'y--》{y}')
        print(f'predict-->{predict}')
        print('*'*80)

# 定义函数：实现注意力权重的绘图
def test_attention():
    # 1. 加载训练好的编码器模型
    my_encoder = EncoderGRU(english_vocab_size=english_word_n, hidden_size=256)
    my_encoder.load_state_dict(torch.load('./save_model/encoder_2.pth'))
    my_encoder = my_encoder.to(device=device)
    # 2.加载训练好的解码器模型
    my_decoder = AttenDecoder(vocab_size=french_word_n, hidden_size=256)
    my_decoder.load_state_dict(torch.load('./save_model/atten_decoder_2.pth'))
    my_decoder = my_decoder.to(device=device)
    # 3.准备数据
    sentence = "we re both teachers ."
    temp_x = [english_word2index[word] for word in sentence.split(" ")]
    temp_x.append(EOS_token)
    tensor_x = torch.tensor([temp_x], dtype=torch.long, device=device)
    decoder_list, atten_weight = seq2seq2_Evaluate(tensor_x, my_encoder, my_decoder)
    predict = ' '.join(decoder_list)
    print(f'predict-->{predict}')

    # 绘图
    plt.matshow(atten_weight)
    plt.savefig('./data/attention.png')
    plt.show()



if __name__ == '__main__':
    # test_dataset()
    # test_encoder()
    # test_decoder()
    # test_attentionDecoder()
    # train_seq2seq()
    # test_seq2seqEvaluate()
    test_attention()