#  coding:utf-8
# 导入torch工具
import torch
# 导入nn准备构建模型
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# 导入torch的数据源 数据迭代器工具包
from torch.utils.data import Dataset, DataLoader
# 用于获得常见字母及字符规范化
import string
# 导入时间工具包
import time
# 引入制图工具包
import matplotlib.pyplot as plt

# 获取常用的字符
all_letters = string.ascii_letters + " .,;'"
print(f'all_letters--》{all_letters}')
n_letters = len(all_letters)
print(n_letters)

# 国家名 种类数
categorys = ['Italian', 'English', 'Arabic', 'Spanish', 'Scottish', 'Irish', 'Chinese', 'Vietnamese', 'Japanese',
             'French', 'Greek', 'Dutch', 'Korean', 'Polish', 'Portuguese', 'Russian', 'Czech', 'German']
# 国家名 个数
categorynum = len(categorys)
print('categorynum--->', categorynum)

# 读取数据到内存
def read_data(filepath):
    # 定义两个空列表
    my_list_x = [] # list()
    my_list_y = []
    # 读文件
    with open(filepath,encoding='utf-8')as fr:
        lines = fr.readlines()
    for line in lines:
        # 如过line的长度小于等于5，跳过
        if len(line) <= 5:
            continue
        x, y = line.strip().split('\t')
        my_list_x.append(x)
        my_list_y.append(y)
    return my_list_x, my_list_y


# 自定义Dataset类
class NameClassDataset(Dataset):
    def __init__(self, my_list_x, my_list_y):
        super().__init__()
        # 获取x, y
        self.my_list_x = my_list_x
        self.my_list_y = my_list_y
        # 统计样本长度
        self.sample_len = len(my_list_x)

    # 定义魔法方法__len__()，可以方便的直接对类的对象进行操作
    def __len__(self):
        return self.sample_len

    # 定义魔法方法__getitem__(index)方法，通过该模型方法，直接操作对象，根据索引取出某个样本(x, y）
    def __getitem__(self, index):
        # index异常值处理[0, self.sample_len-1]
        index = min(max(index, 0), self.sample_len-1)
        # 根据索引取x和y
        x = self.my_list_x[index]
        # print(f'x-->{x}')
        y = self.my_list_y[index]
        # print(f'y-->{y}')
        # 接下来文本张量化
        # tensor_x，就是人名对应的oen-hot编码
        tensor_x = torch.zeros(len(x), n_letters)
        # print(f'tensor_x--》{tensor_x}')
        for idx, letter in enumerate(x):
            tensor_x[idx][all_letters.find(letter)] = 1
         # 对目标值需要张量化
        tensor_y = torch.tensor(categorys.index(y), dtype=torch.long)
        return tensor_x, tensor_y


def test_dataset():
    my_list_x, my_list_y = read_data(filepath='./data/name_classfication.txt')
    # 实例化自定义的Dataset
    my_dataset = NameClassDataset(my_list_x, my_list_y)
    print(len(my_dataset))
    # print(my_dataset.__len__())
    # my_dataset.__getitem__(0)
    # 实例化Dataloader
    my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
    print('11', len(my_dataloader))
    for x, y in my_dataloader:
        print(f'x-->{x.shape}')
        print(f'y-->{y.shape}')
        break


# 定义传统RNN
class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        # input_size代表输入x的词嵌入表示维度
        self.input_size = input_size
        # hidden_size代表RNN的输出维度
        self.hidden_size = hidden_size
        # output_size代表国家类别总数-->18
        self.output_size = output_size
        # num_layers代表RNN层数
        self.num_layers = num_layers
        # 实例化RNN模型
        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers)
        # 实例化linear层
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        # 实例化softmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        #  input-->输入是二维的--》[seq_len, input_size]-->[6, 57]
        # hidden-->输入三维的--》[num_layers, batch_size, hidden_size]-->[1,1,128]
        # 第一步需要对input升维，按照dim=1这个维度来升-->input--》[seq_len,batch_size,input_size]-[6, 1, 57]
        input = input.unsqueeze(dim=1)
        # 第二步：将input和hidden送入模型:output-->[6, 1, 128], hn-->[1, 1, 128]
        output, hn = self.rnn(input, hidden)
        # 第三步：取出最后一个单词对应的向量当作真个样本的向量语意表示
        # tempv = hidden[0, :, :]
        tempv = output[-1, :, :]
        # 第四步：将tempv-->[1, 128]送入linear得到预测结果:
        result = self.linear(tempv) # result-->[1, 18]

        return self.softmax(result), hn

    def inithidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)

# 测试RNN
def test_rnn_model():
    my_list_x, my_list_y = read_data(filepath='./data/name_classfication.txt')
    # 实例化自定义的Dataset
    my_dataset = NameClassDataset(my_list_x, my_list_y)
    print(len(my_dataset))
    # print(my_dataset.__len__())
    # my_dataset.__getitem__(0)
    # 实例化Dataloader
    my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
    # 实例化模型
    my_rnn = MyRNN(input_size=57, hidden_size=128, output_size=18)
    #循环
    for i, (x, y) in enumerate(my_dataloader):
        # x.shape--》[batch_size, seq_len, input_size]
        print(f'x-->{x.shape}')
        hidden = my_rnn.inithidden()
        # x[0]--》【seq_len, input_size】
        output, hn = my_rnn(x[0], hidden)
        print(f'output-->{output.shape}')
        break


# 自定义LSTM
class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        # input_size代表输入x的词嵌入表示维度
        self.input_size = input_size
        # hidden_size代表RNN的输出维度
        self.hidden_size = hidden_size
        # output_size代表国家类别总数-->18
        self.output_size = output_size
        # num_layers代表RNN层数
        self.num_layers = num_layers
        # 实例化LSTM模型
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)
        # 实例化linear层
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        # 实例化softmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden, c):
        #  input-->输入是二维的--》[seq_len, input_size]-->[6, 57]
        # hidden,c-->输入三维的--》[num_layers, batch_size, hidden_size]-->[1,1,128]
        # 第一步需要对input升维，按照dim=1这个维度来升-->input--》[seq_len,batch_size,input_size]-[6, 1, 57]
        input = input.unsqueeze(dim=1)
        # 第二步：将input和hidden,c送入模型:output-->[6, 1, 128], hn,cn-->[1, 1, 128]
        output, (hn, cn) = self.lstm(input, (hidden, c))
        # 第三步：取出最后一个单词对应的向量当作真个样本的向量语意表示
        # tempv = hidden[0, :, :]
        tempv = output[-1, :, :]
        # 第四步：将tempv-->[1, 128]送入linear得到预测结果:
        result = self.linear(tempv) # result-->[1, 18]

        return self.softmax(result), hn, cn

    def inithidden(self):
        hidden = torch.zeros(self.num_layers, 1, self.hidden_size)
        c0 = torch.zeros(self.num_layers, 1, self.hidden_size)
        return hidden, c0

# 测试LSTM
def test_lstm_model():
    my_list_x, my_list_y = read_data(filepath='./data/name_classfication.txt')
    # 实例化自定义的Dataset
    my_dataset = NameClassDataset(my_list_x, my_list_y)
    print(len(my_dataset))
    # print(my_dataset.__len__())
    # my_dataset.__getitem__(0)
    # 实例化Dataloader
    my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
    # 实例化模型
    my_lstm = MyLSTM(input_size=57, hidden_size=128, output_size=18)
    #循环
    for i, (x, y) in enumerate(my_dataloader):
        # x.shape--》[batch_size, seq_len, input_size]
        print(f'x-->{x.shape}')
        h0, c0 = my_lstm.inithidden()
        # x[0]--》【seq_len, input_size】
        output, hn, cn = my_lstm(x[0], h0, c0)
        print(f'lstm-->output-->{output.shape}')
        break

# 定义GRu
class MyGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        # input_size代表输入x的词嵌入表示维度
        self.input_size = input_size
        # hidden_size代表RNN的输出维度
        self.hidden_size = hidden_size
        # output_size代表国家类别总数-->18
        self.output_size = output_size
        # num_layers代表RNN层数
        self.num_layers = num_layers
        # 实例化GRU模型
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers)
        # 实例化linear层
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        # 实例化softmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        #  input-->输入是二维的--》[seq_len, input_size]-->[6, 57]
        # hidden-->输入三维的--》[num_layers, batch_size, hidden_size]-->[1,1,128]
        # 第一步需要对input升维，按照dim=1这个维度来升-->input--》[seq_len,batch_size,input_size]-[6, 1, 57]
        input = input.unsqueeze(dim=1)
        # 第二步：将input和hidden送入模型:output-->[6, 1, 128], hn-->[1, 1, 128]
        output, hn = self.gru(input, hidden)
        # 第三步：取出最后一个单词对应的向量当作真个样本的向量语意表示
        # tempv = hidden[0, :, :]
        tempv = output[-1, :, :]
        # 第四步：将tempv-->[1, 128]送入linear得到预测结果:
        result = self.linear(tempv) # result-->[1, 18]

        return self.softmax(result), hn

    def inithidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)

# 测试GRU
def test_gru_model():
    my_list_x, my_list_y = read_data(filepath='./data/name_classfication.txt')
    # 实例化自定义的Dataset
    my_dataset = NameClassDataset(my_list_x, my_list_y)
    print(len(my_dataset))
    # print(my_dataset.__len__())
    # my_dataset.__getitem__(0)
    # 实例化Dataloader
    my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
    # 实例化模型
    my_gru = MyGRU(input_size=57, hidden_size=128, output_size=18)
    #循环
    for i, (x, y) in enumerate(my_dataloader):
        # x.shape--》[batch_size, seq_len, input_size]
        print(f'x-->{x.shape}')
        hidden = my_gru.inithidden()
        # x[0]--》【seq_len, input_size】
        output, hn = my_gru(x[0], hidden)
        print(f'output-->{output.shape}')
        break
if __name__ == '__main__':
    # test_dataset()
    # test_rnn_model()
    # test_lstm_model()
    test_gru_model()
