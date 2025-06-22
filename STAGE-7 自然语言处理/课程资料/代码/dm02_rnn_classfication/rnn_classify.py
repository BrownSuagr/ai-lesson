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
import json
from tqdm import tqdm
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
my_lr = 1e-3
epochs = 2
# 定义RNN模型的训练函数
def dm_train_rnn():
    # 读数据
    my_list_x, my_list_y = read_data(filepath='./data/name_classfication.txt')
    # 实例化自定义的Dataset
    my_dataset = NameClassDataset(my_list_x, my_list_y)
    # 实例化模型
    input_size = n_letters # 57
    hidden_size = 128
    output_size = categorynum # 18
    my_rnn = MyRNN(input_size, hidden_size, output_size)
    print(f'my_rnn--》{my_rnn}')
    # 定义损失函数和优化器
    my_crossentropy = nn.NLLLoss()
    my_adam = optim.Adam(my_rnn.parameters(), lr=my_lr)
    # 定义打印训练日志的参数
    start_time = time.time() # 开始的时间
    total_iter_num = 0 # 已经训练的样本数量
    total_loss = 0.0 # 已经训练样本的总损失
    total_loss_list = [] # 将每迭代100个样本，计算平均损失并保存列表中
    total_acc_num = 0 # 已经训练的样本中预测正确的样本数量
    total_acc_list = [] # 将每迭代100个样本，计算平均准确率并保存列表中
    # 开始外部epoch循环
    for epoch_idx in range(epochs):
        # 实例化dataloader
        my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
        # 开始内部迭代数据
        for i, (x, y) in enumerate(tqdm(my_dataloader)):
            # x-->[1, 6, 57]-->x[0]=[6, 57]
            h0 = my_rnn.inithidden() # [1,1,128]
            output, hn = my_rnn(x[0], h0) #  output[1, 18]
            # print(f'output--》{output}')
            # print(f'y-->{y}')
            # 计算损失
            my_loss = my_crossentropy(output, y)
            # print(f'my_loss-->{my_loss}')
            # 梯度清零
            my_adam.zero_grad()
            # 反向传播
            my_loss.backward()
            # 梯度更新
            my_adam.step()
            # 统计已经训练的样本总个数
            total_iter_num = total_iter_num + 1
            # 计算已经训练过的样本的总损失
            total_loss = total_loss + my_loss.item()
            # 计算已经训练的样本预测正确个数
            tag = 1 if torch.argmax(output).item() == y.item() else 0
            total_acc_num = total_acc_num + tag
            # 每隔100步计算一下平均损失和平均准确率
            if total_iter_num % 100 == 0:
                # 平均损失
                avg_loss = total_loss / total_iter_num
                total_loss_list.append(avg_loss)
                # 平均准确率
                avg_acc = total_acc_num / total_iter_num
                total_acc_list.append(avg_acc)
            # 每隔2000步打印一下训练日志
            if total_iter_num % 2000 == 0:
                temp_loss = total_loss / total_iter_num
                temp_acc = total_acc_num / total_iter_num
                use_time = time.time() - start_time
                print('轮次:%d, 损失:%.6f, 时间:%d，准确率:%.3f' % (epoch_idx + 1, temp_loss, use_time, temp_acc))
        # 每一轮都保存模型
        torch.save(my_rnn.state_dict(), './save_model/rnn_%d.bin' % (epoch_idx+1))
    # 计算总时间
    all_time = time.time() - start_time
    # 定义字典保存数据
    dict1 = {"total_loss_list": total_loss_list,
             "all_time": all_time,
             "total_acc_list": total_acc_list}
    with open('./data/rnn.json', 'w', encoding='utf-8')as fw:
        fw.write(json.dumps(dict1))
    return total_loss_list, all_time, total_acc_list

# 定义LSTM模型的训练函数
def dm_train_lstm():
    # 读数据
    my_list_x, my_list_y = read_data(filepath='./data/name_classfication.txt')
    # 实例化自定义的Dataset
    my_dataset = NameClassDataset(my_list_x, my_list_y)
    # 实例化模型
    input_size = n_letters # 57
    hidden_size = 128
    output_size = categorynum # 18
    my_lstm = MyLSTM(input_size, hidden_size, output_size)
    print(f'my_lstm--》{my_lstm}')
    # 定义损失函数和优化器
    my_crossentropy = nn.NLLLoss()
    my_adam = optim.Adam(my_lstm.parameters(), lr=my_lr)
    # 定义打印训练日志的参数
    start_time = time.time() # 开始的时间
    total_iter_num = 0 # 已经训练的样本数量
    total_loss = 0.0 # 已经训练样本的总损失
    total_loss_list = [] # 将每迭代100个样本，计算平均损失并保存列表中
    total_acc_num = 0 # 已经训练的样本中预测正确的样本数量
    total_acc_list = [] # 将每迭代100个样本，计算平均准确率并保存列表中
    # 开始外部epoch循环
    for epoch_idx in range(epochs):
        # 实例化dataloader
        my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
        # 开始内部迭代数据
        for i, (x, y) in enumerate(tqdm(my_dataloader)):
            # x-->[1, 6, 57]-->x[0]=[6, 57]
            h0, c0 = my_lstm.inithidden() # [1,1,128]
            output, hn, cn = my_lstm(input=x[0], hidden=h0, c=c0) #  output[1, 18]
            # print(f'output--》{output}')
            # print(f'y-->{y}')
            # 计算损失
            my_loss = my_crossentropy(output, y)
            # print(f'my_loss-->{my_loss}')
            # 梯度清零
            my_adam.zero_grad()
            # 反向传播
            my_loss.backward()
            # 梯度更新
            my_adam.step()
            # 统计已经训练的样本总个数
            total_iter_num = total_iter_num + 1
            # 计算已经训练过的样本的总损失
            total_loss = total_loss + my_loss.item()
            # 计算已经训练的样本预测正确个数
            tag = 1 if torch.argmax(output).item() == y.item() else 0
            total_acc_num = total_acc_num + tag
            # 每隔100步计算一下平均损失和平均准确率
            if total_iter_num % 100 == 0:
                # 平均损失
                avg_loss = total_loss / total_iter_num
                total_loss_list.append(avg_loss)
                # 平均准确率
                avg_acc = total_acc_num / total_iter_num
                total_acc_list.append(avg_acc)
            # 每隔2000步打印一下训练日志
            if total_iter_num % 2000 == 0:
                temp_loss = total_loss / total_iter_num
                temp_acc = total_acc_num / total_iter_num
                use_time = time.time() - start_time
                print('轮次:%d, 损失:%.6f, 时间:%d，准确率:%.3f' % (epoch_idx + 1, temp_loss, use_time, temp_acc))
        # 每一轮都保存模型
        torch.save(my_lstm.state_dict(), './save_model/lstm_%d.bin' % (epoch_idx+1))
    # 计算总时间
    all_time = time.time() - start_time
    # 定义字典保存数据
    dict1 = {"total_loss_list": total_loss_list,
             "all_time": all_time,
             "total_acc_list": total_acc_list}
    with open('./data/lstm.json', 'w', encoding='utf-8')as fw:
        fw.write(json.dumps(dict1))
    return total_loss_list, all_time, total_acc_list

# 定义GRU模型的训练函数
def dm_train_gru():
    # 读数据
    my_list_x, my_list_y = read_data(filepath='./data/name_classfication.txt')
    # 实例化自定义的Dataset
    my_dataset = NameClassDataset(my_list_x, my_list_y)
    # 实例化模型
    input_size = n_letters # 57
    hidden_size = 128
    output_size = categorynum # 18
    my_gru = MyGRU(input_size, hidden_size, output_size)
    print(f'my_gru--》{my_gru}')
    # 定义损失函数和优化器
    my_crossentropy = nn.NLLLoss()
    my_adam = optim.Adam(my_gru.parameters(), lr=my_lr)
    # 定义打印训练日志的参数
    start_time = time.time() # 开始的时间
    total_iter_num = 0 # 已经训练的样本数量
    total_loss = 0.0 # 已经训练样本的总损失
    total_loss_list = [] # 将每迭代100个样本，计算平均损失并保存列表中
    total_acc_num = 0 # 已经训练的样本中预测正确的样本数量
    total_acc_list = [] # 将每迭代100个样本，计算平均准确率并保存列表中
    # 开始外部epoch循环
    for epoch_idx in range(epochs):
        # 实例化dataloader
        my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
        # 开始内部迭代数据
        for i, (x, y) in enumerate(tqdm(my_dataloader)):
            # x-->[1, 6, 57]-->x[0]=[6, 57]
            h0 = my_gru.inithidden() # [1,1,128]
            output, hn = my_gru(x[0], h0) #  output[1, 18]
            # print(f'output--》{output}')
            # print(f'y-->{y}')
            # 计算损失
            my_loss = my_crossentropy(output, y)
            # print(f'my_loss-->{my_loss}')
            # 梯度清零
            my_adam.zero_grad()
            # 反向传播
            my_loss.backward()
            # 梯度更新
            my_adam.step()
            # 统计已经训练的样本总个数
            total_iter_num = total_iter_num + 1
            # 计算已经训练过的样本的总损失
            total_loss = total_loss + my_loss.item()
            # 计算已经训练的样本预测正确个数
            tag = 1 if torch.argmax(output).item() == y.item() else 0
            total_acc_num = total_acc_num + tag
            # 每隔100步计算一下平均损失和平均准确率
            if total_iter_num % 100 == 0:
                # 平均损失
                avg_loss = total_loss / total_iter_num
                total_loss_list.append(avg_loss)
                # 平均准确率
                avg_acc = total_acc_num / total_iter_num
                total_acc_list.append(avg_acc)
            # 每隔2000步打印一下训练日志
            if total_iter_num % 2000 == 0:
                temp_loss = total_loss / total_iter_num
                temp_acc = total_acc_num / total_iter_num
                use_time = time.time() - start_time
                print('轮次:%d, 损失:%.6f, 时间:%d，准确率:%.3f' % (epoch_idx + 1, temp_loss, use_time, temp_acc))
        # 每一轮都保存模型
        torch.save(my_gru.state_dict(), './save_model/gru_%d.bin' % (epoch_idx+1))
    # 计算总时间
    all_time = time.time() - start_time
    # 定义字典保存数据
    dict1 = {"total_loss_list": total_loss_list,
             "all_time": all_time,
             "total_acc_list": total_acc_list}
    with open('./data/gru.json', 'w', encoding='utf-8')as fw:
        fw.write(json.dumps(dict1))
    return total_loss_list, all_time, total_acc_list

# 返回json文件的结果
def read_json(path_name):
    with open(path_name, 'r', encoding='utf-8') as fr:
        line = fr.readline()
    dict1 = json.loads(line)
    total_loss_list = dict1["total_loss_list"]
    all_time = dict1["all_time"]
    total_acc_list = dict1["total_acc_list"]
    return total_loss_list, all_time, total_acc_list
# 画图
def dm_show_picture():
    # 获取rnn、lstm、gru对应训练模型的loss_list,use_time,acc_list
    rnn_total_loss_list, rnn_all_time, rnn_total_acc_list = read_json('./data/rnn.json')
    lstm_total_loss_list, lstm_all_time, lstm_total_acc_list = read_json('./data/lstm.json')
    gru_total_loss_list, gru_all_time, gru_total_acc_list = read_json('./data/gru.json')
    # 画损失曲线
    plt.figure(0)
    plt.plot(rnn_total_loss_list, label="RNN")
    plt.plot(lstm_total_loss_list, color="red", label="LSTM")
    plt.plot(gru_total_loss_list, color="orange", label="GRU")
    plt.legend(loc='upper left')
    plt.savefig('./image/rnn_loss.png')
    plt.show()
    # 画时间对比图
    plt.figure(1)
    x_data = ["RNN", "LSTM", "GRU"]
    y_data = [rnn_all_time, lstm_all_time, gru_all_time]
    plt.bar(range(len(x_data)), y_data, tick_label=x_data)
    plt.legend(loc="upper left")
    plt.savefig('./image/rnn_time.png')
    plt.show()

    # 画准确率曲线
    plt.figure(2)
    plt.plot(rnn_total_acc_list, label="RNN")
    plt.plot(lstm_total_acc_list, color="red", label="LSTM")
    plt.plot(gru_total_acc_list, color="orange", label="GRU")
    plt.legend(loc='upper left')
    plt.savefig('./image/rnn_acc.png')
    plt.show()

# 定义模型训练好的保存路径
my_rnn_path = './save_model/rnn_2.bin'
my_lstm_path = './save_model/lstm_2.bin'
my_gru_path = './save_model/gru_2.bin'

# 定义文本转换为张量的函数
def line2tensor(x):
    #  x--》人名--eg："bai"
    # 初始化全零的张量
    tensor_x = torch.zeros(len(x), n_letters)
    # 进行one-hot表示
    for li, letter in enumerate(x):
        tensor_x[li][all_letters.find(letter)] = 1
    return tensor_x


# 定义RNN预测函数
def dm_rnn_predict(x):
    # x-->eg:bai--》人名
    # 将人名转换为一个张量形式
    tensor_x = line2tensor(x)
    # 实例化模型，并加载训练好的模型参数
    my_rnn = MyRNN(input_size=n_letters, hidden_size=128, output_size=18)
    my_rnn.load_state_dict(torch.load(my_rnn_path))
    # 开始预测
    with torch.no_grad():
        h0 = my_rnn.inithidden()
        output, hn = my_rnn(tensor_x, h0) # output-->【1， 18】
        # print(f'output--》{output}')
        # 获取output结果中，概率值最大的前三个值，并且包括其索引
        topv, topi = torch.topk(output, k=3, dim=-1)
        # topv, topi = output.topk(3, -1, True)
        # print(f'topv-->{topv}')
        # print(f'topi-->{topi}')
        print(f'一个类别一个类别去解码')
        for i in range(3):
            value = topv[0][i]
            idx = topi[0][i]
            category = categorys[idx]
            print('当前人名是%s, value:%d, 国家类别：%s'%(x, value, category))


# 定义LSTM预测函数
def dm_lstm_predict(x):
    # x-->eg:bai--》人名
    # 将人名转换为一个张量形式
    tensor_x = line2tensor(x)
    # 实例化模型，并加载训练好的模型参数
    my_lstm = MyLSTM(input_size=n_letters, hidden_size=128, output_size=18)
    my_lstm.load_state_dict(torch.load(my_lstm_path))
    # 开始预测
    with torch.no_grad():
        h0, c0 = my_lstm.inithidden()
        output, hn, cn = my_lstm(tensor_x, h0, c0) # output-->【1， 18】
        # print(f'output--》{output}')
        # 获取output结果中，概率值最大的前三个值，并且包括其索引
        topv, topi = torch.topk(output, k=3, dim=-1)
        # topv, topi = output.topk(3, -1, True)
        # print(f'topv-->{topv}')
        # print(f'topi-->{topi}')
        print(f'一个类别一个类别去解码')
        for i in range(3):
            value = topv[0][i]
            idx = topi[0][i]
            category = categorys[idx]
            print('当前人名是%s, value:%d, 国家类别：%s'%(x, value, category))


# 定义GRU预测函数
def dm_gru_predict(x):
    # x-->eg:bai--》人名
    # 将人名转换为一个张量形式
    tensor_x = line2tensor(x)
    # 实例化模型，并加载训练好的模型参数
    my_gru = MyGRU(input_size=n_letters, hidden_size=128, output_size=18)
    my_gru.load_state_dict(torch.load(my_gru_path))
    # 开始预测
    with torch.no_grad():
        h0 = my_gru.inithidden()
        output, hn = my_gru(tensor_x, h0) # output-->【1， 18】
        # print(f'output--》{output}')
        # 获取output结果中，概率值最大的前三个值，并且包括其索引
        topv, topi = torch.topk(output, k=3, dim=-1)
        # topv, topi = output.topk(3, -1, True)
        # print(f'topv-->{topv}')
        # print(f'topi-->{topi}')
        print(f'一个类别一个类别去解码')
        for i in range(3):
            value = topv[0][i]
            idx = topi[0][i]
            category = categorys[idx]
            print('当前人名是%s, value:%d, 国家类别：%s'%(x, value, category))

if __name__ == '__main__':
    # test_dataset()
    # test_rnn_model()
    # test_lstm_model()
    # test_gru_model()
    # dm_train_rnn()
    # dm_train_lstm()
    # dm_train_gru()
    # dm_show_picture()
    dm_rnn_predict(x='Ballalatak')
    dm_lstm_predict(x='Ballalatak')
    dm_gru_predict(x='Ballalatak')