# RNN人名分类器案例

## 1 任务目的：

```properties
目的: 给定一个人名，来判定这个人名属于哪个国家
典型的文本分类任务: 18分类---多分类任务
```

### 2 数据格式

- 注意：两列数据，第一列是人名，第二列是国家类别，中间用制表符号"\t"隔开

```properties
Ang	Chinese
AuYong	Chinese
Yuasa	Japanese
Yuhara	Japanese
Yunokawa	Japanese
```

### 3 任务实现流程

```properties
1. 获取数据:案例中是直接给定的
2. 数据预处理: 脏数据清洗、数据格式转换、数据源Dataset的构造、数据迭代器Dataloader的构造
3. 模型搭建: RNN、LSTM、GRU一系列模型
4. 模型训练评估（测试）
5. 模型上线---API接口
```

### 4 数据预处理

#### 4.1读取txt文档数据

目的：

```properties
将文档里面的数据读取到内存中，实际上我们做了一个操作: 将人名存放到一个列表中，国家类别存放到一个类别中
```

代码实现

```python
def read_data(filename):
    # 1. 初始化两个空列表
    my_list_x, my_list_y = [], []
    # 2. 读取文件内容
    with open(filename,'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            if len(line) <= 5:
                continue
            # strip()方法默认将字符串首尾两端的空白去掉
            x, y = line.strip().split('\t')
            my_list_x.append(x)
            my_list_y.append(y)

    return my_list_x, my_list_y
```

#### 4.2 构建自己的数据源DataSet

目的：

```properties
使用Pytorch框架，一般遵从一个规矩：使用DataSet方法构造数据源，来让模型进行使用
构造数据源的过程中:必须继承torch.utils.data.Dataset类，必须构造两个魔法方法：__len__(), __getitem__()
__len__(): 一般返回的是样本的总个数，我们可以直接len(dataset对象)直接就可以获得结果
__getitem__(): 可以根据某个索引取出样本值，我们可以直接用dataset对象[index]来直接获得结果
```

代码实现：

```python
class NameClassDataset(Dataset):
    def __init__(self, mylist_x, mylist_y):
        self.mylist_x = mylist_x
        self.mylist_y = mylist_y
        self.sample_len = len(mylist_x)

    # 定义魔法方法len
    def __len__(self):
        return self.sample_len

    # 定义魔法方法getitem
    def __getitem__(self, index):
        # 1.index异常值处理
        index = min(max(index, 0), self.sample_len - 1)
        # 2. 根据index取出人名和国家名
        x = self.mylist_x[index]
        # print(f'x--->{x}')
        y = self.mylist_y[index]
        # print(f'y--->{y}')
        # 3.需要对人名进行one-hot编码表示：这里的思路是：针对每个人名组成的单词进行one-hot，然后再拼接
        tensor_x = torch.zeros(len(x), n_letter)
        # print(f'tensor_x-->{tensor_x}')
        for li, letter in enumerate(x):
            tensor_x[li][all_letters.find(letter)] = 1
       # 4.获取标签
       #  print(f'dataset内部的tensor_x--》{tensor_x.shape}')
        tensor_y = torch.tensor(categorys.index(y), dtype=torch.long)
        # print(f'dataset内部的tensor_y-->{tensor_y}')
        return tensor_x, tensor_y
```

#### 4.3 构建数据源Dataloader

目的：

```properties
为了将Dataset我们上一步构建的数据源，进行再次封装，变成一个迭代器，可以进行for循环，而且，可以自动为我们dataset里面的数据进行增维（bath_size）,也可以随机打乱我们的取值顺序
```

代码实现：

```python
filename = './data/name_classfication.txt'
my_list_x, my_list_y = read_data(filename)
mydataset = NameClassDataset(mylist_x=my_list_x, mylist_y=my_list_y)
my_dataloader = DataLoader(dataset=mydataset, batch_size=1, shuffle=True)
```

### 5 模型搭建

#### 5.1 搭建RNN模型

- 注意事项

```properties
RNN模型在实例化的时候，默认batch_first=False，因此，需要小心输入数据的形状
因为: dataloader返回的结果x---》shape--〉[batch_size, seq_len, input_size], 所以课堂上代码和讲义稍微有点不同，讲义是默认的batch_first=False，而我们的代码是batch_first=True，这样做的目的，可以直接承接x的输入。
```

- 代码实现

```python
class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, ouput_size, num_layers=1):
        super().__init__()
        # input_size 代表词嵌入维度；
        self.input_size = input_size
        # hidden_size代表RNN隐藏层维度
        self.hidden_size = hidden_size
        # output_size代表：国家种类个数
        self.ouput_size = ouput_size
        self.num_layers = num_layers
        # 定义RNN网络层
        # 和讲义不一样，我设定了batch_first=True,意味着rnn接受的input第一个参数是batch_size
        self.rnn = nn.RNN(self.input_size, self.hidden_size,
                          num_layers=self.num_layers, batch_first=True)
        # 定义输出网络层
        self.linear = nn.Linear(self.hidden_size, self.ouput_size)

        # 定义softmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        # input的shape---》[batch_size, seq_len, input_size] [1, 9, 57]
        # hidden的shape---》[num_layers, batch_size, hidden_size] [1,1,128]

        # 将input和hidden送入RNN模型得到结果rnn_output【1,9,128】,rnn_hn[1,1,128]
        rnn_output, rnn_hn = self.rnn(input, hidden)
        # print(f'rnn_output--》{rnn_output.shape}')
        # temp:[1, 128]
        tmep = rnn_output[0][-1].unsqueeze(0)
        # print(f'tmep--》{tmep.shape}')
        # 将临时tmep：代表当前样本最后一词的隐藏层输出结果[1, 18]
        output = self.linear(tmep)
        # print(f'output--》{output.shape}')
        # 经过softmax
        return self.softmax(output), rnn_hn

    def inithidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)
```

RNN模型测试

```python
def test_RNN():
    # 1.得到数据
    my_dataloader = get_dataloader()
    # 2.实例化模型
    input_size = n_letter # 57
    hidden_size = 128 # 自定设定RNN模型输出结果维度
    output_size = len(categorys) # 18
    my_rnn = MyRNN(input_size, hidden_size, output_size)
    h0 = my_rnn.inithidden()
    # 3.将数据送入模型
    for i, (x, y) in enumerate(my_dataloader):
        print(f'x--->{x.shape}')
        output, hn = my_rnn(input=x, hidden=h0)
        print(f'output模型输出结果-->{output.shape}')
        print(f'hn-->{hn.shape}')
        break
```

#### 5.2 搭建LSTM模型

- 注意事项

```properties
LSTM模型在实例化的时候，默认batch_first=False，因此，需要小心输入数据的形状
因为: dataloader返回的结果x---》shape--〉[batch_size, seq_len, input_size], 所以课堂上代码和讲义稍微有点不同，讲义是默认的batch_first=False，而我们的代码是batch_first=True，这样做的目的，可以直接承接x的输入。
```

- 代码实现

```python
class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, ouput_size, num_layers=1):
        super().__init__()
        # input_size 代表词嵌入维度；
        self.input_size = input_size
        # hidden_size代表RNN隐藏层维度
        self.hidden_size = hidden_size
        # output_size代表：国家种类个数
        self.ouput_size = ouput_size
        self.num_layers = num_layers
        # 定义LSTM网络层
        # 和讲义不一样，我设定了batch_first=True,意味着rnn接受的input第一个参数是batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)
        # 定义输出网络层
        self.linear = nn.Linear(self.hidden_size, self.ouput_size)

        # 定义softmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden, c0):
        # input的shape---》[batch_size, seq_len, input_size] [1, 9, 57]
        # hidden的shape---》[num_layers, batch_size, hidden_size] [1,1,128]

        # 将input和hidden送入RNN模型得到结果rnn_output【1,9,128】,rnn_hn[1,1,128]
        lstm_output, (lstm_hn, lstm_cn) = self.lstm(input, (hidden, c0))
        # print(f'rnn_output--》{rnn_output.shape}')
        # temp:[1, 128]
        tmep = lstm_output[0][-1].unsqueeze(0)
        # print(f'tmep--》{tmep.shape}')
        # 将临时tmep：代表当前样本最后一词的隐藏层输出结果[1, 18]
        output = self.linear(tmep)
        # print(f'output--》{output.shape}')
        # 经过softmax
        return self.softmax(output), lstm_hn, lstm_cn

    def inithidden(self):
        h0 = torch.zeros(self.num_layers, 1, self.hidden_size)
        c0 = torch.zeros(self.num_layers, 1, self.hidden_size)
        return h0, c0
```

LSTM测试

```python
def test_LSTM():
    # 1.得到数据
    my_dataloader = get_dataloader()
    # 2.实例化模型
    input_size = n_letter # 57
    hidden_size = 128 # 自定设定LSTM模型输出结果维度
    output_size = len(categorys) # 18
    my_lstm = MyLSTM(input_size, hidden_size, output_size)
    h0, c0 = my_lstm.inithidden()
    # 3.将数据送入模型
    for i, (x, y) in enumerate(my_dataloader):
        print(f'x--->{x.shape}')
        output, hn, cn = my_lstm(input=x, hidden=h0, c0=c0)
        print(f'output模型输出结果-->{output.shape}')
        print(f'hn-->{hn.shape}')
        print(f'cn-->{cn.shape}')
        break
```

#### 5.3 搭建GRU模型

- 注意事项

```properties
GRU模型在实例化的时候，默认batch_first=False，因此，需要小心输入数据的形状
因为: dataloader返回的结果x---》shape--〉[batch_size, seq_len, input_size], 所以课堂上代码和讲义稍微有点不同，讲义是默认的batch_first=False，而我们的代码是batch_first=True，这样做的目的，可以直接承接x的输入。
```

- 代码实现

```python
class MyGRU(nn.Module):
    def __init__(self, input_size, hidden_size, ouput_size, num_layers=1):
        super().__init__()
        # input_size 代表词嵌入维度；
        self.input_size = input_size
        # hidden_size代表RNN隐藏层维度
        self.hidden_size = hidden_size
        # output_size代表：国家种类个数
        self.ouput_size = ouput_size
        self.num_layers = num_layers
        # 定义GRU网络层
        # 和讲义不一样，我设定了batch_first=True,意味着rnn接受的input第一个参数是batch_size
        self.gru = nn.GRU(self.input_size, self.hidden_size,
                          num_layers=self.num_layers, batch_first=True)
        # 定义输出网络层
        self.linear = nn.Linear(self.hidden_size, self.ouput_size)

        # 定义softmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        # input的shape---》[batch_size, seq_len, input_size] [1, 9, 57]
        # hidden的shape---》[num_layers, batch_size, hidden_size] [1,1,128]

        # 将input和hidden送入RNN模型得到结果rnn_output【1,9,128】,rnn_hn[1,1,128]
        gru_output, gru_hn = self.gru(input, hidden)
        # print(f'rnn_output--》{rnn_output.shape}')
        # temp:[1, 128]
        tmep = gru_output[0][-1].unsqueeze(0)
        # print(f'tmep--》{tmep.shape}')
        # 将临时tmep：代表当前样本最后一词的隐藏层输出结果[1, 18]
        output = self.linear(tmep)
        # print(f'output--》{output.shape}')
        # 经过softmax
        return self.softmax(output), gru_hn

    def inithidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)
```

GRU测试

```python
def test_GRU():
    # 1.得到数据
    my_dataloader = get_dataloader()
    # 2.实例化模型
    input_size = n_letter # 57
    hidden_size = 128 # 自定设定RNN模型输出结果维度
    output_size = len(categorys) # 18
    my_gru = MyGRU(input_size, hidden_size, output_size)
    # 2.1 初始化参数
    h0 = my_gru.inithidden()
    # 3.将数据送入模型
    for i, (x, y) in enumerate(my_dataloader):
        print(f'x--->{x.shape}')
        output, hn = my_gru(input=x, hidden=h0)
        print(f'output模型输出结果-->{output.shape}')
        print(f'hn-->{hn.shape}')
        break
```

