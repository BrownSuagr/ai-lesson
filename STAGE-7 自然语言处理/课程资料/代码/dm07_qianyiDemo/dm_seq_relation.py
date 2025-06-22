# coding:utf-8
import random

from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from transformers import AdamW
import torch.optim as optim
import torch
import time
import torch.nn as nn
from  tqdm import tqdm
# windows或linux：使用GPU命令
# device = "cuda" if torch.cuda.is_available() else "cpu"
# M1
device = "cpu"

#加载tokenizer
bert_toekenizer = BertTokenizer.from_pretrained('../dm06_transformers/model/bert-base-chinese')
#加载预训练模型
bert_model = BertModel.from_pretrained("../dm06_transformers/model/bert-base-chinese")
bert_model = bert_model.to(device)


# 自定义Dataset类
class RelationDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        # 使用load_dataset方法，获得初始的dataset对象
        original_dataset = load_dataset('csv', data_files=data_path, split='train')
        # print(f'original_dataset--》{original_dataset}')
        # print(f'original_dataset--》{original_dataset[0]}')
        # 过滤出text文本长度大于44的样本
        self.dataset = original_dataset.filter(lambda x: len(x["text"]) > 44)
        # print(f'self.dataset--》{self.dataset}')
        # 获得样本的总长度
        self.sample_len = len(self.dataset)

    def __len__(self):
        return self.sample_len

    def __getitem__(self, index):
        # 构造中文句子关系任务的样本：sent1,sent2, label
        label = 1
        text = self.dataset[index]["text"]
        # print(f'text--》{text}')
        sent1 = text[:22]
        sent2 = text[22:44]
        # 需要构造负样本
        if random.randint(0, 1) == 0:
            # 随机挑选出一个其他样本(while true的代码是，方式j和index相等)
            # while True:
            #     j = random.randint(0, self.sample_len - 1)
            #     if index != j:
            #         break
            j = random.randint(0, self.sample_len-1)
            sent2 = self.dataset[j]["text"][22:44]
            label = 0
        return sent1, sent2, label


# 自定义函数
def collate_fn3(data):
    # print(f'data-->{len(data)}')
    # print(f'data-->{data[:2]}')
    sents = [value[:2] for value in data]
    # print(f'sents--》{sents}')
    labels = [value[-1] for value in data]
    # print(labels)

    inputs = bert_toekenizer.batch_encode_plus(sents,
                                               padding="max_length",
                                               truncation=True,
                                               max_length=50, # 44+cls+sep+sep+other
                                               return_tensors='pt')
    # print(f'inputs-->{inputs}')
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]
    labels = torch.tensor(labels, dtype=torch.long)
    return input_ids, attention_mask, token_type_ids, labels

def test_dataset():
    rd_dataset = RelationDataset(data_path="./data/train.csv")
    # 使用Dataloader进行再次封装
    train_dataloader = DataLoader(dataset=rd_dataset,
                                  batch_size=8,
                                  shuffle=True,
                                  drop_last=True,
                                  collate_fn=collate_fn3)

    for input_ids, attention_mask, token_type_ids, labels in train_dataloader:
        # print(f'input_ids---》{input_ids}')
        print(f'labels---》{labels}')
        break



# 自定义下游任务的模型
class NSPModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义全连接层：768代表bert预训练模型输出的结果，单词的词嵌入维度，2代表二分类
        self.linear = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 将数据送入预训练模型，得到特征向量表示，在这里我们不更新预训练模型的参数
        with torch.no_grad():
            bert_output = bert_model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids)
        # 如果对句子分类，bert模型原始论文中提出：以CLS这个token的向量，会作为整个句子的特征向量
        # pooler_output-->就是cls的向量，但是拿到last_hidden_state[:, 0]之后，又进行了一次linear,但是形状没有发生改变
        # 直接拿pooler_output这个结果去做下一步处理
        # last_hidden_state = bert_output.last_hidden_state
        # print(f'last_hidden_state--》{last_hidden_state.shape}')
        # print(last_hidden_state[:, 0].shape)
        pooler_output = bert_output.pooler_output
        # print(f'pooler_output--》{pooler_output.shape}')
        # result-->[8,2]
        result = self.linear(pooler_output)
        return result

def test_model():
    rd_dataset = RelationDataset(data_path="./data/train.csv")
    # 使用Dataloader进行再次封装
    train_dataloader = DataLoader(dataset=rd_dataset,
                                  batch_size=8,
                                  shuffle=True,
                                  drop_last=True,
                                  collate_fn=collate_fn3)
    nsp_model = NSPModel()
    for input_ids, attention_mask, token_type_ids, labels in train_dataloader:
        output = nsp_model(input_ids, attention_mask, token_type_ids)
        print(f'output--》{output.shape}')
        break



# 定义模型训练函数
def model2train():
    # 1. 获取训练数据：dataset
    train_dataset = RelationDataset(data_path='./data/train.csv')
    # 2. 实例化模型
    nsp_model = NSPModel()
    nsp_model = nsp_model.to(device)
    # 3. 实例化优化器
    nsp_adamw = AdamW(nsp_model.parameters(), lr=5e-4)
    # 4. 实例化损失函数对象
    nsp_entropy = nn.CrossEntropyLoss()
    # 5. 对预训练模型参数不进行更新的另外一种写法
    for param in bert_model.parameters():
        param.requires_grad_(False)
    # 6. 定义当前模型为训练模式：因为添加了预训练模型的结果，
    nsp_model.train()
    # 7.定义epochs
    epochs = 1
    # 8. 开始训练
    for epoch_idx in range(epochs):
        # 定义变量
        start_time = time.time()
        # 实例化dataloader
        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=8,
                                      shuffle=True,
                                      collate_fn=collate_fn3,
                                      drop_last=True)
        # 开始内部迭代
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(tqdm(train_dataloader),start=1):
            # 将input_ids, attention_mask, token_type_ids送入模型，得到预测结果
            # output-->【8，2】
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)
            output = nsp_model(input_ids, attention_mask, token_type_ids)
            # 计算损失
            nsp_loss = nsp_entropy(output, labels)
            # 梯度清零
            nsp_adamw.zero_grad()
            # 反向传播
            nsp_loss.backward()
            # 梯度更新
            nsp_adamw.step()
            # 打印日志
            # 每隔20步打印一下
            if i % 20 == 0:
                # 求出预测结果中[8,2]-->8个样本最大概率值对应的索引
                # print(f'output--》{output}')
                temp_idx = torch.argmax(output, dim=-1)
                # 得到准确率
                avg_acc = (temp_idx == labels).sum().item() / len(labels)
                use_time = time.time() - start_time
                print(f'当前轮次：%d, 损失：%.3f, 准确率：%.3f, 时间：%d'%(epoch_idx+1, nsp_loss.item(),avg_acc, use_time))

        torch.save(nsp_model.state_dict(), './save_model/nsp_model_%d.bin'%(epoch_idx+1))


# 定义模型评估（测试）函数
def model2dev():
    # 1.加载测试数据dataset:
    test_dataset = RelationDataset(data_path='./data/test.csv')
    # 2. 实例化dataloader
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=8,
                                 shuffle=True,
                                 drop_last=True,
                                 collate_fn=collate_fn3)
    # 3. 加载训练好的模型的参数
    nsp_model = NSPModel()
    nsp_model.load_state_dict(torch.load('./save_model/nsp_model_1.bin'))
    # 4. 设置模型为评估模式
    nsp_model.eval()
    # 5. 定义参数；
    correct = 0 # 预测正确的样本个数
    total = 0 # 已经预测的样本个数
    # 6. 开始预测
    with torch.no_grad():
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(tqdm(test_dataloader), start=1):
            # print(f'input_ids--》{input_ids}')
            output = nsp_model(input_ids, attention_mask, token_type_ids)
            # 求出预测结果中[8,2]-->8个样本最大概率值对应的索引
            # print(f'output--》{output}')
            temp_idx = torch.argmax(output, dim=-1) # 【8】
            # 得到准确率
            correct = correct + (temp_idx == labels).sum().item()
            total = total + len(labels)
            # 每隔5步，打印平均准确率，还随机选择一个样本人工查验模型效果
            if i % 5 == 0:
                avg_acc = correct / total
                print(avg_acc)

if __name__ == '__main__':
    # test_dataset()
    # test_model()
    # model2train()
    model2dev()













