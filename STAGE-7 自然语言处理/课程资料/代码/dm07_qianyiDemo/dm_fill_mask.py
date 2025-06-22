# coding:utf-8
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from transformers import AdamW
import torch.optim as optim
import torch
import time
import torch.nn as nn
from  tqdm import tqdm
# device = "mps"
device = "cpu"
#加载tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('../dm06_transformers/model/bert-base-chinese')
# print('1', bert_tokenizer.vocab_size)
#加载预训练模型
bert_model = BertModel.from_pretrained("../dm06_transformers/model/bert-base-chinese")
bert_model = bert_model.to(device)
# print(bert_model)

def collate_fn2(data):
    sents = [value["text"] for value in data]
    # 对一个批次的样本进行编码
    inputs = bert_tokenizer.batch_encode_plus(sents,
                                              truncation=True,
                                              padding="max_length",
                                              max_length=32,
                                              return_tensors='pt')
    # print(f'inputs--》{inputs}')
    input_ids = inputs["input_ids"] # [8, 32]
    # print(f'input_ids--》{input_ids}')
    attention_mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]
    # 在案例里面，选择的是每个样本的第16个位置信息，进行label选择并mask
    labels = input_ids[:, 16].reshape(-1).clone()
    # 将input_ids中第16个位置，替换为[MASK]对应的id
    # input_ids[:, 16] = bert_tokenizer.mask_token_id
    # bert_tokenizer.mask_token->[MASK];.get_vocab()是词表字典
    input_ids[:, 16] = bert_tokenizer.get_vocab()[bert_tokenizer.mask_token]
    labels = torch.tensor(labels, dtype=torch.long)
    # print(f'替换之后的结果input_ids-->{input_ids}')
    return input_ids, attention_mask, token_type_ids, labels


# 定义测试dataset函数
def test_dataset():
    # 第一步加载数据：train_dataset
    original_dataset = load_dataset('csv', data_files='./data/train.csv', split="train")
    print(f'original_dataset--》{original_dataset}')
    # 第二步过滤出text长度大于32的样本
    train_dataset = original_dataset.filter(lambda x: len(x["text"]) > 32)
    # print(f'train_dataset--》{train_dataset}')
    # print(f'train_dataset--》{train_dataset[3]}')
    # 第三步：实例化dataloader对象
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=8,
                                  shuffle=True,
                                  drop_last=True,
                                  collate_fn=collate_fn2)
    for input_ids, attention_mask, token_type_ids, labels in train_dataloader:
        print(f'input_ids--》{input_ids}')
        print('第一个样本：', bert_tokenizer.decode(input_ids[0]))
        print(f'labels--》{labels}')
        print('第一个样本：', bert_tokenizer.decode(labels[0]))
        break


# 自定义模型
class FMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(768, bert_tokenizer.vocab_size)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 将数据送入预训练模型，得到特征向量表示，在这里我们不更新预训练模型的参数
        with torch.no_grad():
            bert_output = bert_model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids)

        # last_hidden_state-->[8, 32, 768]
        # 取出bert_output第16个位置的词向量信息:[8, 768]
        last_hidden_state = bert_output.last_hidden_state
        # print(f'last_hidden_state--》{last_hidden_state.shape}')
        result = self.linear(last_hidden_state[:, 16])
        return result

def test_model():
    # 第一步加载数据：train_dataset
    original_dataset = load_dataset('csv', data_files='./data/train.csv', split="train")
    print(f'original_dataset--》{original_dataset}')
    # 第二步过滤出text长度大于32的样本
    train_dataset = original_dataset.filter(lambda x: len(x["text"]) > 32)
    # print(f'train_dataset--》{train_dataset}')
    # print(f'train_dataset--》{train_dataset[3]}')
    # 第三步：实例化dataloader对象
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=8,
                                  shuffle=True,
                                  drop_last=True,
                                  collate_fn=collate_fn2)
    # 第四步：实例化模型
    fm_model = FMModel()
    # 第五步：将数据送入模型
    for input_ids, attention_mask, token_type_ids, labels in train_dataloader:
        print(f'input_ids-->{input_ids.shape}')
        output = fm_model(input_ids, attention_mask, token_type_ids)
        print(f'output--》{output.shape}')
        break


# 定义模型训练函数
def model2train():
    # 1. 获取训练数据：dataset
    original_dataset = load_dataset('csv', data_files='./data/train.csv', split='train')
    # 过滤数据：长度大于32
    train_dataset = original_dataset.filter(lambda x: len(x["text"]) > 32)
    # 2. 实例化模型
    fm_model = FMModel()
    fm_model = fm_model.to(device)
    # 3. 实例化优化器
    fm_adamw = AdamW(fm_model.parameters(), lr=5e-4)
    # 4. 实例化损失函数对象
    fm_entropy = nn.CrossEntropyLoss()
    # 5. 对预训练模型参数不进行更新的另外一种写法
    for param in bert_model.parameters():
        param.requires_grad_(False)
    # 6. 定义当前模型为训练模式：因为添加了预训练模型的结果，
    fm_model.train()
    # 7.定义epochs
    epochs = 3
    # 8. 开始训练
    for epoch_idx in range(epochs):
        # 定义变量
        start_time = time.time()
        # 实例化dataloader
        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=8,
                                      shuffle=True,
                                      collate_fn=collate_fn2,
                                      drop_last=True)
        # 开始内部迭代
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(tqdm(train_dataloader),start=1):
            # 将input_ids, attention_mask, token_type_ids送入模型，得到预测结果
            # output-->【8，2】
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)
            output = fm_model(input_ids, attention_mask, token_type_ids)
            # 计算损失
            fm_loss = fm_entropy(output, labels)
            # 梯度清零
            fm_adamw.zero_grad()
            # 反向传播
            fm_loss.backward()
            # 梯度更新
            fm_adamw.step()
            # 打印日志
            # 每隔5步打印一下
            if i % 5 == 0:
                # 求出预测结果中[8,2]-->8个样本最大概率值对应的索引
                # print(f'output--》{output}')
                temp_idx = torch.argmax(output, dim=-1)
                # 得到准确率
                avg_acc = (temp_idx == labels).sum().item() / len(labels)
                use_time = time.time() - start_time
                print(f'当前轮次：%d, 损失：%.3f, 准确率：%.3f, 时间：%d'%(epoch_idx+1, fm_loss.item(),avg_acc, use_time))

        torch.save(fm_model.state_dict(), './save_model/fm_model_%d.bin'%(epoch_idx+1))


# 定义模型评估（测试）函数
def model2dev():
    # 1.加载测试数据dataset:
    test_dataset = load_dataset('csv', data_files='./data/test.csv', split='train')
    test_dataset = test_dataset.filter(lambda x: len(x["text"]) > 32)
    # 2. 实例化dataloader
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=8,
                                 shuffle=True,
                                 drop_last=True,
                                 collate_fn=collate_fn2)
    # 3. 加载训练好的模型的参数
    fm_model = FMModel()
    fm_model.load_state_dict(torch.load('./save_model/fm_model_3.bin'))
    # 4. 设置模型为评估模式
    fm_model.eval()
    # 5. 定义参数；
    correct = 0 # 预测正确的样本个数
    total = 0 # 已经预测的样本个数
    # 6. 开始预测
    with torch.no_grad():
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(tqdm(test_dataloader), start=1):
            # print(f'input_ids--》{input_ids}')# [8, 32]
            # print(f'labels--》{labels}')
            output = fm_model(input_ids, attention_mask, token_type_ids)
            # 求出预测结果中[8,2]-->8个样本最大概率值对应的索引
            # print(f'output--》{output}')
            temp_idx = torch.argmax(output, dim=-1) # 【8】
            # print(f'模型预测结果-->{temp_idx}')
            # 得到准确率
            correct = correct + (temp_idx == labels).sum().item()
            total = total + len(labels)
            # 每隔5步，打印平均准确率，还随机选择一个样本人工查验模型效果
            if i % 5 == 0:
                avg_acc = correct / total
                # 选择第一个样本进行展示
                text = bert_tokenizer.decode(input_ids[0])
                # 获得第一个样本预测的结果
                predict_token = bert_tokenizer.decode(temp_idx[0])
                label = bert_tokenizer.decode(labels[0])
                print(avg_acc,"|", text,"|")
                print(predict_token,"|", label)

if __name__ == '__main__':
    # test_dataset()

    # test_model()

    # model2train()

    model2dev()





