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
bert_toekenizer = BertTokenizer.from_pretrained('../dm06_transformers/model/bert-base-chinese')

#加载预训练模型
bert_model = BertModel.from_pretrained("../dm06_transformers/model/bert-base-chinese")

bert_model = bert_model.to(device)
# print(bert_model)

# device = "cuda" if torch.cuda.is_available() else "cpu"



def dm_loaddataset():
    # 加载训练集
    # 当没有注明split="train",返回的结果类型是：DatasetDict样式
    # 当注明split="train",返回的结果类型是：Dataset样式
    train_dataset = load_dataset('csv', data_files='./data/train.csv', split='train')
    # print(f'train_dataset-->{train_dataset}')
    # print(f'当前数据的长度--》{len(train_dataset)}')
    # print(f'根据索引取出某个元素--》{train_dataset[0]}')
    # print(f'根据切片取出多个元素--》{train_dataset[:3]}')

    # 加载验证
    dev_dataset = load_dataset('csv', data_files='./data/validation.csv', split='train')
    print(f'dev_dataset--》{len(dev_dataset)}')
    print(f'dev_dataset1--》{dev_dataset[0]}')
    print(f'dev_dataset2--》{dev_dataset[:3]}')
    # 加载测试集
    test_dataset = load_dataset('csv', data_files='./data/test.csv', split='train')
    # print(f'test_dataset--》{len(test_dataset)}')
    # print(f'test_dataset1--》{test_dataset[0]}')
    # print(f'test_dataset2--》{test_dataset[:3]}')

# 自定义函数
def collate_fn1(data):
    # data是个列表样式：[{label:0, text:...}, {label:1, text:...}, ...]
    # collate_fn1自定义函数在调用dataloader时，自动会被使用，batch_size等于几，自动传递几个样本（包含标签和文本）
    # print(f'自定义函数得到的参数data-->{data}')
    # print(f'data的长度-->{len(data)}')
    # 取出对应数据的text以及标签
    seqs = [value["text"] for value in data]
    labels = [value["label"] for value in data]
    # print(f'labels-->{labels}')
    # 1.对一个批次的样本进行编码，实现张量的表示
    inputs = bert_toekenizer.batch_encode_plus(seqs,
                                               truncation=True,
                                               padding="max_length",
                                               max_length=300,
                                               return_tensors='pt',
                                               )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]
    labels = torch.tensor(labels, dtype=torch.long)
    return input_ids, attention_mask, token_type_ids, labels

# 测试dataset
def dm_test_dataset():
    # 1.加载训练集的dataset对象
    train_dataset = load_dataset('csv', data_files='./data/train.csv', split="train")
    print(f'train_dataset--》{len(train_dataset)}')

    # 2.实例化dataloader对象
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=8,
                                  shuffle=True,
                                  drop_last=True,
                                  collate_fn=collate_fn1)
    # 3.for循环dataloader，查验collate_fn1函数的内部逻辑
    for input_ids, attention_mask, token_type_ids, labels in train_dataloader:
        print(f'input_ids--》{input_ids.shape}')
        print(f'attention_mask--》{attention_mask.shape}')
        print(f'token_type_ids--》{token_type_ids.shape}')
        print(f'labels--》{labels.shape}')
        # print('test')
        break


# 自定义下游任务的模型
class ClsModel(nn.Module):
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




# 测试模型
def test_model():
    # 1.加载训练数据集:dataset
    train_dataset = load_dataset('csv', data_files='./data/train.csv', split='train')
    # 2.实例化dataloader对象
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=8,
                                  shuffle=True,
                                  collate_fn=collate_fn1,
                                  drop_last=True)

    # 3.实例化model
    cls_model = ClsModel()
    # 4.遍历数据送入模型
    for input_ids, attention_mask, token_type_ids, labels in train_dataloader:
        print(f'input_ids--》{input_ids.shape}')
        output = cls_model(input_ids, attention_mask, token_type_ids)
        print(f'output--》{output.shape}')
        break

# 定义模型训练函数
def model2train():
    # 1. 获取训练数据：dataset
    train_dataset = load_dataset('csv', data_files='./data/train.csv', split='train')
    # 2. 实例化模型
    cls_model = ClsModel()
    cls_model = cls_model.to(device)
    # 3. 实例化优化器
    cls_adamw = AdamW(cls_model.parameters(), lr=5e-4)
    # 4. 实例化损失函数对象
    cls_entropy = nn.CrossEntropyLoss()
    # 5. 对预训练模型参数不进行更新的另外一种写法
    for param in bert_model.parameters():
        param.requires_grad_(False)
    # 6. 定义当前模型为训练模式：因为添加了预训练模型的结果，
    cls_model.train()
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
                                      collate_fn=collate_fn1,
                                      drop_last=True)
        # 开始内部迭代
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(tqdm(train_dataloader),start=1):
            # 将input_ids, attention_mask, token_type_ids送入模型，得到预测结果
            # output-->【8，2】
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)
            output = cls_model(input_ids, attention_mask, token_type_ids)
            # 计算损失
            cls_loss = cls_entropy(output, labels)
            # 梯度清零
            cls_adamw.zero_grad()
            # 反向传播
            cls_loss.backward()
            # 梯度更新
            cls_adamw.step()
            # 打印日志
            # 每隔5步打印一下
            if i % 5 == 0:
                # 求出预测结果中[8,2]-->8个样本最大概率值对应的索引
                # print(f'output--》{output}')
                temp_idx = torch.argmax(output, dim=-1)
                # 得到准确率
                avg_acc = (temp_idx == labels).sum().item() / len(labels)
                use_time = time.time() - start_time
                print(f'当前轮次：%d, 损失：%.3f, 准确率：%.3f, 时间：%d'%(epoch_idx+1, cls_loss.item(),avg_acc, use_time))

        torch.save(cls_model.state_dict(), './save_model/cls_model_%d.bin'%(epoch_idx+1))


# 定义模型评估（测试）函数
def model2dev():
    # 1.加载测试数据dataset:
    test_dataset = load_dataset('csv', data_files='./data/test.csv', split='train')
    # 2. 实例化dataloader
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=8,
                                 shuffle=True,
                                 drop_last=True,
                                 collate_fn=collate_fn1)
    # 3. 加载训练好的模型的参数
    cls_model = ClsModel()
    cls_model.load_state_dict(torch.load('./save_model/cls_model_1.bin'))
    # 4. 设置模型为评估模式
    cls_model.eval()
    # 5. 定义参数；
    correct = 0 # 预测正确的样本个数
    total = 0 # 已经预测的样本个数
    # 6. 开始预测
    with torch.no_grad():
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(tqdm(test_dataloader), start=1):
            # print(f'input_ids--》{input_ids}')
            output = cls_model(input_ids, attention_mask, token_type_ids)
            # 求出预测结果中[8,2]-->8个样本最大概率值对应的索引
            # print(f'output--》{output}')
            temp_idx = torch.argmax(output, dim=-1) # 【8】
            # 得到准确率
            correct = correct + (temp_idx == labels).sum().item()
            total = total + len(labels)
            # 每隔5步，打印平均准确率，还随机选择一个样本人工查验模型效果
            if i % 5 == 0:
                avg_acc = correct / total
                # 选择第一个样本进行展示
                text = bert_toekenizer.decode(input_ids[0], skip_special_tokens=True)
                # 获得第一个样本预测的结果
                predict = temp_idx[0].item()
                label = labels[0].item()
                print(avg_acc,"|", text,"|", predict,"|", label)


if __name__ == '__main__':
    # dm_loaddataset()
    # dm_test_dataset()
    # test_model()
    # model2train()
    model2dev()