# 导入相关模块
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torchsummary import summary
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time


# todo:1-构建数据集
def create_dataset():
	print('===========================构建张量数据集对象===========================')
	# todo:1-1 加载csv文件数据集
	data = pd.read_csv('data/手机价格预测.csv')
	print('data.head()->', data.head())
	print('data.shape->', data.shape)
	# todo:1-2 获取x特征列数据集和y目标列数据集
	# iloc属性 下标取值
	x, y = data.iloc[:, :-1], data.iloc[:, -1]
	# 将特征列转换成浮点类型
	x = x.astype(np.float32)
	print('x->', x.head())
	print('y->', y.head())
	# todo:1-3 数据集分割 8:2
	x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8, random_state=88)
	# todo:1-4 数据集转换成张量数据集
	# x_train,y_train类型是df对象, df不能直接转换成张量对象
	# x_train.values():获取df对象的数据值, 得到numpy数组
	# torch.tensor(): numpy数组对象转换成张量对象
	train_dataset = TensorDataset(torch.tensor(data=x_train.values), torch.tensor(data=y_train.values))
	valid_dataset = TensorDataset(torch.tensor(data=x_valid.values), torch.tensor(data=y_valid.values))
	# todo:1-5 返回训练数据集, 测试数据集, 特征数, 类别数
	# shape->(行数, 列数) [1]->元组下标取值
	# np.unique()->去重 len()->去重后的长度 类别数
	print('x.shape[1]->', x.shape[1])
	print('len(np.unique(y)->', len(np.unique(y)))
	return train_dataset, valid_dataset, x.shape[1], len(np.unique(y))


# todo:2-构建神经网络分类模型
class PhonePriceModel(nn.Module):
	print('===========================构建神经网络分类模型===========================')

	# todo:2-1 构建神经网络  __init__()
	def __init__(self, input_dim, output_dim):
		# 继承父类的构造方法
		super().__init__()
		# 第一层隐藏层
		self.linear1 = nn.Linear(in_features=input_dim, out_features=128)
		# 第二层隐藏层
		self.linear2 = nn.Linear(in_features=128, out_features=256)
		# 输出层
		self.output = nn.Linear(in_features=256, out_features=output_dim)

	# todo:2-2 前向传播方法 forward()
	def forward(self, x):
		# 第一层隐藏层计算
		x = torch.relu(input=self.linear1(x))
		# 第二层隐藏层计算
		x = torch.relu(input=self.linear2(x))
		# 输出层计算
		# 没有进行softmax激活计算, 后续创建损失函数时CrossEntropyLoss=softmax+损失计算
		output = self.output(x)
		return output


# todo:3-模型训练
def train(train_dataset, input_dim, class_num):
	print('===========================模型训练===========================')
	# todo:3-1 创建数据加载器 批量训练
	dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
	# todo:3-2 创建神经网络分类模型对象, 初始化w和b
	model = PhonePriceModel(input_dim=input_dim, output_dim=class_num)
	print("======查看模型参数w和b======")
	for name, parameter in model.named_parameters():
		print(name, parameter)
	# todo:3-3 创建损失函数对象 多分类交叉熵损失=softmax+损失计算
	criterion = nn.CrossEntropyLoss()
	# todo:3-4 创建优化器对象 SGD
	optimizer = optim.SGD(params=model.parameters(), lr=1e-3)
	# todo:3-5 模型训练 min-batch 随机梯度下降
	# 训练轮数
	num_epoch = 50
	for epoch in range(num_epoch):
		# 定义变量统计每次训练的损失值, 训练batch数
		total_loss = 0.0
		batch_num = 0
		# 训练开始的时间
		start = time.time()
		# 批次训练
		for x, y in dataloader:
			# 切换模型模式
			model.train()
			# 模型预测 y预测值
			y_pred = model(x)
			# print('y_pred->', y_pred)
			# 计算损失值
			loss = criterion(y_pred, y)
			# print('loss->', loss)
			# 梯度清零
			optimizer.zero_grad()
			# 计算梯度
			loss.backward()
			# 更新参数 梯度下降法
			optimizer.step()
			# 统计每次训练的所有batch的平均损失值和和batch数
			# item(): 获取标量张量的数值
			total_loss += loss.item()
			batch_num += 1
		# 打印损失变换结果
		print('epoch: %4s loss: %.2f, time: %.2fs' % (epoch + 1, total_loss / batch_num, time.time() - start))
	# todo:3-6 模型保存, 将模型参数保存到字典, 再将字典保存到文件
	torch.save(model.state_dict(), 'model/phone.pth')


# todo:4-模型评估
def test(valid_dataset, input_dim, class_num):
	# todo:4-1 创建神经网络分类模型对象
	model = PhonePriceModel(input_dim=input_dim, output_dim=class_num)
	# todo:4-2 加载训练模型的参数字典
	model.load_state_dict(torch.load(f='model/phone.pth'))
	# todo:4-3 创建测试集数据加载器
	# shuffle: 不需要为True, 预测, 不是训练
	dataloader = DataLoader(dataset=valid_dataset, batch_size=8, shuffle=False)
	# todo:4-4 定义变量, 初始值为0, 统计预测正确的样本个数
	correct = 0
	# todo:4-5 按batch进行预测
	for x, y in dataloader:
		print('y->', y)
		# 切换模型模式为预测模式
		model.eval()
		# 模型预测 y预测值 -> 输出层的加权求和值
		output = model(x)
		print('output->', output)
		# 根据加权求和值得到类别, argmax() 获取最大值对应的下标就是类别 y->0,1,2,3
		# dim=1:一行一行处理, 一个样本一个样本
		y_pred = torch.argmax(input=output, dim=1)
		print('y_pred->', y_pred)
		# 统计预测正确的样本个数
		print(y_pred == y)
		# 对布尔值求和, True->1 False->0
		print((y_pred == y).sum())
		correct += (y_pred == y).sum()
		print('correct->', correct)
	# 计算预测精度 准确率
	print('Acc: %.5f' % (correct.item() / len(valid_dataset)))


if __name__ == '__main__':
	# 创建张量数据集对象
	train_dataset, valid_dataset, input_dim, class_num = create_dataset()
	# 创建模型对象
	# model = PhonePriceModel(input_dim=input_dim, output_dim=class_num)
	# 计算模型参数
	# input_size: 输入层样本形状
	# summary(model, input_size=(16, input_dim))
	# 模型训练
	# train(train_dataset=train_dataset, input_dim=input_dim, class_num=class_num)
	# 模型评估
	test(valid_dataset=valid_dataset, input_dim=input_dim, class_num=class_num)
