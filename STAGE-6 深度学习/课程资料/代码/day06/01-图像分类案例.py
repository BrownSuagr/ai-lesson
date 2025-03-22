import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from torchsummary import summary

# 每批次样本数
BATCH_SIZE = 8


# todo: 1-加载数据集转换成张量数据集
def create_dataset():
	# root: 文件夹所在目录路径
	# train: 是否加载训练集
	# ToTensor(): 将图片数据转换成张量数据
	train_dataset = CIFAR10(root='./data', train=True, transform=ToTensor())
	valid_dataset = CIFAR10(root='./data', train=False, transform=ToTensor())
	return train_dataset, valid_dataset


# todo: 2-构建卷积神经网络分类模型
class ImageModel(nn.Module):
	# todo:2-1 构建init构造函数, 实现搭建神经网络
	def __init__(self):
		super().__init__()
		# 第1层卷积层
		# 输入通道3
		# 输出通道6 6个神经元提取出6张特征图
		# 卷积核大小3
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)
		# 第1层池化层
		# 窗口大小2*2
		# 步长2
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		# 第2层卷积层
		self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
		# 第2层池化层
		# 池化层输出的特征图 16*6*6
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		# 第1层隐藏层
		# in_features: 将最后池化层的16*6*6三维矩阵转换为一维矩阵
		# 一维矩阵就是池化层这图像
		self.linear1 = nn.Linear(in_features=16 * 6 * 6, out_features=120)
		# 第2层隐藏层
		self.linear2 = nn.Linear(in_features=120, out_features=84)
		# 输出层
		# out_features: 10, 10分类问题
		self.out = nn.Linear(in_features=84, out_features=10)

	# todo:2-2 构建forward函数, 实现前向传播
	def forward(self, x):
		# 第1层 卷积+激活+池化 计算
		x = self.pool1(torch.relu(self.conv1(x)))
		# 第2层 卷积+激活+池化 计算
		# x->(8, 16, 6, 6) 8个样本, 每个样本是16*6*6
		x = self.pool2(torch.relu(self.conv2(x)))
		# print('x->', x.shape)
		# 第1层隐藏层  只能接收二维数据集
		# 四维数据集转换成二维数据集
		# x.shape[0]: 每批样本数, 最后一批可能不够8个, 所以不是写死8
		#  -1*8=8*16*6*6 -1=16*6*6=576
		x = x.reshape(shape=(x.shape[0], -1))
		# print('x->', x.shape)
		x = torch.relu(self.linear1(x))
		# 第2层隐藏层
		x = torch.relu(self.linear2(x))
		# 输出层 没有使用softmax激活函数, 后续多分类交叉熵损失函数会自动进行softmax激活
		x = self.out(x)
		return x


# todo: 3-模型训练
def train(train_dataset):
	# 创建数据加载器
	dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
	# 创建模型对象
	model = ImageModel()
	# 创建损失函数对象
	criterion = nn.CrossEntropyLoss()
	# 创建优化器对象
	optimizer = optim.Adam(model.parameters(), lr=1e-3)
	# 循环遍历epoch
	# 定义epoch变量
	epoch = 10
	for epoch_idx in range(epoch):
		# 定义总损失变量
		total_loss = 0.0
		# 定义预测正确样本个数变量
		total_correct = 0
		# 定义总样本数据变量
		total_samples = 0
		# 定义开始时间变量
		start = time.time()
		# 循环遍历数据加载器 min-batch
		for x, y in dataloader:
			# print('y->', y)
			# 切换训练模式
			model.train()
			# 模型预测y
			output = model(x)
			# print('output->', output)
			# 计算损失值 平均损失值
			loss = criterion(output, y)
			# print('loss->', loss)
			# 梯度清零
			optimizer.zero_grad()
			# 梯度计算
			loss.backward()
			# 参数更新
			optimizer.step()
			# 统计预测正确的样本个数
			# tensor([9, 9, 9, 9, 9, 9, 9, 9])
			# print(torch.argmax(output, dim=-1))
			# tensor([False, False, False, False, False, False, False, False])
			# print(torch.argmax(output, dim=-1) == y)
			# tensor(0)
			# print((torch.argmax(output, dim=-1) == y).sum())
			total_correct += (torch.argmax(output, dim=-1) == y).sum()
			# 统计当前批次的总损失值
			# loss.item(): 当前批次平均损失值
			total_loss += loss.item() * len(y)
			# 统计当前批次的样本数
			total_samples += len(y)
		end = time.time()
		print('epoch:%2s loss:%.5f acc:%.2f time:%.2fs' % (
			epoch_idx + 1, total_loss / total_samples, total_correct / total_samples, end - start))
	# 保存训练模型
	torch.save(obj=model.state_dict(), f='model/imagemodel.pth')


# todo: 4-模型评估
def test(valid_dataset):
	# 创建测试集数据加载器
	dataloader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
	# 创建模型对象, 加载训练模型参数
	model = ImageModel()
	model.load_state_dict(torch.load('model/imagemodel.pth'))
	# 定义统计预测正确样本个数变量 总样本数据变量
	total_correct = 0
	total_samples = 0
	# 遍历数据加载器
	for x, y in dataloader:
		# 切换推理模型
		model.eval()
		# 模型预测
		output = model(x)
		# 将预测分值转成类别
		y_pred = torch.argmax(output, dim=-1)
		print('y_pred->', y_pred)
		# 统计预测正确的样本个数
		total_correct += (y_pred == y).sum()
		# 统计总样本数
		total_samples += len(y)

	# 打印精度
	print('Acc: %.2f' % (total_correct / total_samples))

if __name__ == '__main__':
	train_dataset, valid_dataset = create_dataset()
	# print('图片类别对应关系->', train_dataset.class_to_idx)
	# print('train_dataset->', train_dataset.data[0])
	# print('train_dataset->', train_dataset.data[0].shape)
	# print('train_dataset.data.shape->', train_dataset.data.shape)
	# print('valid_dataset.data.shape->', valid_dataset.data.shape)
	# print('train_dataset.targets->', train_dataset.targets[0])
	# # 图像展示
	# plt.figure(figsize=(2, 2))
	# plt.imshow(train_dataset.data[1])
	# plt.title(train_dataset.targets[1])
	# plt.show()
	# model = ImageModel()
	# summary(model, input_size=(3, 32, 32))
	# 模型训练
	# train(train_dataset)
	# 模型预测
	test(valid_dataset)
