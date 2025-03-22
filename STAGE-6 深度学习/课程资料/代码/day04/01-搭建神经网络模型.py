import torch
import torch.nn as nn  # 线性模型和初始化方法
from torchsummary import summary


# todo:1-创建类继承 nn.module类
class ModelDemo(nn.Module):
	# todo:2-定义__init__构造方法, 构建神经网络
	def __init__(self):
		# todo:2-1 调用父类的__init__方法
		super().__init__()
		# todo:2-2 创建隐藏层和输出层  定义属性
		# in_features: 输入特征数(上一层神经元个数)
		# out_features: 输出特征数(当前层神经元个数)
		self.linear1 = nn.Linear(in_features=3, out_features=3)
		self.linear2 = nn.Linear(in_features=3, out_features=2)
		self.output = nn.Linear(in_features=2, out_features=2)
		# todo:2-3 对隐藏层进行参数初始化
		# self.linear1.weight: 在类的内部调用对象属性
		nn.init.xavier_normal_(tensor=self.linear1.weight)
		nn.init.zeros_(tensor=self.linear1.bias)
		nn.init.kaiming_normal_(tensor=self.linear2.weight, nonlinearity='relu')
		nn.init.zeros_(tensor=self.linear2.bias)

	# todo:3-定义前向传播方法 forward(方法名固定) 得到预测y值
	def forward(self, x):  # x->输入样本的特征值
		# todo:3-1 第一层计算 加权求和值计算  激活值计算
		x = torch.sigmoid(input=self.linear1(x))
		# todo:3-2 第二层计算
		x = torch.relu(input=self.linear2(x))
		# todo:3-3 输出层计算  假设多分类问题
		# dim=-1: 按行计算, 一个样本一个样本算
		x = torch.softmax(input=self.output(x), dim=-1)
		# 返回预测值
		return x


# 创建模型预测函数
def train():
	# todo:1-创建神经网络模型对象
	my_model = ModelDemo()
	print('my_model->', my_model)
	# todo:2-构造数据集样本, 随机生成
	data = torch.randn(size=(5, 3))
	print('data->', data)
	print('data.shape->', data.shape)
	print('data.requires_grad->', data.requires_grad)
	# todo:3-调用神经网络模型对象进行模型训练
	output = my_model(data)
	print('output->', output)
	print('output.shape->', output.shape)
	print('output.requires_grad->', output.requires_grad)
	# todo:4-计算和查看模型参数
	print(('====================计算和查看模型参数==================='))
	# input_size: 样本的特征数
	# batch_size: 批量训练的样本数
	# summary(model=my_model, input_size=(3,), batch_size=5)
	summary(model=my_model, input_size=(5, 3))
	for name, param in my_model.named_parameters():
		print('name->', name)
		print('param->', param)


if __name__ == '__main__':
	train()
