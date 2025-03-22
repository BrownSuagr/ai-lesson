import torch
from torch.utils.data import TensorDataset  # 创建x和y张量数据集对象
from torch.utils.data import DataLoader  # 创建数据集加载器
import torch.nn as nn  # 损失函数和回归函数
from torch.optim import SGD  # 随机梯度下降函数, 取一个训练样本算梯度值
from sklearn.datasets import make_regression  # 创建随机样本, 工作中不使用
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# todo: 1-创建线性回归样本 x y coef(w) b
def create_datasets():
	x, y, coef = make_regression(n_samples=100,  # 样本数
								 n_features=1,  # 特征数
								 noise=10,  # 标准差, 噪声, 样本离散程度
								 coef=True,  # 返回系数, w
								 bias=14.5,  # 截距 b
								 random_state=0)

	# 将数组转换成张量, 避免后续类型问题报错, 可以指定张量元素类型
	x = torch.tensor(data=x, dtype=torch.float32)
	y = torch.tensor(data=y, dtype=torch.float32)
	# print('x->', x)
	# print('y->', y)
	# print('coef->', coef)
	return x, y, coef


# todo: 2-模型训练
def train(x, y, coef):
	# 创建张量数据集对象
	datasets = TensorDataset(x, y)
	print('datasets->', datasets)
	# 创建数据加载器对象
	# dataset: 张量数据集对象
	# batch_size: 每个batch的样本数
	# shuffle: 是否打乱样本
	dataloader = DataLoader(dataset=datasets, batch_size=16, shuffle=True)
	print('dataloader->', dataloader)
	# for batch in dataloader:  # 每次遍历取每个batch样本
	# 	print('batch->', batch)  # [x张量对象, y张量对象]
	# 	break
	# 创建初始回归模型对象, 随机生成w和b, 元素类型为float32
	# in_features: 输入特征数 1个
	# out_features: 输出特征数 1个
	model = nn.Linear(in_features=1, out_features=1)
	print('model->', model)
	# 获取模型对象的w和b参数
	print('model.weight->', model.weight)
	print('model.bias->', model.bias)
	print('model.parameters()->', list(model.parameters()))
	# 创建损失函数对象, 计算损失值
	criterion = nn.MSELoss()
	# 创建SGD优化器对象, 更新w和b
	optimizer = SGD(params=model.parameters(), lr=0.01)
	# 定义变量, 接收训练次数, 损失值, 训练样本数
	epochs = 100
	loss_list = []  # 存储每次训练的平均损失值
	total_loss = 0.0
	train_samples = 0
	for epoch in range(epochs):  # 训练100次
		# 借助循环实现 mini-batch SGD 模型训练
		for train_x, train_y in dataloader:
			# 模型预测
			# train_x->float64
			# w->float32
			# y_pred = model(train_x.type(dtype=torch.float32))  # y=w*x+b
			y_pred = model(train_x)  # y=w*x+b
			print('y_pred->', y_pred)
			# 计算损失值, 调用损失函数对象
			# print('train_y->', train_y)
			# y_pred: 二维张量
			# train_y: 一维张量, 修改成二维张量, n行1列
			# 可能发生报错, 修改形状
			# 修改train_y元素类型, 和y_pred类型一致, 否则发生报错
			# loss = criterion(y_pred, train_y.reshape(shape=(-1, 1)).type(dtype=torch.float32))
			loss = criterion(y_pred, train_y.reshape(shape=(-1, 1)))
			print('loss->', loss)
			# 获取loss标量张量的数值 item()
			# 统计n次batch的总MSE值
			total_loss += loss.item()
			# 统计batch次数
			train_samples += 1
			# 梯度清零
			optimizer.zero_grad()
			# 计算梯度值
			loss.backward()
			# 梯度更新 w和b更新
			# step()等同 w=w-lr*grad
			optimizer.step()
		# 每次训练的平均损失值保存到loss列表中
		loss_list.append(total_loss / train_samples)
		print('每次训练的平均损失值->', total_loss / train_samples)
	print('loss_list->', loss_list)
	print('w->', model.weight)
	print('b->', model.bias)

	# 绘制每次训练损失值曲线变化图
	plt.plot(range(epochs), loss_list)
	plt.title('损失值曲线变化图')
	plt.grid()
	plt.show()

	# 绘制预测值和真实值对比图
	# 绘制样本点分布
	plt.scatter(x, y)
	# 获取1000个样本点
	# x = torch.linspace(start=x.min(), end=x.max(), steps=1000)
	# 计算训练模型的预测值
	y1 = torch.tensor(data=[v * model.weight + model.bias for v in x])
	# 计算真实值
	y2 = torch.tensor(data=[v * coef + 14.5 for v in x])
	plt.plot(x, y1, label='训练')
	plt.plot(x, y2, label='真实')
	plt.legend()
	plt.grid()
	plt.show()


if __name__ == '__main__':
	x, y, coef = create_datasets()
	train(x, y, coef)
