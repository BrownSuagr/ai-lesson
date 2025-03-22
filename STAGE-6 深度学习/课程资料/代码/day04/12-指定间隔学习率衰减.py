# 指定间隔: 通过步长列表指定训练次数后修改学习率  lr=lr*gamma
import torch
from torch import optim
import matplotlib.pyplot as plt


def dm01():
	# todo: 1-初始化参数
	# lr epoch iteration
	lr = 0.1
	epoch = 200
	iteration = 10
	# todo: 2-创建数据集
	# y_true x w
	y_true = torch.tensor([0])
	x = torch.tensor([1.0], dtype=torch.float32)
	w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
	# todo: 3-创建优化器对象 动量法
	optimizer = optim.SGD(params=[w], lr=lr, momentum=0.9)
	# todo: 4-创建等间隔学习率衰减对象
	# optimizer: 优化器对象
	# milestones: 指定间隔列表, 指定训练次数后修改学习率
	# gamma: 衰减系数 默认0.1
	scheduer = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[50, 100, 160], gamma=0.5, last_epoch=-1)
	# todo: 5-创建两个列表, 收集训练次数, 收集每次训练lr
	lr_list, epoch_list = [], []
	# todo: 6-循环遍历训练次数
	for i in range(epoch):
		# todo: 7-获取每次训练的次数和lr保存到列表中
		# scheduer.get_last_lr(): 获取最后lr
		lr_list.append(scheduer.get_last_lr())
		epoch_list.append(i)
		# todo: 8-循环遍历, batch计算
		for batch in range(iteration):
			# 先算预测y值 wx, 计算损失值 (wx-y_true)**2
			y_pred = w * x
			loss = (y_pred - y_true) ** 2
			# 梯度清零
			optimizer.zero_grad()
			# 梯度计算
			loss.backward()
			# 参数更新
			optimizer.step()
		# todo: 9-更新下一次训练的学习率
		scheduer.step()
	print('lr_list->', lr_list)

	plt.plot(epoch_list, lr_list)
	plt.xlabel("Epoch")
	plt.ylabel("Learning rate")
	plt.legend()
	plt.show()


if __name__ == '__main__':
	dm01()
