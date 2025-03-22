# tanh激活值: torch.tanh(x)

import torch
import matplotlib.pyplot as plt
from torch.nn import functional as F
# F.sigmoid()
# F.tanh()

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def dm01():
	# 创建x值, 线性模型输出值作为激活函数的输入值
	x = torch.linspace(-20, 20, 1000)
	# 计算激活值
	y = torch.tanh(input=x)
	# 创建画布对象和坐标轴对象
	_, axes = plt.subplots(1, 2)  # 一行两列, 绘制两个子图
	axes[0].plot(x, y)
	axes[0].grid()
	axes[0].set_title('tanh激活函数')

	# 创建x值,可以自动微分, 线性模型输出值作为激活函数的输入值
	x = torch.linspace(-20, 20, 1000, requires_grad=True)
	torch.tanh(input=x).sum().backward()
	axes[1].plot(x.detach().numpy(), x.grad)
	axes[1].grid()
	axes[1].set_title('tanh激活函数')
	plt.show()


if __name__ == '__main__':
	dm01()
