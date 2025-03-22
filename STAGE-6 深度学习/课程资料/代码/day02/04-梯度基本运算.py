"""
梯度: 求导,求微分 上山下山最快的方向
梯度下降法: W1=W0-lr*梯度   lr是可调整已知参数  W0:初始模型的权重,已知  计算出W0的梯度后更新到W1权重
pytorch中如何自动计算梯度 自动微分模块
注意点: ①loss标量和w向量进行微分  ②梯度默认累加,计算当前的梯度, 梯度值是上次和当前次求和  ③梯度存储.grad属性中
"""
import torch


def dm01():
	# 创建标量张量 w权重
	# requires_grad: 是否自动微分,默认False
	# dtype: 自动微分的张量元素类型必须是浮点类型
	# w = torch.tensor(data=10, requires_grad=True, dtype=torch.float32)
	# 创建向量张量 w权重
	w = torch.tensor(data=[10, 20], requires_grad=True, dtype=torch.float32)
	# 定义损失函数, 计算损失值
	loss = 2 * w ** 2
	print('loss->', loss)
	print('loss.sum()->', loss.sum())
	# 计算梯度 反向传播  loss必须是标量张量,否则无法计算梯度
	loss.sum().backward()
	# 获取w权重的梯度值
	print('w.grad->', w.grad)
	w.data = w.data - 0.01 * w.grad
	print('w->', w)


if __name__ == '__main__':
	dm01()
