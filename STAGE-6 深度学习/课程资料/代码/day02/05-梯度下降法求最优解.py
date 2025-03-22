"""
① 创建自动微分w权重张量
② 自定义损失函数 loss=w**2+20  后续无需自定义,导入不同问题损失函数模块
③ 前向传播 -> 先根据上一版模型计算预测y值, 根据损失函数计算出损失值
④ 反向传播 -> 计算梯度
⑤ 梯度更新 -> 梯度下降法更新w权重
"""
import torch


def dm01():
	# ① 创建自动微分w权重张量
	w = torch.tensor(data=10, requires_grad=True, dtype=torch.float32)
	print('w->', w)
	# ② 自定义损失函数 后续无需自定义, 导入不同问题损失函数模块
	loss = w ** 2 + 20
	print('loss->', loss)
	# 0.01 -> 学习率
	print('开始 权重x初始值:%.6f (0.01 * w.grad):无 loss:%.6f' % (w, loss))
	for i in range(1, 1001):
		# ③ 前向传播 -> 先根据上一版模型计算预测y值, 根据损失函数计算出损失值
		loss = w ** 2 + 20
		# 梯度清零 -> 梯度累加, 没有梯度默认None
		if w.grad is not None:
			w.grad.zero_()
		# ④ 反向传播 -> 计算梯度
		loss.sum().backward()
		# ⑤ 梯度更新 -> 梯度下降法更新w权重
		# W = W - lr * W.grad
		# w.data -> 更新w张量对象的数据, 不能直接使用w(将结果重新保存到一个新的变量中)
		w.data = w.data - 0.01 * w.grad
		print('w.grad->', w.grad)
		print('次数:%d 权重w: %.6f, (0.01 * w.grad):%.6f loss:%.6f' % (i, w, 0.01 * w.grad, loss))

	print('w->', w, w.grad, 'loss最小值', loss)


if __name__ == '__main__':
	dm01()
