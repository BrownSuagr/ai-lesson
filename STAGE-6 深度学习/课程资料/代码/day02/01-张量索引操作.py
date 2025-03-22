import torch

# 下标从左到右从0开始(0->第一个值), 从右到左从-1开始
# data[行下标, 列下标]
# data[0轴下标, 1轴下标, 2轴下标]

def dm01():
	# 创建张量
	torch.manual_seed(0)
	data = torch.randint(low=0, high=10, size=(4, 5))
	print('data->', data)
	# 根据下标值获取对应位置的元素
	# 行数据 第一行
	print('data[0] ->', data[0])
	# 列数据 第一列
	print('data[:, 0]->', data[:, 0])
	# 根据下标列表取值
	# 第二行第三列的值和第四行第五列值
	print('data[[1, 3], [2, 4]]->', data[[1, 3], [2, 4]])
	# [[1], [3]: 第二行第三列 第二行第五列值   第四行第三列 第四行第五列值
	print('data[[[1], [3]], [2, 4]]->', data[[[1], [3]], [2, 4]])
	# 根据布尔值取值
	# 第二列大于6的所有行数据
	print(data[:, 1] > 6)
	print('data[data[:, 1] > 6]->', data[data[:, 1] > 6])
	# 第三行大于6的所有列数据
	print('data[:, data[2]>6]->', data[:, data[2] > 6])
	# 根据范围取值  切片  [起始下标:结束下标:步长]
	# 第一行第三行以及第二列第四列张量
	print('data[::2, 1::2]->', data[::2, 1::2])

	# 创建三维张量
	data2 = torch.randint(0, 10, (3, 4, 5))
	print("data2->", data2)
	# 0轴第一个值
	print(data2[0, :, :])
	# 1轴第一个值
	print(data2[:, 0, :])
	# 2轴第一个值
	print(data2[:, :, 0])


if __name__ == '__main__':
	dm01()
