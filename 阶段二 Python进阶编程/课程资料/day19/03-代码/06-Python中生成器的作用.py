'''
生成器可以降低程序的能耗，加快程序的执行时间
'''
# 导入time模块
import time

# 获取程序开始前的时间
start = time.time()

# 使用列表存储一组数据，数据一共有100万个，要求存储时，每个列表元素是循环变量的平方
square_nums = [i * i for i in range(10000000)]
# square_nums = (i * i for i in range(10000000))

# 获取程序结束前的时间
end = time.time()

# 计算程序执行的总时间
print(f'程序执行的总时间为{end - start}')