'''
生成器可以降低程序的能耗，加快程序的执行时间
'''
# 导入memory内存模块
import memory_profiler as mem

# 获取程序开始前的内存信息
start = mem.memory_usage()
print(f'程序执行前的内存情况：{start}')

# 使用列表存储一组数据，数据一共有100万个，要求存储时，每个列表元素是循环变量的平方
# square_nums = [i * i for i in range(10000000)]
square_nums = (i * i for i in range(10000000))

# 获取程序结束前的内存信息
end = mem.memory_usage()
print(f'程序结束前的内存情况：{end}')