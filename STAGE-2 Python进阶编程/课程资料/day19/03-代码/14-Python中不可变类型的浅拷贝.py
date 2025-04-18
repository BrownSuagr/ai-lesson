# 导入模块
import copy

# 定义一个变量a
a = (1, 3, 5, (7, 9))
print(id(a))

# 对元组进行浅拷贝
b = copy.copy(a)
print(id(b))

print(id(a[3]))
print(id(b[3]))

# print(a)
# print(b)

# 结论：① 对于简单的不可变类型数据，不可变数据类型地址一旦固定，值就无法改变了，又由于浅拷贝需要把自身的对象空间赋值给
# 另外一个变量，为了保证数据一致，只能让其指向相同的内存空间（不需要额外开辟内存空间）

# 结论：② 对于复杂的不可变类型数据，浅拷贝只能拷贝变量的值，无法拷贝内存空间（无法开辟新的内存空间），无法拷贝内层对象。