import copy

# 1、Python中深浅拷贝特殊案例1（可变嵌套不可变类型）
a = [1, 3, 5, (7, 9)]
b = copy.copy(a)
c = copy.deepcopy(a)
# print(id(a))
# print(id(b))
# print(id(c))

# print(id(a[3]))
# print(id(b[3]))
# print(id(c[3]))

# 结论：外层是可变类型，所以可以进行完全拷贝（需要生成内存空间），但是内层对象是不可变数据类型，所以只能拷贝引用关系

# 2、Python中深浅拷贝特殊案例2（不可变类型嵌套可变）
d = (1, 3, 5, [7, 9])
e = copy.copy(d)
f = copy.deepcopy(d)

print(id(d))
print(id(e))
print(id(f))

print(id(d[3]))
print(id(e[3]))
print(id(f[3]))

# 特殊结论：如果一个不可变数据类型包含了可变数据类型，浅拷贝结论与之前结论一致，都只能拷贝引用关系
# 但是对于深拷贝，这个有点不同了，如果这种类型使用深拷贝，其整体都可以进行完全拷贝。