import copy

a = (1, 3, 5, (7, 9))

b = copy.deepcopy(a)

# print(a)
# print(b)

print(id(a))
print(id(b))

print(id(a[3]))
print(id(b[3]))

# 结论：① 对于简单的不可变类型数据，深拷贝也只能拷贝对象的引用关系，所以看到的结果就是a和b指向了相同的内存空间
# 结论：② 对于复杂的不可变类型数据，深拷贝也只能拷贝对象的引用关系，所以看到的结果就是a和b指向了相同的内存空间