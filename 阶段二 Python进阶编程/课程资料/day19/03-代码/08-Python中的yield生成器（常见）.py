'''
yield生成器，其结构一共分为两部分：① 首先要定义一个函数 ② 在函数内部存在一个yield关键字
我们把①②组合在一起就称之为叫做yield生成器
另外要特别强调：yield生成器是一个对象而不是一个函数

重点：理解yield生成器的执行流程（其使用方式和意义和刚才介绍的是一模一样的）
'''
def generator(n):
    for i in range(n):  # 0 1 2 3 4
        print('开始生成数据')
        yield i  # 暂时把yield当做return理解，每次遇到yield，生成器就相当于执行1次（把yield理解为暂停）
        print('完成1次数据生成')

# 在使用时，由于生成器需要传递参数，所以通常将其赋予给某个变量
g = generator(5)
# print(next(g))  # 开始生成数据，弹出数字0
# print(next(g))  # 弹出数字1
# print(next(g))  # 2
# print(next(g))  # 3
# print(next(g))  # 4
#
# print(next(g))  # 生成器中默认只有5个元素的生成规则，当尝试打印第6个元素时，会弹出yield后面的输出语句

for i in g:  # for循环中，取数据使用的方式就是next()
    print(i)