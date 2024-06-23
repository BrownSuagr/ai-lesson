# 编写通用装饰器（① 有嵌套 ② 有引用 ③ 有返回）
def logging(fn):
    def inner(*args, **kwargs):
        # 输出日志信息
        print('--输出日志：正在进行努力计算--')
        # 返回结果
        return fn(*args, **kwargs)
    return inner

# 定义一个原函数
@logging
def sum_num(num1, num2):
    return num1 + num2

# 定义一个原函数
@logging
def sub_num(num1, num2):
    return num1 - num2

print(sum_num(10, 20))
print(sub_num(30, 10))