# 定义一个装饰器，用于装饰func函数（① 有嵌套 ② 有引用 ③ 有返回）
def logging(fn):
    def inner(*args, **kwargs):
        # 输出装饰器额外的功能
        print('--输出日志：正在努力进行计算--')
        # 调用原函数进行修饰
        return fn(*args, **kwargs)  # 30
    return inner



# 定义一个带有返回的函数，装饰器首先被执行@logging 等价于 inner
@logging
def func(num1, num2):
    result = num1 + num2
    return result

# 调用func函数
print(func(10, 20))  # func(10, 20) => @logging => inner
