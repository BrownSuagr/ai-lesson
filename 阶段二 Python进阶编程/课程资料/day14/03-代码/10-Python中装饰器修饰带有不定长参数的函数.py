# 编写装饰器，用于给每个函数增加一个日志输出功能（① 有嵌套 ② 有引用 ③ 有返回）
def logging(fn):
    def inner(*args, **kwargs):
        # 输出装饰器要额外添加功能
        print('--正在努力进行进行计算--')
        fn(*args, **kwargs)
    # 返回内层函数的内存地址
    return inner

# 定义原函数（参数不固定），可以接收任意长度的参数，其返回结果就是所有参数的和
@logging
def sum_num(*args, **kwargs):
    result = 0
    for i in args:
        result += i
    for i in kwargs.values():
        result += i
    print(result)


# 调用原函数
sum_num(1, 2, a=3, b=4, c=5)  # 1+2+3+4+5