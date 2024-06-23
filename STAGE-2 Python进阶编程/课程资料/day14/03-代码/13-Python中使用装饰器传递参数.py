# 编写通用装饰器
def logging(flag):
    # ① 有嵌套 ② 有引用 ③ 有返回
    def decoration(fn):
        def inner(*args, **kwargs):
            if flag == '+':
                print('--输出日志：正在努力进行加法运算--')
            elif flag == '-':
                print('--输出日志：正在努力进行减法运算--')
            return fn(*args, **kwargs)
        return inner
    # ④ 再次返回（返回logging内层函数在内存中地址
    return decoration


# 定义一个原函数
@logging('+')
def sum_num(num1, num2):
    return num1 + num2

# 定义一个原函数
@logging('-')
def sub_num(num1, num2):
    return num1 - num2

print(sum_num(10, 20))
print(sub_num(30, 10))