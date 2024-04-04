def main():
    # print('主函数入口')
    # 闭包

    # 装饰器
    #     - 装饰器带有返回值的函数
    #     - 装饰器带有参数的函数
    #     - 装饰器传递参数（高级装饰器）
    #     - 类中装饰器

    sub_num(1, 2)
    sum_num(1, 2)

# 装饰器传递参数
# 定一个外层的函数接收装饰器传递参数
def logging(flag):
    # 定义装饰器方法
    def decoration(fn):
        # 定一个装饰器内层函数
        def inner(*args, **kwargs):
            if '+' == flag:
                print('---输出日志，正在努力进行加法运算')
            else:
                print('---输出日志，正在努力进行减法运算')
            # 返回装饰器调用函数
            return fn(*args, **kwargs)
        # 返回内置函数地址
        return inner
    # 返回修饰器函数
    return decoration

@logging('+')
def sum_num(num1, num2):
    return num1 + num2

@logging('-')
def sub_num(num1, num2):
    return num1 - num2


# 类装饰器
class Check(object):
    # 定一个初始化方法用户接收fn函数的内存地址
    def __init__(self, fn):
        # 把函数的内存地址赋值给类的私有属性
        self.__fn = fn

    # 定一个__call__魔法值方法，用于把类当作一个函数调用
    def __call__(self, *args, **kwargs):
        # 装饰器的功能逻辑
        print('请先登录')
        # 调用装饰器修饰的函数
        self.__fn()

@Check
def comment():
    print('开始编写评论')


if __name__ == '__main__':
    main()
