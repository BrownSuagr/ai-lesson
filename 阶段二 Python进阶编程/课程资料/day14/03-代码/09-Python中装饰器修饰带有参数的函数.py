# 编写装饰器，用于给每个函数增加一个日志输出功能（① 有嵌套 ② 有引用 ③ 有返回）
def logging(fn):
    def inner(n1, n2):
        print('--输出日志：正在执行加法运算--')
        # 调用原函数
        fn(n1, n2)
    return inner

# 原有函数
@logging
def sum_num(num1, num2):
    print(num1 + num2)

# 调用方式
sum_num(10, 20)  # 通过装饰器的执行流程可知 sum_num(10, 20) 等价于 inner(10, 20)