import time

# 定义一个装饰器，用于获取某个函数的执行时间（① 有嵌套 ② 有引用 ③ 有返回）
def get_time(fn):
    # fn局部变量
    def inner():
        # 给fn函数添加一些额外的功能
        start = time.time()
        # 相当于执行了原函数
        fn()
        end = time.time()
        print(f'{fn}函数执行一共花费了{end - start}s')
    # 把内层函数的地址使用return
    return inner

# 原有函数
@get_time
def func():
    list1 = []
    for i in range(100000):
        list1.append(i)

# 调用函数
func()