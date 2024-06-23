# 1、有嵌套
def outer():
    # 局部变量
    b = 20
    def inner():
        # 2、有引用（内部函数引用了外部函数的局部变量）
        print(b)
    # 3、有返回（把内部函数的名称=>内存地址返回）
    return inner

# 调用outer函数
f = outer()  # 把outer函数的执行结果赋值给全局变量f
f()  # 调用inner函数，返回outer函数中的局部变量b