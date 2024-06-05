# 1、有嵌套
def outer():
    num = 10
    def inner():
        # nonlocal关键字
        nonlocal num
        # 2、有引用
        num = 20
    print(f'outer函数中的num变量值：{num}') # 10
    inner()
    print(f'当inner函数执行完毕后，查看outer函数中的num是否受到影响：{num}')  # 20

outer()

# global，用于声明全局变量（使用或修改全局作用域中的某个变量）
# nonlocal，主要应用嵌套环境，适用于在嵌套函数的内部修改外部或访问外部函数的局部变量