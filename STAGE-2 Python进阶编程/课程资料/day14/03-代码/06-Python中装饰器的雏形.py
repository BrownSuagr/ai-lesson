# 定义一个装饰器
# 参数fn，要求传递要修饰函数的名称
# 1、有嵌套
def check(fn):  # fn = comment函数！
    def inner():
        # 编写装饰器功能
        print('请先登录')       # ① 输出额外增加的代码
        # 2、有引用
        fn()                  # ② 执行原有函数
    # 3、有返回
    return inner


# 定义一个已有函数
def comment():
    print('编写评论')


# 调用装饰器代码装饰comment函数
comment = check(comment)  # 调用check装饰器，然后把comment函数的内存地址赋值给check的fn参数
comment()  # 由于comment指向了inner函数，所以comment()代表执行inner函数