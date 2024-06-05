# 1、有嵌套
def check(fn):
    def inner():
        # 编写装饰器功能（修饰代码）
        print('请先登录')
        # 2、与引用
        fn()
    # 3、有返回
    return inner

# 定义一个已有函数
@check
def comment():
    print('编写评论')

# 调用comment函数
comment()