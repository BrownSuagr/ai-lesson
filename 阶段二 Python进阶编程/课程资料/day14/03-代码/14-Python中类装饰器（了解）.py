# 定义一个类装饰器
class Check():
    # 定义一个__init__初始化方法，用于接收参数fn(原函数名称）
    def __init__(self, fn):
        # 把原函数名称赋值给自身的__fn私有属性
        self.__fn = fn
    # 定义一个__call__魔术方法，用于把类当做一个函数调用
    def __call__(self, *args, **kwargs):
        # 编写装饰器功能
        print('请先登录')
        # 调用comment原函数
        self.__fn()

# 定义原函数
@Check
def comment():
    print('编写评论')

# 调用函数
comment()