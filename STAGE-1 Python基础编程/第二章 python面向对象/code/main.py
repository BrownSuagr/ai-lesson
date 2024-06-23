# 主函数入口
def OOP_Main():
    print('主函数入口')

    # OOA：面向对象分析
    # OOD：面向对象设计
    # OOP：面向对象编程

    # 1、Python类常见两种：经典类和新式类
    # 基本语法：
    #   - 经典类：class 类名: pass
    #   - 新式类：class 类名('继承对象'): pass
    # 实例化语法：对象 = 类名()
    # self关键字：指向创建的对象引用地址
    # 类魔术方法：
    #   - __init__()（初始化方法/构造方法）、
    #   - __del__()（删除方法/析构方法）
    #   - __str__()（对象串化方法/toString方法）

    s = Students('Alice', 61)
    # print(s.__age)
    print(s.get__age())
    print(s.set__age())
    print(s)
    del s

    test = Test('咪咪', 4)
    print(test)
    del test

    # 文件触发现实生活中的业务就是这样的，文件内部的项目内容

    # 2、面向对象基本特性：
    # - 封装：
    #   - 公有方法：可以在类内外部都可以访问到的属性和方法
    #   - 私有方法：只能在类内部访问的属性和方法通过"__"创建私有属性或方法，权限验证后可以通过特定函数访问私有属性或者方法
    #       - 属性：self.__age = 18 (def get_age(self): pass; def set_age(self): pass)
    #       - 方法：def __learn(self): pass
    # - 多态
    # - 继承


class Students(object):

    # 参数初始化
    def __init__(self, name, score):
        self.name = name
        self.score = score
        self.__age = 18

    def __del__(self):
        print('当对象被删除或者销毁时，会被自动调用')

    def get__age(self):
        print(f'get__age:{self.__age}')

    def set__age(self):
        print(f'set__age:{self.__age + 1}')

    def __str__(self):
        score_str = ''
        if self.score <= 60:
            score_str = '不合格'
        elif 60 <= self.score < 70:
            score_str = '合格'
        elif 70 <= self.score < 80:
            score_str = '中等'
        elif 80 <= self.score < 90:
            score_str = '良好'
        elif 90 <= self.score:
            score_str = '优秀'
        return f'姓名：{self.name} 成绩：{self.score} 等级：{score_str}'


class Test(object):

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def eat(self):
        print(f'我喜欢吃零食{self}')

    def drink(self):
        print(f'我喜欢喝可乐{self}')

    def __del__(self):
        print('当对象被删除或者销毁时，会被自动调用')

    def __str__(self):
        return '姓名是{},年龄是{}'.format(self.name, self.age)


if __name__ == '__main__':
    OOP_Main()
