if __name__ == '__main__':

    name = '黑糖'
    age = 12

    # 1、变量（Python中变量一般使用下划线连接）

    # 1-3、参数格式化输出

    # 格式化符号输出
    print("name: %s age: %d" % (name, age))

    # format函数格式化
    print("格式化函数输出（formant()）:name:{} age:{}".format(name, age))
    print(f'简化的格式化函数输出：name:{name} age:{age}')

    # 1-4、Python中input函数 input('函数输入提示信息')
    # i = input('请输入：')
    i = 1
    print("数据类型：%s " % type(i))
    print(f'input:{i}')

    # username = input('请输入用户名')
    username = 'admin'
    # password = input('请输入密码')
    password = '123456'

    if username == 'admin' and password == '123456':
        print('login success')
    else:
        print('login error')

    # 1-5、Python运算符
    # 常见7种数据类型转换

    num = int('123')
    print(type(num))

    # 转换为整型
    print(f'num:{num}')

    # 转换为浮点型
    num = float('123')
    print(f'num:{num}')
    print(f'num 格式化:{num:.4f}')

    # 转换数据到原始类型
    num = eval('123')
    print(f'num:{num} type of num:{type(num)}')

    f = eval('123.0')
    print(f's:{f} type of str:{type(f)}')

    # 常见的运算形式
    num1 = 9
    num2 = 3

    print(f'减法运算 : {num1 - num2}')
    print(f'加法运算 : {num1 + num2}')
    print(f'乘法运算 : {num1 * num2}')
    print(f'除法运算 : {num1 / num2}')
    print(f'整除运算 : {num1 // num2}')
    print(f'求余运算 : {num1 % num2}')
    print(f'幂指数运算 : {num1 ** num2}')

    # 单个和多个变量赋值
    num3, s, b = 1, 'Str', True
    print(num3)
    print(s)
    print(b)
    a = b = c = '1'
    print(a)
    print(b)
    print(c)

    # 若表达式两边不为布尔类型的判断结果，那么逻辑运算符两边返回的值是根据逻辑表达式两边短路运算的优先级决定的（0 '' null 都判断为false ）
    # 如果布尔值跟非布尔值进行逻辑运算，那一边为真则返回对应的数据类型
    print(3 and 4 and 5)
    print(5 and 6 or 7)
    4 > 3 and print('hello word')

    # 1-6、分支语句

    # if …… else …… 分支语句
    # if …… elif …… else  多重分支语句

    height = 180
    weight = 70
    bmi = weight / (height / 100) ** 2

    if 10 <= bmi <= 18.4:
        print(f'bim:{bmi:.4f} 体重状态「偏瘦」')
    elif 18.4 < bmi <= 23.9:
        print(f'bim:{bmi:.4f} 体重状态「正常」')
    elif 23.9 < bmi <= 27.9:
        print(f'bim:{bmi:.4f} 体重状态「过重」')
    else:
        print(f'bim:{bmi:.4f} 体重状态「肥胖」')

    # 三木运算符 返回值 = value1 if 条件1 else value2

    # 1-7、while循环 while 条件：逻辑代码
    i = 0
    while i < 10:
        i += 1
        print(f'i:{i}')
        if 2 == i:
            print('Termination loop')
            break
    else:
        print('This loop is end')

    # continue和break关键词

    for i in range(1, 50):
        print(f'i:{i}')
        i += 1
        if 4 == i:
            print('Get out of current loop')
    else:
        print('This loop is end')

    # 1-11、字符串
    # 声明方式 1、单引号或双引号 2、三引号（针对于存在换行的文本）
    name1 = 'Bob'
    name2 = "Smith"
    name3 = ''' 
        Jack likes learning Python
        because it is very interesting
    '''
    # 字符串遍历
    j = 0
    while j < len(name2):
        print(f'name2:{name2[j]}')
        j += 1

    for i in name2:
        print(f'name2:{i}')

    # 1-12、字符串切片 API格式：字符串[起始位置:结束位置:步长] （步长可以为负数，正向从0开始，反向从-1开始，但截取方向总是从左向右）
    s = '123456789'
    print('切片测试：' + s[1:3:2])
    print('包含起始位置：' + s[1:])
    print('包含起始位置（负数）：' + s[-1:])
    print('包含结束位置：' + s[:4])
    print('包含结束位置（负数）：' + s[:-4])
    print('包含步长：' + s[::2])
    print('包含步长（负数）：' + s[::-2])
    print('都没有：' + s[:])

    # 字符串查找 格式：字符串.find('目标字符串') 存在则返回目标字符串起始位置下标，否则返回-1
    # 字符串查找 格式：字符串.index('目标字符串') 存在则返回目标字符串起始位置下标，否则报异常
    # 字符串查找 格式：字符串.rfind('目标字符串') 从左边开始查找 存在则返回目标字符串起始位置下标，否则返回-1

    f = 'I like learning Python'
    print('find 存在 %d' % (f.find('like')))
    print('find不存在%d' % (f.find('job')))

    print('index 存在 %d' % (f.index('like')))
    # print('index不存在%d' %(f.index('job')))

    print('rfind 存在 %d' % (f.rfind('like')))
    # print('rfind 存在 %d' % (f.rindex('j')))

    # 字符串替换 格式：字符串.replace('目标字符串') 返回替换后字符串
    # 字符串切割 格式：字符串.split('目标字符串') 返回切割后字符串
    print(f.split())  # 默认以空格为分隔符
    print(f.split('i', 1))  # 以 i 为分隔符
    print(f.split('i'))  # 以 w 为分隔符

    # 字符串连接 格式：连接字符串.join('目标字符串') 返回连接后字符串
    print('-'.join(("a", "b", "c")))
    print('-'.join("abc"))

    # 字符串开始结尾判断 格式：字符串.startswith('目标字符串') 字符串.endswith('目标字符串') 返回True 否则False
    print('123'.startswith("1"))
    print('123'.endswith("3"))

    # 12、列表
    # 基本语法：列表名 = [val1, val2, val3 ……]

    name_list = ['Tom', 'Jack', 'Smith', 'Tom']
    print(name_list)

    for i in name_list: print(i)

    # 列表查找语法：list.index('target') 存在返回下标，不存在报错
    print(name_list.index('Tom'))
    # print(name_list.index('Amy'))

    # 在列表中出现次数语法：list.count('target') 返回出现次数，没有返回0
    print(name_list.count('Tom'))
    print(name_list.count('Amy'))

    # 在列表中是否存在： target in list 存在True，否则False
    # 在列表中是否不存在： target not in 列表名 存在True，否则False
    print('Amy' in name_list)
    print('Amy' not in name_list)

    # 在列表尾部增加元素：list.append(target)
    name_list.append('Amy')
    print(name_list)

    # 将列表2数据合并到列表1：list1.extend(list2)
    name_list2 = ['Amy', 'Alice', 'Anna', 'Juliet']
    name_list.extend(name_list2)
    name_list.extend(['Mandy'])
    print(name_list)

    # 将列表指定位置插入数据：list.insert(index, target)
    name_list.insert(2, 'Bob')
    print(name_list)

    # 将列表删除指定位置数据：list.pop(index) 返回对应位置数据
    pop = name_list.pop(2)
    print(pop)
    print(name_list)

    # 列表移除匹配数据：list.remove(target)
    name_list.remove('Mandy')
    print(name_list)

    # 列表修改数据：list[index] = target
    name_list[3] = 'Emma'
    print(name_list)

    # 列表倒序：list.reverse()
    name_list.reverse()
    print(name_list)

    # 列表正序：list.sort()
    name_list.sort()
    print(name_list)

    # 列表依然适用字符串切片：list[::2]
    print(name_list[::2])

    # 列表移除匹配数据：list.clear()
    name_list.clear()
    print(name_list)

    # 二维列表 list = [list1, list2, list3]
    tow_dimension_list = [[1, 2], [3, 4], [5, 6]]
    print(tow_dimension_list)

    # 13、元组
    # 基本语法(数据无法修改，只有查询方法)：tuple = (val1, val2, val3 ……)
    t = ('Tom',)
    print(f'单元素元组（不能少逗号）:{t}')

    t = ('Tom', 'Anna', 'Alice')
    print(f'多元素元组:{t}')

    # 14、字典
    # 基本语法(类似于map) dict = {'key1':'val1', 'key2':'val2', 'key3':'val3', }
    dic = {'id': 1, 'name': 'Alice', 'age': 20}
    print(type(dic))
    print(dic)
    print(dic['name'])

    person = {'id': '2', 'name': 'Amy'}
    print(person)
    person['age'] = 44
    print(person)
    del person['age']
    print(person)
    print(person.keys())
    print(person.values())
    print(person.get('names', '您要查找的内容不存在'))

    for (key, value) in person.items():
        print(f'key:{key} value:{value}')
        print(key, ':', value)

    person.clear()
    print(person)

    # 15、集合
    # 基本语法(无序不重复)：set = {val1, val2, val3 ……} 或者 set = set(val1, val2, val3 ……)

    s = {1, 2, 3, 4}
    print(s)

    s.add(10)
    print(s)

    ls = [99, 88]
    s.update(ls)
    print(s)

    # 存在则删除，不存在报错
    s.remove(99)
    print(s)

    # 存在则删除，不抱错
    s.discard(88)
    print(s)

    # 随机删除，返回随机删除的数据
    print(s.pop())

    # 元素是否在集合内，在True 否则False
    print(10 in s)

    # 元素是否不在集合内，不在True 否则False
    print(10 not in s)

    # 集合交集&
    s2 = {1, 2}
    print(s & s2)

    # 集合并集｜
    s3 = {77, 66}
    print(s | s3)

    # 集合差集-
    s3 = {77, 66}
    print(s - s3)

    # 常见的公共方法：
    # 字符串、列表、元组：+ 合并、* 复制、
    # 字符串、列表、元组、字典、集合：in 是否存在、not in 是否不存在

    # 字符串、列表、元组、字典、集合：len()：容器元素、
    # del/del()：删除元素、
    # max()：最大值、
    # min()：最小值、
    # range(start, end, step)：生成集合、
    # enumerate()：

    # 18、推导式

    # 列表推导式
    # 列表推导式(基本) 基本语法： 集合 = [i for i in range(10)]
    list1 = [i for i in range(10)]
    print(list1)

    # 列表推导式(if) 基本语法： 集合 = [i for i in range(10) if 条件] 满足条件的循环
    list2 = [i for i in range(10) if i % 2 == 0]
    print(list2)

    # 列表推导式(for嵌套) 基本语法： 集合 = [(i, j) for i in range(10) for j in range(0, 3)]
    list3 = [(i, j) for i in range(1, 3) for j in range(0, 3)]
    print(list3)

    # 字典推导式
    d = {i: i ** 2 for i in range(1, 6)}
    print(d)

    l1 = ['name', 'age', 'gender']
    l2 = ['Tom', 20, 'male']
    d = {l1[i]: l2[i] for i in range(3)}
    print(d)

    computer_dict = {'MBP': 268, 'HP': 125, 'DELL': 201, 'Lenovo': 199, 'ACER': 99}
    print(computer_dict)

    new_computer_dict = {i: computer_dict.get(i) for i in computer_dict.keys() if computer_dict.get(i) > 200}
    print(new_computer_dict)

    new_computer_dict2 = {k: v for k, v in computer_dict.items() if v > 200}
    print(new_computer_dict2)


    # 19、函数
    # 定义 def 函数名(参数)： 函数体 return 返回值
    # 调用：函数名(参数)

    def a(param):
        print(param + '执行函数')
        print('~' * 66)
        return 1


    print(a('调用函数，'))


    def b(*args):
        print(f'{args}执行函数')
        print('~' * 66)
        return 1


    print(b('调用函数，', '1', '2'))


    def c(*args, **kwargs):
        print(f'args:{args}')
        print(f'kwargs:{kwargs}')
        print('~' * 66)


    args = ('调用函数', '1', '2')
    kwargs = {'a': 1, 'b': 2, "c": 3, "d": 4}

    c(*args, **kwargs)
    c(**kwargs)
    c(*args)

    c1 = 10
    c2 = 2

    c1 = c1 + c2
    c2 = c1 - c2
    c1 = c1 - c2

    print('不引入第三方便利那个实现两个数交换')
    print(c1)
    print(c2)
    print('~' * 66)

    print('Python拆包数据交换')
    c1, c2 = c2, c1
    print(c1)
    print(c2)

    c1, c2 = c2, c1
    print(c1)
    print(c2)
    print('~' * 66)

    print('百分号格式化正确输出')
    print('%d%%及格' % 50)
    print('~' * 66)

    print('字符串切片')
    s = 'abcdefg'
    print(s[3::-2])
    print('~' * 66)

    print('字典中的key数据类型可以是很多（整型、浮点、字符串、元组）')

    print('~' * 66)

    # 典型的数据函数

    # 20、可变和非可变数据类型
    # 栈内存（小、快）、堆内存（大、慢）、数据段（变量）、代码段（函数体）
    # 先创建一个变量在数据段、再在栈内存为变量创建一个地址，通过引用关联数据，把内存地址引用赋值给新的变量

    print('可变和非可变数据类型')
    a = 10
    print(f'a内存地址：{id(a)}')
    b = a
    print(f'b内存地址：{id(b)}')
    a = 100
    print(f'a内存地址：{id(a)}')
    print(f'b内存地址：{id(b)}')

    print('~' * 66)
    # 引用关系

    # 21、递归函数
    print('递归函数')
    print()

    print('~' * 66)

    # 22、Lambda表达式
    print('Lambda表达式')

    students = [
        {'name': 'Tom', 'age': 20},
        {'name': 'Alice', 'age': 14},
        {'name': 'Jack', 'age': 18},
    ]

    students.sort(key=lambda x: x['age'], reverse=True)
    print(students)

    print('~' * 66)

    fun_name = lambda arguments: arguments
    print(fun_name(1))
    print('~' * 66)

    # 23、文件操作 定义：文件操作为内置函数，open('文件路径：相对路径/绝对路径', '操作模式：常见的r/w/a/rb/wb/ab(read/write/append/read binary/write
    # binary/append binary)', '编码格式') 二进制的读写追加方式适用于非文本类文件读写 eg:.png/.mp3/.avi/.mp4
    file = open("../temp/py-file.txt", "r")
    read = file.read()
    print(f'测试文件read:{read}')
    file.close()
    print('~' * 66)

    file = open("../temp/py-file-encoding.txt", "r", encoding='utf-8')
    read = file.read()
    print(f'测试文件带编码格式read:{read}')
    file.close()
    print('~' * 66)

    file = open('../temp/py-file-write.txt', 'a', encoding='utf-8')
    write = file.write('创建py-file-write.txt文件，并写入Hello World')
    print(f'读取写入内容长度:{write}')
    file.close()
    print('~' * 66)

    # 文件读取: file.read() 参数：整型代表数据长度，无参：默认读取全部；根据编码格式可以读取字节/字符，
    file = open("../temp/py-file-encoding.txt", "r", encoding='utf-8')
    read = file.read(1)
    print(f'文件read（字符长度）:{read}')
    file.close()
    print('~' * 66)

    file = open("../temp/test.png", "rb")
    read = file.read(1)
    print(f'文件read（字节长度）:{read}')
    file.close()
    print('~' * 66)

    # 按行文件读取 file.readline() 代表一次读取一行
    file = open("../temp/py-file-encoding.txt", "r", encoding='utf-8')
    read1 = file.readline()
    read2 = file.readline()
    print(f'文件read-1（按行读取）:{read1}')
    print(f'文件read-2（按行读取）:{read2}')
    file.close()
    print('~' * 66)

    # 按行文件读取 file.readlines() 代表一次读取多行
    file = open("../temp/py-file-encoding.txt", "r", encoding='utf-8')
    read = file.readlines()
    print(f'文件read（按行读取多行）:{read}')
    file.close()
    print('~' * 66)

    # 文件指针移动 file.seek(offest, whence= 0) whence：0:文件头读取 1:从当前位置 2:从文件末尾 offset：偏移量
    file = open("../temp/py-file-encoding.txt", "r", encoding='utf-8')
    read = file.readlines()
    print(f'第一次文件read（按行读取多行）:{read}')
    # 第一次读取结束指针指向了文件末尾，导致第二次读取无数据，因此借助seek函数调整指针位置
    file.seek(0, 0)
    read = file.readlines()
    print(f'第二次文件read（按行读取多行）:{read}')
    file.close()
    print('~' * 66)

    # 文件备份
    old_file = open('../temp/test.mp4', 'rb')
    # old_file = open('../temp/test.png', 'rb')
    back_up_file = open('../temp/test[备份].mp4', 'wb')
    # back_up_file = open('../temp/test[备份].png', 'wb')
    while True:
        temp = old_file.read(1024)
        if 0 == len(temp):
            break
        else:
            back_up_file.write(temp)
    old_file.close()
    back_up_file.close()

    # os模块
    import os
    import time

    # 文件重命名 定义：os.rename('old_name', 'new_name')
    # 文件删除   定义：os.remove('file_name')

    # os.rename('../temp/test[备份].mp4', '../temp/test_rename[备份].mp4')
    # time.sleep(5)
    # os.remove('../temp/test[备份].png')

    # 24、Python异常
    # 基本语法：try: 会出现异常逻辑 except 异常类型 as e: 异常后处理 指定异常类型只会捕获指定异常（可以是多个），否则会捕获全部异常

    try:
        print('可能出现错误')
        pass
    except Exception as e:
        print(f'出现异常后执行, 捕获全部异常：{e}')
    else:
        print('没有异常执行')
    finally:
        print('无论是否出现异常都执行')

    try:
        pass
    except (NameError, ZeroDivisionError, IOError, OSError) as e:
        print(f'捕获多个异常：{e}')

    if 0 == 0:
        raise Exception('抛出自定义异常')

    # 25、Python模块和包
    # 模块：内置模块和自定义模块两种
    # 导入方式：
    #   import 模块名(多个模块之间使用逗号分隔) 导入模块的所有函数 使用方法：模块名.函数名()
    #   from 模块名 import 函数名            导入模块的指定功能 使用方法：函数名()
    #   from 模块名 import *                导入模块的所有函数 使用方法：函数名()
    #   from 模块名 as 别名                  导入模块的所有函数 使用方法：别名.函数名()
    #   from 模块名 import 函数名 as 别名     导入模块的指定功能 使用方法：函数别名()

    # 魔术变量：__variable__可以对导入自定义模块的内部测试函数进行忽略，在其他模块导入时不会执行测试代码
    # if __name__ == '__main__':
    #     print('执行测试代码')

    # 魔术变量 __all__ 限制只能使用模块中的指定功能 __all__ = ['函数名'] 限制使用函数集合

    # 包的基本结构
    # 在同一个文件夹内需要一个__init__.py 通过魔术变量__all__ 限制引入文件内的指定模块




















































