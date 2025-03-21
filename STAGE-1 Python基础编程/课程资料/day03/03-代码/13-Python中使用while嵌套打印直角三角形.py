'''
打印直角三角形，一般都是通过while嵌套来实现的
外层循环 ：打印行数
内层循环 ：控制1行显示几颗星星
注意：
第1行，打印1颗
第2行，打印2颗
...
第5行，打印5颗
'''
# 定义外层计数器
i = 0 # 第1行
# 编写外层循环条件
while i < 5:
    # 定义内层计数器
    j = 0
    # 编写循环条件
    while j <= i:
        print('*', end='  ')
        # 更新内部计数器
        j += 1
    # 内层循环结束，输出1次换行
    print('')
    # 更新外层计数器
    i += 1