'''
需求：女朋友生气了，要惩罚：连续说5遍“老婆大人，我错了”，如果道歉正常完毕后女朋友就原谅我了，这个程序怎么写？
'''
# 定义初始化计数器
i = 0
# 编写循环条件
while i < 5:
    print('老婆大人，我错了')
    # 在循环体内部更新计数器
    i += 1
else:
    print('真开森，老婆原谅我了！')