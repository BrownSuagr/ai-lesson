'''
if嵌套：其实就是在if结构中又出现了if结构
if 外层条件判断:
   if 内层条件判断（此语句必须在外层条件成立后，才会执行）:
      ...
else:
   ...
执行顺序：先执行外层判断，如果外层条件为True，则进入if语句段；然后再执行内层的if条件判断。

案例：法律规定，车辆驾驶员的血液酒精含量小于 20mg/100ml 不构成酒驾；酒精含量大于或等于 20mg/100ml 为酒驾；
酒精含量大于或等于 80mg/100ml 为醉驾。编写 Python 程序判断是否为酒后驾车。
'''
proof = int(input('请输入驾驶员每100ml血液中酒精含量(mg)：'))
# 判断酒精含量是否小于20mg
if proof < 20:
    print('酒精含量小于20mg，不构成酒驾！')
else:
    # 涉嫌酒驾，深层次判断到底是酒驾还是醉驾
    if 80 > proof >= 20:
        print('您的酒精含量处于20~80之间，属于酒驾')
    else:
        print('危险，您的酒精含量超过80mg，已构成醉驾，拘留')
