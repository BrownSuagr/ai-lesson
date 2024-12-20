'''
思考题：
① 中国合法工作年龄为18-60岁，即如果年龄小于18的情况为童工，不合法；
② 如果年龄在18-60岁之间为合法工龄；
③ 大于60岁为法定退休年龄。
'''
# 1、接收用户的年龄
age = int(input('请输入您的年龄：'))
# 2、进行age条件判断（① 小于18 == 童工 ② 18 ~ 60之间 == 合法工龄 ③ 大于60岁 == 退休年龄）
if age < 18:
    print('您还未成年，童工一枚')
# elif age >= 18 and age <= 60:
# 多重判断的另外一种写法（Python语言特有属性）
elif 60 >= age >= 18:
    print('您的年龄处于18~60周岁之间，合法工龄')
else:
    print('您的年龄大于60岁，已到了退休年龄')