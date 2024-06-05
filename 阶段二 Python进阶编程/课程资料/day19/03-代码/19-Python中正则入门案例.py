# 1、导入模块
import re
# 2、使用match方法通过正则匹配结果（match只能匹配开头，如果匹配任意位置也可以使用findall())
# str1 = '13575006899'
# 使用正则进行匹配
# result = re.findall('8', str1)
# 3、打印输出结果
# print(result)

# 2、定义一个字符串，判断这个字符串中是否包含数字
# str1 = 'abc6de8fg'
# 3、使用re模块结合正则表达式获取所有的数字
# result = re.findall('\d', str1)
# 4、打印输出结果
# print(result)

# 2、定义一个字符串，获取字符串中所有的非数字字符
str1 = 'abcd6efg!@#$%'
# 3、使用re模块结合正则表达式获取所有非数字
result = re.findall('\D', str1)
# 4、打印输出结果
print(result)