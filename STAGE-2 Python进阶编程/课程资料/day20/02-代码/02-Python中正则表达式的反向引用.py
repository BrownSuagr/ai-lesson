# 导入模块
import re

# 定义一个字符串
str1 = 'abc2222defg'
# 定义正则表达式
result = re.search(r'(\d)\1\1\1', str1)
print(result.group())