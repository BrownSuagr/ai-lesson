# 导入模块
import re

# 定义一个字符串
str1 = 'qq:10567'
# 获取qq文本与qq号码，简单来说就是按：冒号进行分割
result1 = re.split(':', str1)
print(result1[0])
print(result1[1])


str2 = 'abc123def456ghi'
result2 = re.split('\d{3}', str2)
print(result2)