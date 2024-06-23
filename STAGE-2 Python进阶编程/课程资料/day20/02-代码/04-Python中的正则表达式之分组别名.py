'''
匹配结果 => <book></book>
分析标签的特点：
html大部分都是双标签，<book></book>也是一个双标签
<标签名></标签名>，通过分析可以发现<>中的标签名都是一致的
<(标签名)></引用前面分组匹配的结果>
'''

# 导入模块
import re

# 定义一个html字符串
str1 = '<book></book>'
# 定义一个正则表达式
# result = re.search(r'<(\w+)></\1>', str1)
# <book></book>  =>  <标签名></标签名>  \w+   (?P<mark>\w+)   =>  (?P=mark)
result = re.search(r'<(?P<mark>\w+)></(?P=mark)>', str1)

# 输出匹配的内容
print(result.group())