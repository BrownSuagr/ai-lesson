# 正则表达式模块
import re

'''
    正则表达式：
        - 分组别名：
            - (?P<name>)：分组起别名
            - (?P=name) : 引用别名为name的分组匹配到字符串

'''

# 主函数
def main():
    string = 'hellojava, hellopython'
    result = re.finditer(r'hello(java|python)', string)
    if result:
        # 匹配到了结果
        for i in result:
            print(f'第{i} 匹配到结果是：{i.group()}')

    else:
        print('没有匹配到结果')

    str1 = '<book></book>'
    result1 = re.search(r'<(?P<mark>\w+)></(?P=mark)>', str1)
    print(result1.group())

    str2 = '125363@qq.com, go@126.com, heima123@163.com'
    result2 = re.finditer('\w+@(qq|163|126)\.com', str2)
    if result2:
        for i in result2:
            print(f'第{i} 匹配到结果是：{i.group()}')
    else:
        print('没有匹配到结果')





# 主函数判断器
if __name__ == '__main__':
    main()
