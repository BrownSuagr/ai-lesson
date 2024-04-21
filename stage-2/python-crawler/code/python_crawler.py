# 导入requests包
import requests
# 导入正则表达式包
import re


# 主函数入口
def main():
    result = requests.get('https://wwww.baidu.com')
    content = result.content.decode('utf-8')
    print(f'content:{content}')

    m = re.finditer('.*src="(.*)" width.*', content)
    print(f'm:{m}')

    content_list = content.split('\n')
    for line in content_list:
        match = re.match('.*src="(.*)" width.*', line)
        print(match)







if __name__ == '__main__':
    main()
