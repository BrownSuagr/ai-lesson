# coding:utf-8
import jieba.posseg as pseg

content = "我爱自然语言处理"
result = pseg.lcut(content)
print(f'result-->{result}')