'''
如何创建一个字符串I'm Tom
'I'm Tom'报错：报错在于引号必须成对出现，不能单独出现，而且Python解析器在解析引号的时候，采用就近原则
第一个引号与其最近的一个引号，系统认为是一对
'''
str1 = 'I\'m Tom'
str2 = "I'm Tom"