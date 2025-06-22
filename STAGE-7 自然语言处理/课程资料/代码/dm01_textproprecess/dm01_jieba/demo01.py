# coding:utf-8
import jieba

# todo:第一种分词模式：精确模式分词
def dm01_test():
    # 需要切分的文档
    content = "传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能"
    # 使用jieba.cut方法进行分词 #返回一个generator：生成器
    result1 = jieba.cut(content, cut_all=False)
    # print(f'第一种方式：result1--》{result1}')
    # # 对生成器取元素
    # # for value in result1:
    # #     print(value)
    # # print(next(result1))
    # # print(next(result1))
    # # print(next(result1))
    # # print(next(result1))
    # # print(next(result1))
    # # print(next(result1))
    # print(list(result1))
    result2 = jieba.lcut(content, cut_all=False)
    print(f'精确模式：第二种方式：result2-->{result2}')

# todo:第二种分词模式：全模式分词
def dm02_test():
    # 需要切分的文档
    content = "传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能"
    # 使用jieba.cut方法进行分词 #返回一个generator：生成器
    result1 = jieba.cut(content, cut_all=True)
    # print(f'第一种方式：result1--》{result1}')
    # # 对生成器取元素
    # # for value in result1:
    # #     print(value)
    # # print(next(result1))
    # # print(next(result1))
    # # print(next(result1))
    # # print(next(result1))
    # # print(next(result1))
    # # print(next(result1))
    # print(list(result1))

    result2 = jieba.lcut(content, cut_all=True)
    print(f'全模式：第二种方式：result2-->{result2}')


# todo:第三种分词模式：搜索引擎分词：在精确模式基础上，对长词再次切分
def dm03_test():
    # 需要切分的文档
    content = "传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能"
    # 使用jieba.cut_for_search方法进行分词 #返回一个generator：生成器
    result1 = jieba.cut_for_search(content)

    # print(f'第一种方式：result1--》{result1}')
    # # 对生成器取元素
    # for value in result1:
    #     print(value)
    # # print(next(result1))
    # # print(next(result1))
    # # print(next(result1))
    # # print(next(result1))
    # # print(next(result1))
    # # print(next(result1))
    # print(list(result1))

    result2 = jieba.lcut_for_search(content)
    print(f'搜索引擎模式：第二种方式：result2-->{result2}')

# todo：jieba支持繁体分词
def dm04_test():
    content = "煩惱即是菩提，我暫且不提"
    result = jieba.lcut(content)
    print(f'繁体分词result-->{result}')



# todo:jieba支持用户自定义词典
def dm05_test():
    # 1,不采用词典来进行分词
    sentence = '传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能'
    result1 = jieba.lcut(sentence)
    print(f'不经过自定义词典的分词结果--》{result1}')

    # 2，经过自定义词典
    jieba.load_userdict('./user_dict.txt')
    result2 = jieba.lcut(sentence)
    print(f'经过自定义词典的分词结果--》{result2}')

if __name__ == '__main__':
    # dm01_test()
    # dm02_test()
    # dm03_test()
    # dm04_test()
    dm05_test()