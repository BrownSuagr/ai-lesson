# coding:utf-8
import jieba
# 导入keras中的词汇映射器Tokenizer
# from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
# 导入用于对象保存与加载的joblib
import joblib


# 定义one-hot编码生成函数
def dm_onehot_gen():
    # 准备语料
    vocabs = {"周杰伦", "陈奕迅", "王力宏", "李宗盛", "吴亦凡", "鹿晗"}
    # 实例化分词器Tokenizer
    my_tokenizer = Tokenizer()
    my_tokenizer.fit_on_texts(vocabs)
    print(my_tokenizer.word_index)
    print(my_tokenizer.index_word)
    # 生成one-hot
    for vocab in vocabs:
        zero_list = [0]*len(vocabs)
        idx = my_tokenizer.word_index[vocab] - 1
        zero_list[idx] = 1
        print(f'当前单词{vocab}的one-hot编码是--》{zero_list}')
    # 保存训练好的分词器
    my_path = './my_tokenizer'
    joblib.dump(my_tokenizer, my_path)
    print(f'分词器保存完毕')


def dm_one_hot_01():
    vocabs = {"周杰伦", "陈奕迅", "王力宏", "李宗盛", "吴亦凡", "鹿晗"}
    # 生成一个word_index的字典
    word_index = {word: i for i, word in enumerate(vocabs)}
    print(word_index)
    # 生成one-hot
    for vocab in vocabs:
        zero_list = [0]*len(vocabs)
        idx = word_index[vocab]
        zero_list[idx] = 1
        print(f'当前单词{vocab}的one-hot编码是--》{zero_list}')


# one-hot编码的使用
def dm_onehot_use():
    vocabs = {"周杰伦", "陈奕迅", "王力宏", "李宗盛", "吴亦凡", "鹿晗"}
    # 加载训练好的分词器
    my_tokenizer = joblib.load('./my_tokenizer')
    # 指定查询的单词
    token = '李宗盛'
    # token = '狗儿'# 如果字典里面不存在的词报错
    zero_list = [0]*len(vocabs)
    idx = my_tokenizer.word_index[token] - 1
    zero_list[idx] = 1
    print(f'当前单词{token}, one-hot编码是--》{zero_list}')


if __name__ == '__main__':
    # dm_onehot_gen()
    # dm_one_hot_01()
    dm_onehot_use()