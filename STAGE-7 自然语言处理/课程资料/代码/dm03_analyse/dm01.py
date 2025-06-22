# coding：utf-8
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain
import jieba
import jieba.posseg as pseg
from wordcloud import WordCloud
# todo：标签数量分布
def dm1_labels_counts():
    # 1 设置显示风格plt.style.use('fivethirtyeight')
    plt.style.use('fivethirtyeight')
    # 读取数据
    train_data = pd.read_csv('./cn_data/train.tsv', sep='\t')
    # print(f'train_data--》{train_data}')
    dev_data = pd.read_csv('./cn_data/dev.tsv', sep='\t')
    # print(f'dev_data--》{dev_data}')
    # 画图
    sns.countplot(x='label', data=train_data)
    # sns.countplot(y='label', data=train_data, hue='label')
    plt.title("train_data")
    plt.show()



    # 画图
    sns.countplot(x='label', data=dev_data)
    # sns.countplot(y='label', data=dev_data, hue='label')
    plt.title("dev_data")
    plt.show()

# todo：句子长度分布
def dm2_length():
    # 1 设置显示风格plt.style.use('fivethirtyeight')
    plt.style.use('fivethirtyeight')
    # 读取数据
    train_data = pd.read_csv('./cn_data/train.tsv', sep='\t')
    train_data["sentence_length"] = list(map(lambda x: len(x), train_data["sentence"]))
    dev_data = pd.read_csv('./cn_data/dev.tsv', sep='\t')
    dev_data["sentence_length"] = list(map(lambda x: len(x), dev_data["sentence"]))

    # # print(f'dev_data--》{dev_data}')
    # # 训练集句子长度画图--柱状图
    sns.countplot(x='sentence_length', data=train_data, hue='label')
    plt.xticks([])
    plt.title("train_data")
    plt.show()

    # # 训练集句子长度画图--曲线图
    sns.displot(x='sentence_length', data=train_data, kind='kde')
    plt.xticks([])
    plt.title("train_data")
    plt.show()

    # # 验证集句子长度画图--柱状图
    sns.countplot(x='sentence_length', data=dev_data,)
    plt.xticks([])
    plt.title("dev_data")
    plt.show()

    # # 验证集句子长度画图--曲线图
    # sns.displot(x='sentence_length', data=dev_data)
    sns.displot(x='sentence_length', data=dev_data, kde=True, bins=30)
    plt.xticks([])
    plt.title("dev_data")
    plt.show()

#todo:正负样本句子长度散点图
def dm3_striplot():
    # 1 设置显示风格plt.style.use('fivethirtyeight')
    plt.style.use('fivethirtyeight')
    # 2 读取数据
    train_data = pd.read_csv('./cn_data/train.tsv', sep='\t')
    train_data["sentence_length"] = list(map(lambda x: len(x), train_data["sentence"]))
    dev_data = pd.read_csv('./cn_data/dev.tsv', sep='\t')
    dev_data["sentence_length"] = list(map(lambda x: len(x), dev_data["sentence"]))
    # 3 画散点图
    sns.stripplot(y="sentence_length", x='label', data=train_data, hue='label')
    plt.title("train_data")
    plt.show()
    sns.stripplot(y="sentence_length", x='label', data=dev_data)
    plt.title("dev_data")
    plt.show()

# todo:获取不同词汇的总量

def dm4_vocabs():
    # 1 读取数据
    train_data = pd.read_csv('./cn_data/train.tsv', sep='\t')
    dev_data = pd.read_csv('./cn_data/dev.tsv', sep='\t')
    # 2 统计词汇数量
    train_vocab = set(chain(*map(lambda x: jieba.lcut(x), train_data["sentence"])))
    print(f'train_vocab-->{len(train_vocab)}')
    dev_vocab = set(chain(*map(lambda x: jieba.lcut(x), dev_data["sentence"])))
    print(f'dev_vocab-->{len(dev_vocab)}')


# todo:高频词云展示
def get_a_list(x):
    r = []
    for g in pseg.lcut(x):
        if g.flag == 'a':
            r.append(g.word)
    return r

def get_word_cloud(keywords):
    # 实例化词云对象
    word_cloud = WordCloud(font_path='./cn_data/SimHei.ttf', max_words=100, background_color='white')
    # 准备数据
    keywords_str = ' '.join(keywords)
    word_cloud.generate(keywords_str)
    # 画图
    plt.figure()
    # plt.imshow(word_cloud)
    plt.imshow(word_cloud,interpolation="bilinear")
    plt.axis('off')
    plt.show()

def dm5_train_wordCloud():
    # 1 读取数据
    train_data = pd.read_csv('./cn_data/train.tsv', sep='\t')
    dev_data = pd.read_csv('./cn_data/dev.tsv', sep='\t')
    # 2 获取训练集的正样本
    p_train_data = train_data[train_data["label"] == 1]["sentence"]
    # 2 获取训练集的负样本
    n_train_data = train_data[train_data["label"] == 0]["sentence"]
    # print(p_train_data)
    # 3 获取训练集正样本所有的形容词
    p_train_a_vocab = list(chain(*map(lambda x: get_a_list(x), p_train_data)))
    # print(f'p_train_a_vocab--》{len(p_train_a_vocab)}')
    # 4 获取训练集负样本所有的形容词
    n_train_a_vocab = list(chain(*map(lambda x: get_a_list(x), n_train_data)))
    # print(f'p_train_a_vocab--》{len(p_train_a_vocab)}')
    # 4 展示训练集p_train_data词云
    get_word_cloud(p_train_a_vocab)
    get_word_cloud(n_train_a_vocab)

def dm5_dev_wordCloud():
    # 1 读取数据
    dev_data = pd.read_csv('./cn_data/dev.tsv', sep='\t')
    # 2 获取验证集的正样本
    p_dev_data = dev_data[dev_data["label"] == 1]["sentence"]
    # 2 获取验证集的负样本
    n_dev_data = dev_data[dev_data["label"] == 0]["sentence"]
    # print(p_train_data)
    # 3 获取验证集正样本所有的形容词
    p_dev_a_vocab = list(chain(*map(lambda x: get_a_list(x), p_dev_data)))
    # 4 获取验证集负样本所有的形容词
    n_dev_a_vocab = list(chain(*map(lambda x: get_a_list(x), n_dev_data)))
    # print(f'p_train_a_vocab--》{len(p_train_a_vocab)}')
    # 4 展示训练集p_train_data词云
    get_word_cloud(p_dev_a_vocab)
    get_word_cloud(n_dev_a_vocab)
if __name__ == '__main__':
    # dm1_labels_counts()
    # dm2_length()
    # dm3_striplot()
    # dm4_vocabs()
    # dm5_train_wordCloud()
    dm5_dev_wordCloud()