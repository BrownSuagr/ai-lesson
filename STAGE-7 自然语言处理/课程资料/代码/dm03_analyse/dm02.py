# coding:utf-8
from keras.preprocessing import sequence
# from tensorflow.keras.preprocessing import sequence
#todo: N-gram特征
n_range = 2


def dm_n_gram(input_list):
    return set(zip(*[input_list[i:] for i in range(n_range)]))

# todo: 文本长度规范

cut_len = 10 # 句子长度分布90%的样本符合的长度
def padding(x_train):
    # padding:填充，truncating截断-->post从后面；默认pre从前面
    result = sequence.pad_sequences(x_train, cut_len, padding='post', truncating='post')

    return result


def my_padding(x_train):
    new_list = []
    max_len = 10
    for x in x_train:
        if len(x) >= max_len:
            new_list.append(x[:10])
        else:
            counts = max_len - len(x)
            list1 = x + [0]*counts
            new_list.append(list1)
    return new_list

if __name__ == '__main__':
    # 假定x_train里面有两条文本, 一条长度大于10, 一天小于10
    x_train = [[1, 23, 5, 32, 55, 63, 2, 21, 78, 32, 23, 1],
               [2, 32, 1, 23, 1]]
    # res = padding(x_train)
    res = my_padding(x_train)

    print(res)