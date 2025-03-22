import torch
import jieba
import torch.nn as nn


def dm01():
	# 一句话包含多个词
	text = '北京冬奥的进度条已经过半，不少外国运动员在完成自己的比赛后踏上归途。'
	# 使用jieba模块进行分词
	words = jieba.lcut(text)
	# 返回词的列表
	print('words->', words)
	# 创建词嵌入层
	# num_embeddings:词数量
	# embedding_dim:词向量维度
	embed = nn.Embedding(num_embeddings=len(words), embedding_dim=8)
	# 获取每个词对象的下标索引
	for i, word in enumerate(words):
		# 将词索引转换成张量对象 向量
		word_vec = embed(torch.tensor(data=i))
		print('word_vec->', word_vec)


if __name__ == '__main__':
	dm01()
