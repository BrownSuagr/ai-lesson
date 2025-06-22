# coding:utf-8
import fasttext

##  原始语料进行训练
# model = fasttext.train_supervised('./fasttext_data/cooking.train')
#
# result = model.predict("Which baking dish is best to bake a banana bread ?")
# # result = model.predict("Why not put knives in the dishwasher?")
# print(f'result--》{result}')
#
# result1 = model.test('./fasttext_data/cooking.valid')
# print(f'result1模型测试效果-->{result1}')

#todo: 优化1：清洗数据 清洗完数据后，模型效果
# 原始语料进行训练
# model = fasttext.train_supervised('./fasttext_data/cooking_pre.train')
#
# result = model.predict("Which baking dish is best to bake a banana bread ?")
# # result = model.predict("Why not put knives in the dishwasher?")
# print(f'result--》{result}')
#
# result1 = model.test('./fasttext_data/cooking_pre.valid')
# print(f'修改数据后模型测试效果-->{result1}')

# #todo: 优化2：清洗数据+修改轮次
# # 原始语料进行训练
# model = fasttext.train_supervised('./fasttext_data/cooking_pre.train', epoch=25)
#
# result1 = model.test('./fasttext_data/cooking_pre.valid')
# print(f'修改数据后+轮次模型测试效果-->{result1}')

# #todo: 优化3：清洗数据+修改轮次+修改学习率
# # 原始语料进行训练
# model = fasttext.train_supervised('./fasttext_data/cooking_pre.train', epoch=25, lr=1.0)
#
# result1 = model.test('./fasttext_data/cooking_pre.valid')
# print(f'清洗数据+修改轮次+修改学习率-->{result1}')
# #todo: 优化4：清洗数据+修改轮次+修改学习率+添加N_gram特征
# # 原始语料进行训练
# model = fasttext.train_supervised('./fasttext_data/cooking_pre.train', epoch=25, lr=1.0, wordNgrams=2)
#
# result1 = model.test('./fasttext_data/cooking_pre.valid')
# print(f'清洗数据+修改轮次+修改学习率+添加N_gram特征-->{result1}')
# #todo: 优化5：清洗数据+修改轮次+修改学习率+添加N_gram特征+修改损失方式
# model = fasttext.train_supervised('./fasttext_data/cooking_pre.train', epoch=25, lr=1.0, wordNgrams=2, loss='hs')
#
# result1 = model.test('./fasttext_data/cooking_pre.valid')
# print(f'清洗数据+修改轮次+修改学习率+添加N_gram特征+修改损失方式-->{result1}')

# #todo: 优化6：自动超参数调优
# model = fasttext.train_supervised('./fasttext_data/cooking_pre.train',
#                                   autotuneValidationFile='./fasttext_data/cooking_pre.valid',
#                                   autotuneDuration=60)
#
# result1 = model.test('./fasttext_data/cooking_pre.valid')
# print(f'自动超参数调优-->{result1}')
#todo: 7：设置loss为多个二分类器输出
# model = fasttext.train_supervised('./fasttext_data/cooking_pre.train',
#                                   epoch=25, lr=0.2, wordNgrams=2, loss='ova')
#
# result1 = model.test('./fasttext_data/cooking_pre.valid')
# print(f'loss为多个二分类器输出-->{result1}')
# model.save_model('./fasttext_data/cooking.bin')
model = fasttext.load_model("./fasttext_data/cooking.bin")
result = model.predict("Which baking dish is best to bake a banana bread ?", k=-1, threshold=0.5)
print(result)



