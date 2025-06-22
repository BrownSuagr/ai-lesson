# def fun(n):
#     a = n + 1
#     return a
#
# list1 = [1, 2, 3]
#
# # c = map(fun, list1)
# c = map(lambda x: x+1, list1)
# print(list(c))

from itertools import chain

# list1 = [1,2, 3]
# list2 = [2, 3, 4]
# list3 = [2, 3, 4, 6]
# a = chain(list1, list2, list3)
# print(list(a))

# a = map(lambda x: [x+1], list1)
# # print(list(a))
# b = chain(*a)
# print(list(b))
# import jieba.posseg as pseg
#
# word_list = pseg.lcut("交通很方便，房间小了一点，但是干净整洁，很有香港的特色，性价比较高，推荐一下哦")
# print(word_list)
# for g in word_list:
#     if g.flag == 'a':
#         print(g.word)
alist = [1, 2, 3]
list1 = [alist[i:] for i in range(2)]
print(list1)


blist = [1, 3, 5]
# clist = [2, 4, 6, 1]
# dlist = [2, 4, 6, 1, 2]
# elist = [2, 4, 6, 10, 20, 30]
# a = zip(blist, clist, dlist, elist)
# print(list(a))
a = zip(*list1)
print(list(a))
