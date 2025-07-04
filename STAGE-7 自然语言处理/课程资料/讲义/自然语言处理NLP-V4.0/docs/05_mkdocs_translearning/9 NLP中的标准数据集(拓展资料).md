### 学习目标

* 了解NLP中GLUE标准数据集合的相关知识.
* 掌握GLUE标准数据集合的下载方式, 数据样式及其对应的任务类型.



## 1 GLUE数据集合介绍

### 1.1 数据集合介绍

GLUE由纽约大学, 华盛顿大学, Google联合推出, 涵盖不同NLP任务类型, 截止至2020年1月其中包括11个子任务数据集, 成为衡量NLP研究发展的衡量标准. 

* CoLA 数据集
* SST-2 数据集
* MRPC 数据集
* STS-B 数据集
* QQP 数据集
* MNLI 数据集
* SNLI 数据集
* QNLI 数据集
* RTE 数据集
* WNLI 数据集
* diagnostics数据集(官方未完善)

### 1.2 数据集合路径

数据集在虚拟机/root/data/glue_data下

另外这GLUE的11个数据集都放到的百度云 , 需要的可以自取: [GLUE数据集](https://pan.baidu.com/s/15OoSn0dbXWchOMPyLgyyQA )提取码: b6se

## 2 GLUE子数据集的样式及其任务类型

### 2.1 CoLA数据集文件样式

- 数据集释义:CoLA(The Corpus of Linguistic Acceptability，语言可接受性语料库)纽约大学发布的有关语法的数据集
- 本质: 是对一个给定句子，判定其是否语法正确的单个句子的**文本二分类任务**.

```
- CoLA/
	- dev.tsv  
	- original/
  	- test.tsv  
	- train.tsv
```



> * 文件样式说明:
> * 在使用中常用到的文件是train.tsv, dev.tsv, test.tsv, 分别代表训练集, 验证集和测试集. 其中train.tsv与dev.tsv数据样式相同, 都是带有标签的数据, 其中test.tsv是不带有标签的数据. 




> * train.tsv数据样式:

```text
...
gj04	1		She coughed herself awake as the leaf landed on her nose.
gj04	1		The worm wriggled onto the carpet.
gj04	1		The chocolate melted onto the carpet.
gj04	0	*	The ball wriggled itself loose.
gj04	1		Bill wriggled himself loose.
bc01	1		The sinking of the ship to collect the insurance was very devious.
bc01	1		The ship's sinking was very devious.
bc01	0	*	The ship's sinking to collect the insurance was very devious.
bc01	1		The testing of such drugs on oneself is too risky.
bc01	0	*	This drug's testing on oneself is too risky.
...
```



> * train.tsv数据样式说明: 
> * train.tsv中的数据内容共分为4列, 第一列数据, 如gj04, bc01等代表每条文本数据的来源即出版物代号;  第二列数据, 0或1, 代表每条文本数据的语法是否正确, 0代表不正确, 1代表正确; 第三列数据, '*', 是作者最初的正负样本标记, 与第二列意义相同, '*'表示不正确;
>   第四列即是被标注的语法使用是否正确的文本句子.



> * test.tsv数据样式:

```text
index	sentence
0	Bill whistled past the house.
1	The car honked its way down the road.
2	Bill pushed Harry off the sofa.
3	the kittens yawned awake and played.
4	I demand that the more John eats, the more he pay.
5	If John eats more, keep your mouth shut tighter, OK?
6	His expectations are always lower than mine are.
7	The sooner you call, the more carefully I will word the letter.
8	The more timid he feels, the more people he interviews without asking questions of.
9	Once Janet left, Fred became a lot crazier.
...
```



> * test.tsv数据样式说明:
> * test.tsv中的数据内容共分为2列, 第一列数据代表每条文本数据的索引;  第二列数据代表用于测试的句子.



> * CoLA数据集的任务类型:
> * 二分类任务
> * 评估指标为: MCC(马修斯相关系数, 在正负样本分布十分不均衡的情况下使用的二分类评估指标)



### 2.2 SST-2数据集文件样式

- 数据集释义:SST-2(The Stanford Sentiment Treebank，斯坦福情感树库),单句子分类任务，包含电影评论中的句子和它们情感的人类注释.
- 本质:句子级别的**二分类任务**

```
- SST-2/
        - dev.tsv
        - original/
        - test.tsv
        - train.tsv
```



> * 文件样式说明:
> * 在使用中常用到的文件是train.tsv, dev.tsv, test.tsv, 分别代表训练集, 验证集和测试集. 其中train.tsv与dev.tsv数据样式相同, 都是带有标签的数据, 其中test.tsv是不带有标签的数据.




> * train.tsv数据样式:

```text
sentence	label
hide new secretions from the parental units 	0
contains no wit , only labored gags 	0
that loves its characters and communicates something rather beautiful about human nature 	1
remains utterly satisfied to remain the same throughout 	0
on the worst revenge-of-the-nerds clichés the filmmakers could dredge up 	0
that 's far too tragic to merit such superficial treatment 	0
demonstrates that the director of such hollywood blockbusters as patriot games can still turn out a small , personal film with an emotional wallop . 	1
of saucy 	1
a depressed fifteen-year-old 's suicidal poetry 	0
...
```



> * train.tsv数据样式说明:
> * train.tsv中的数据内容共分为2列, 第一列数据代表具有感情色彩的评论文本;  第二列数据, 0或1, 代表每条文本数据是积极或者消极的评论, 0代表消极, 1代表积极.



> * test.tsv数据样式:

```text
index	sentence
0	uneasy mishmash of styles and genres .
1	this film 's relationship to actual tension is the same as what christmas-tree flocking in a spray can is to actual snow : a poor -- if durable -- imitation .
2	by the end of no such thing the audience , like beatrice , has a watchful affection for the monster .
3	director rob marshall went out gunning to make a great one .
4	lathan and diggs have considerable personal charm , and their screen rapport makes the old story seem new .
5	a well-made and often lovely depiction of the mysteries of friendship .
6	none of this violates the letter of behan 's book , but missing is its spirit , its ribald , full-throated humor .
7	although it bangs a very cliched drum at times , this crowd-pleaser 's fresh dialogue , energetic music , and good-natured spunk are often infectious .
8	it is not a mass-market entertainment but an uncompromising attempt by one artist to think about another .
9	this is junk food cinema at its greasiest .
...
```



> * test.tsv数据样式说明:
>       * test.tsv中的数据内容共分为2列, 第一列数据代表每条文本数据的索引;  第二列数据代表用于测试的句子.



> * SST-2数据集的任务类型:
> * 二分类任务
> * 评估指标为: ACC





### 2.3 MRPC数据集文件样式

- 数据集释义:MRPC(The Microsoft Research Paraphrase Corpus，微软研究院释义语料库),相似性和释义任务，是从在线新闻源中自动抽取句子对语料库，并人工注释句子对中的句子是否在语义上等效。
- 本质:句子级别的**二分类任务**

```
- MRPC/
        - dev.tsv
        - test.tsv
        - train.tsv
	- dev_ids.tsv
	- msr_paraphrase_test.txt
	- msr_paraphrase_train.txt
```



> * 文件样式说明:
> * 在使用中常用到的文件是train.tsv, dev.tsv, test.tsv, 分别代表训练集, 验证集和测试集. 其中train.tsv与dev.tsv数据样式相同, 都是带有标签的数据, 其中test.tsv是不带有标签的数据.




> * train.tsv数据样式:

```text
Quality	#1 ID	#2 ID	#1 String	#2 String
1	702876	702977	Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .	Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .
0	2108705	2108831	Yucaipa owned Dominick 's before selling the chain to Safeway in 1998 for $ 2.5 billion .	Yucaipa bought Dominick 's in 1995 for $ 693 million and sold it to Safeway for $ 1.8 billion in 1998 .
1	1330381	1330521	They had published an advertisement on the Internet on June 10 , offering the cargo for sale , he added .	On June 10 , the ship 's owners had published an advertisement on the Internet , offering the explosives for sale .
0	3344667	3344648	Around 0335 GMT , Tab shares were up 19 cents , or 4.4 % , at A $ 4.56 , having earlier set a record high of A $ 4.57 .	Tab shares jumped 20 cents , or 4.6 % , to set a record closing high at A $ 4.57 .
1	1236820	1236712	The stock rose $ 2.11 , or about 11 percent , to close Friday at $ 21.51 on the New York Stock Exchange .	PG & E Corp. shares jumped $ 1.63 or 8 percent to $ 21.03 on the New York Stock Exchange on Friday .
1	738533	737951	Revenue in the first quarter of the year dropped 15 percent from the same period a year earlier .	With the scandal hanging over Stewart 's company , revenue the first quarter of the year dropped 15 percent from the same period a year earlier .
0	264589	264502	The Nasdaq had a weekly gain of 17.27 , or 1.2 percent , closing at 1,520.15 on Friday .	The tech-laced Nasdaq Composite .IXIC rallied 30.46 points , or 2.04 percent , to 1,520.15 .
1	579975	579810	The DVD-CCA then appealed to the state Supreme Court .	The DVD CCA appealed that decision to the U.S. Supreme Court .
...
```



> * train.tsv数据样式说明:
> * train.tsv中的数据内容共分为5列, 第一列数据, 0或1, 代表每对句子是否具有相同的含义, 0代表含义不相同, 1代表含义相同. 第二列和第三列分别代表每对句子的id, 第四列和第五列分别具有相同/不同含义的句子对.



> * test.tsv数据样式:

```text
index	#1 ID	#2 ID	#1 String	#2 String
0	1089874	1089925	PCCW 's chief operating officer , Mike Butcher , and Alex Arena , the chief financial officer , will report directly to Mr So .	Current Chief Operating Officer Mike Butcher and Group Chief Financial Officer Alex Arena will report to So .
1	3019446	3019327	The world 's two largest automakers said their U.S. sales declined more than predicted last month as a late summer sales frenzy caused more of an industry backlash than expected .	Domestic sales at both GM and No. 2 Ford Motor Co. declined more than predicted as a late summer sales frenzy prompted a larger-than-expected industry backlash .
2	1945605	1945824	According to the federal Centers for Disease Control and Prevention ( news - web sites ) , there were 19 reported cases of measles in the United States in 2002 .	The Centers for Disease Control and Prevention said there were 19 reported cases of measles in the United States in 2002 .
3	1430402	1430329	A tropical storm rapidly developed in the Gulf of Mexico Sunday and was expected to hit somewhere along the Texas or Louisiana coasts by Monday night .	A tropical storm rapidly developed in the Gulf of Mexico on Sunday and could have hurricane-force winds when it hits land somewhere along the Louisiana coast Monday night .
4	3354381	3354396	The company didn 't detail the costs of the replacement and repairs .	But company officials expect the costs of the replacement work to run into the millions of dollars .
5	1390995	1391183	The settling companies would also assign their possible claims against the underwriters to the investor plaintiffs , he added .	Under the agreement , the settling companies will also assign their potential claims against the underwriters to the investors , he added .
6	2201401	2201285	Air Commodore Quaife said the Hornets remained on three-minute alert throughout the operation .	Air Commodore John Quaife said the security operation was unprecedented .
7	2453843	2453998	A Washington County man may have the countys first human case of West Nile virus , the health department said Friday .	The countys first and only human case of West Nile this year was confirmed by health officials on Sept . 8 .
...
```



> * test.tsv数据样式说明:
>       * test.tsv中的数据内容共分为5列, 第一列数据代表每条文本数据的索引; 其余列的含义与train.tsv中相同.



> * MRPC数据集的任务类型:
> * 句子对二分类任务
> * 评估指标为: ACC和F1



### 2.4 STS-B数据集文件样式

- 数据集释义: STSB(The Semantic Textual Similarity Benchmark，语义文本相似性基准测试)
- 本质: 回归任务/句子对的文本五分类任务

```
- STS-B/
        - dev.tsv
        - test.tsv
        - train.tsv
	- LICENSE.txt
	- readme.txt
	- original/
```



> * 文件样式说明:
> * 在使用中常用到的文件是train.tsv, dev.tsv, test.tsv, 分别代表训练集, 验证集和测试集. 其中train.tsv与dev.tsv数据样式相同, 都是带有标签的数据, 其中test.tsv是不带有标签的数据.




> * train.tsv数据样式:

```text
index	genre	filename	year	old_index	source1	source2	sentence1	sentence2	score
0	main-captions	MSRvid	2012test	0001	none	none	A plane is taking off.	An air plane is taking off.	5.000
1	main-captions	MSRvid	2012test	0004	none	none	A man is playing a large flute.	A man is playing a flute.	3.800
2	main-captions	MSRvid	2012test	0005	none	none	A man is spreading shreded cheese on a pizza.	A man is spreading shredded cheese on an uncooked pizza.	3.800
3	main-captions	MSRvid	2012test	0006	none	none	Three men are playing chess.Two men are playing chess.	2.600
4	main-captions	MSRvid	2012test	0009	none	none	A man is playing the cello.A man seated is playing the cello.	4.250
5	main-captions	MSRvid	2012test	0011	none	none	Some men are fighting.	Two men are fighting.	4.250
6	main-captions	MSRvid	2012test	0012	none	none	A man is smoking.	A man is skating.	0.500
7	main-captions	MSRvid	2012test	0013	none	none	The man is playing the piano.	The man is playing the guitar.	1.600
8	main-captions	MSRvid	2012test	0014	none	none	A man is playing on a guitar and singing.	A woman is playing an acoustic guitar and singing.	2.200
9	main-captions	MSRvid	2012test	0016	none	none	A person is throwing a cat on to the ceiling.	A person throws a cat on the ceiling.	5.000
...
```



> * train.tsv数据样式说明:
> * train.tsv中的数据内容共分为10列, 第一列数据是数据索引; 第二列代表每对句子的来源, 如main-captions表示来自字幕; 第三列代表来源的具体保存文件名, 第四列代表出现时间(年); 第五列代表原始数据的索引; 第六列和第七列分别代表句子对原始来源; 第八列和第九列代表相似程度不同的句子对; 第十列代表句子对的相似程度由低到高, 值域范围是[0, 5].



> * test.tsv数据样式:

```text
index	genre	filename	year	old_index	source1	source2	sentence1	sentence2
0	main-captions	MSRvid	2012test	0024	none	none	A girl is styling her hair.	A girl is brushing her hair.
1	main-captions	MSRvid	2012test	0033	none	none	A group of men play soccer on the beach.	A group of boys are playing soccer on the beach.
2	main-captions	MSRvid	2012test	0045	none	none	One woman is measuring another woman's ankle.	A woman measures another woman's ankle.
3	main-captions	MSRvid	2012test	0063	none	none	A man is cutting up a cucumber.	A man is slicing a cucumber.
4	main-captions	MSRvid	2012test	0066	none	none	A man is playing a harp.	A man is playing a keyboard.
5	main-captions	MSRvid	2012test	0074	none	none	A woman is cutting onions.	A woman is cutting tofu.
6	main-captions	MSRvid	2012test	0076	none	none	A man is riding an electric bicycle.	A man is riding a bicycle.
7	main-captions	MSRvid	2012test	0082	none	none	A man is playing the drums.	A man is playing the guitar.
8	main-captions	MSRvid	2012test	0092	none	none	A man is playing guitar.	A lady is playing the guitar.
9	main-captions	MSRvid	2012test	0095	none	none	A man is playing a guitar.	A man is playing a trumpet.
10	main-captions	MSRvid	2012test	0096	none	none	A man is playing a guitar.	A man is playing a trumpet.
...
```



> * test.tsv数据样式说明:
> * test.tsv中的数据内容共分为9列, 含义与train.tsv前9列相同.



> * STS-B数据集的任务类型:
> * 句子对多分类任务/句子对回归任务
> * 评估指标为: Pearson-Spearman Corr





### 2.5 QQP数据集文件样式

- 数据集释义: QQP(The Quora Question Pairs, Quora问题对数集),相似性和释义任务，是社区问答网站Quora中问题对的集合。
- 本质: 句子对的**二分类任务**

```
- QQP/
        - dev.tsv
        - original/
        - test.tsv
        - train.tsv
```



> * 文件样式说明:
> * 在使用中常用到的文件是train.tsv, dev.tsv, test.tsv, 分别代表训练集, 验证集和测试集. 其中train.tsv与dev.tsv数据样式相同, 都是带有标签的数据, 其中test.tsv是不带有标签的数据.




> * train.tsv数据样式:

```text
id	qid1	qid2	question1	question2	is_duplicate
133273	213221	213222	How is the life of a math student? Could you describe your own experiences?Which level of prepration is enough for the exam jlpt5?	0
402555	536040	536041	How do I control my horny emotions?	How do you control your horniness?	1
360472	364011	490273	What causes stool color to change to yellow?	What can cause stool to come out as little balls?	0
150662	155721	7256	What can one do after MBBS?	What do i do after my MBBS ?	1
183004	279958	279959	Where can I find a power outlet for my laptop at Melbourne Airport?	Would a second airport in Sydney, Australia be needed if a high-speed rail link was created between Melbourne and Sydney?	0
119056	193387	193388	How not to feel guilty since I am Muslim and I'm conscious we won't have sex together?	I don't beleive I am bulimic, but I force throw up atleast once a day after I eat something and feel guilty. Should I tell somebody, and if so who?	0
356863	422862	96457	How is air traffic controlled?	How do you become an air traffic controller?0
106969	147570	787	What is the best self help book you have read? Why? How did it change your life?	What are the top self help books I should read?	1
...
```



> * train.tsv数据样式说明:
> * train.tsv中的数据内容共分为6列, 第一列代表文本数据索引;  第二列和第三列数据分别代表问题1和问题2的id; 第四列和第五列代表需要进行'是否重复'判定的句子对; 第六列代表上述问题是/不是重复性问题的标签, 0代表不重复, 1代表重复.



> * test.tsv数据样式:

```text
id	question1	question2
0	Would the idea of Trump and Putin in bed together scare you, given the geopolitical implications?	Do you think that if Donald Trump were elected President, he would be able to restore relations with Putin and Russia as he said he could, based on the rocky relationship Putin had with Obama and Bush?
1	What are the top ten Consumer-to-Consumer E-commerce online?	What are the top ten Consumer-to-Business E-commerce online?
2	Why don't people simply 'Google' instead of asking questions on Quora?	Why do people ask Quora questions instead of just searching google?
3	Is it safe to invest in social trade biz?	Is social trade geniune?
4	If the universe is expanding then does matter also expand?	If universe and space is expanding? Does that mean anything that occupies space is also expanding?
5	What is the plural of hypothesis?	What is the plural of thesis?
6	What is the application form you need for launching a company?	What is the application form you need for launching a company in Austria?
7	What is Big Theta? When should I use Big Theta as opposed to big O?	Is O(Log n) close to O(n) or O(1)?
8	What are the health implications of accidentally eating a small quantity of aluminium foil?What are the implications of not eating vegetables?
...
```



> * test.tsv数据样式说明:
> * test.tsv中的数据内容共分为3列, 第一列数据代表每条文本数据的索引;  第二列和第三列数据代表用于测试的问题句子对.



> * QQP数据集的任务类型:
> * 句子对二分类任务
> * 评估指标为: ACC/F1





### 2.6 (MNLI/SNLI)数据集文件样式

- 数据集释义: 
  - MNLI(The Multi-Genre Natural Language Inference Corpus, 多类型自然语言推理数据库)
- 本质: 句子对的**三分类任务**

```
- (MNLI/SNLI)/
	- dev_matched.tsv
	- dev_mismatched.tsv
	- original/
	- test_matched.tsv
	- test_mismatched.tsv
	- train.tsv
```



> * 文件样式说明:
> * 在使用中常用到的文件是train.tsv, dev_matched.tsv, dev_mismatched.tsv, test_matched.tsv, test_mismatched.tsv分别代表训练集, 与训练集一同采集的验证集, 与训练集不是一同采集验证集,  与训练集一同采集的测试集, 与训练集不是一同采集测试集. 其中train.tsv与dev_matched.tsv和dev_mismatched.tsv数据样式相同, 都是带有标签的数据, 其中test_matched.tsv与test_mismatched.tsv数据样式相同, 都是不带有标签的数据.




> * train.tsv数据样式:

```text
index	promptID	pairID	genre	sentence1_binary_parse	sentence2_binary_parse	sentence1_parse	sentence2_parse	sentence1	sentence2	label1	gold_label
0	31193	31193n	government	( ( Conceptually ( cream skimming ) ) ( ( has ( ( ( two ( basic dimensions ) ) - ) ( ( product and ) geography ) ) ) . ) )	( ( ( Product and ) geography ) ( ( are ( what ( make ( cream ( skimming work ) ) ) ) ) . ) )	(ROOT (S (NP (JJ Conceptually) (NN cream) (NN skimming)) (VP (VBZ has) (NP (NP (CD two) (JJ basic) (NNS dimensions)) (: -) (NP (NN product) (CC and) (NN geography)))) (. .)))	(ROOT (S (NP (NN Product) (CC and) (NN geography)) (VP (VBP are) (SBAR (WHNP (WP what)) (S (VP (VBP make) (NP (NP (NN cream)) (VP (VBG skimming) (NP (NN work)))))))) (. .)))	Conceptually cream skimming has two basic dimensions - product and geography.	Product and geography are what make cream skimming work. 	neutral	neutral
1	101457	101457e	telephone	( you ( ( know ( during ( ( ( the season ) and ) ( i guess ) ) ) ) ( at ( at ( ( your level ) ( uh ( you ( ( ( lose them ) ( to ( the ( next level ) ) ) ) ( if ( ( if ( they ( decide ( to ( recall ( the ( the ( parent team ) ) ) ) ) ) ) ) ( ( the Braves ) ( decide ( to ( call ( to ( ( recall ( a guy ) ) ( from ( ( triple A ) ( ( ( then ( ( a ( double ( A guy ) ) ) ( ( goes up ) ( to ( replace him ) ) ) ) ) and ) ( ( a ( single ( A guy ) ) ) ( ( goes up ) ( to ( replace him ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) )	( You ( ( ( ( lose ( the things ) ) ( to ( the ( following level ) ) ) ) ( if ( ( the people ) recall ) ) ) . ) )	(ROOT (S (NP (PRP you)) (VP (VBP know) (PP (IN during) (NP (NP (DT the) (NN season)) (CC and) (NP (FW i) (FW guess)))) (PP (IN at) (IN at) (NP (NP (PRP$ your) (NN level)) (SBAR (S (INTJ (UH uh)) (NP (PRP you)) (VP (VBP lose) (NP (PRP them)) (PP (TO to) (NP (DT the) (JJ next) (NN level))) (SBAR (IN if) (S (SBAR (IN if) (S (NP (PRP they)) (VP (VBP decide) (S (VP (TO to) (VP (VB recall) (NP (DT the) (DT the) (NN parent) (NN team)))))))) (NP (DT the) (NNPS Braves)) (VP (VBP decide) (S (VP (TO to) (VP (VB call) (S (VP (TO to) (VP (VB recall) (NP (DT a) (NN guy)) (PP (IN from) (NP (NP (RB triple) (DT A)) (SBAR (S (S (ADVP (RB then)) (NP (DT a) (JJ double) (NNP A) (NN guy)) (VP (VBZ goes) (PRT (RP up)) (S (VP (TO to) (VP (VB replace) (NP (PRP him))))))) (CC and) (S (NP (DT a) (JJ single) (NNP A) (NN guy)) (VP (VBZ goes) (PRT (RP up)) (S (VP (TO to) (VP (VB replace) (NP (PRP him))))))))))))))))))))))))))))	(ROOT (S (NP (PRP You)) (VP (VBP lose) (NP (DT the) (NNS things)) (PP (TO to) (NP (DT the) (JJ following) (NN level))) (SBAR (IN if) (S (NP (DT the) (NNS people)) (VP (VBP recall))))) (. .)))	you know during the season and i guess at at your level uh you lose them to the next level if if they decide to recall the the parent team the Braves decide to call to recall a guy from triple A then a double A guy goes up to replace him and a single A guy goes up to replace him	You lose the things to the following level if the people recall.	entailment	entailment
2	134793	134793e	fiction	( ( One ( of ( our number ) ) ) ( ( will ( ( ( carry out ) ( your instructions ) ) minutely ) ) . ) )	( ( ( A member ) ( of ( my team ) ) ) ( ( will ( ( execute ( your orders ) ) ( with ( immense precision ) ) ) ) . ) )	(ROOT (S (NP (NP (CD One)) (PP (IN of) (NP (PRP$ our) (NN number)))) (VP (MD will) (VP (VB carry) (PRT (RP out)) (NP (PRP$ your) (NNS instructions)) (ADVP (RB minutely)))) (. .)))	(ROOT (S (NP (NP (DT A) (NN member)) (PP (IN of) (NP (PRP$ my) (NN team)))) (VP (MD will) (VP (VB execute) (NP (PRP$ your) (NNS orders)) (PP (IN with) (NP (JJ immense) (NN precision))))) (. .)))	One of our number will carry out your instructions minutely.	A member of my team will execute your orders with immense precision.	entailment	entailment
3	37397	37397e	fiction	( ( How ( ( ( do you ) know ) ? ) ) ( ( All this ) ( ( ( is ( their information ) ) again ) . ) ) )	( ( This information ) ( ( belongs ( to them ) ) . ) )	(ROOT (S (SBARQ (WHADVP (WRB How)) (SQ (VBP do) (NP (PRP you)) (VP (VB know))) (. ?)) (NP (PDT All) (DT this)) (VP (VBZ is) (NP (PRP$ their) (NN information)) (ADVP (RB again))) (. .)))	(ROOT (S (NP (DT This) (NN information)) (VP (VBZ belongs) (PP (TO to) (NP (PRP them)))) (. .)))	How do you know? All this is their information again.	This information belongs to them.	entailment	entailment
...
```



> * train.tsv数据样式说明:
> * train.tsv中的数据内容共分为12列, 第一列代表文本数据索引;  第二列和第三列数据分别代表句子对的不同类型id; 第四列代表句子对的来源; 第五列和第六列代表具有句法结构分析的句子对表示; 第七列和第八列代表具有句法结构和词性标注的句子对表示, 第九列和第十列代表原始的句子对, 第十一和第十二列代表不同标准的标注方法产生的标签, 在这里，他们始终相同, 一共有三种类型的标签, neutral代表两个句子既不矛盾也不蕴含, entailment代表两个句子具有蕴含关系, contradiction代表两个句子观点矛盾.



> * test_matched.tsv数据样式:

```text
index	promptID	pairID	genre	sentence1_binary_parse	sentence2_binary_parse	sentence1_parse	sentence2_parse	sentence1	sentence2
0	31493	31493	travel	( ( ( ( ( ( ( ( Hierbas , ) ( ans seco ) ) , ) ( ans dulce ) ) , ) and ) frigola ) ( ( ( are just ) ( ( a ( few names ) ) ( worth ( ( keeping ( a look-out ) ) for ) ) ) ) . ) )	( Hierbas ( ( is ( ( a name ) ( worth ( ( looking out ) for ) ) ) ) . ) )	(ROOT (S (NP (NP (NNS Hierbas)) (, ,) (NP (NN ans) (NN seco)) (, ,) (NP (NN ans) (NN dulce)) (, ,) (CC and) (NP (NN frigola))) (VP (VBP are) (ADVP (RB just)) (NP (NP (DT a) (JJ few) (NNS names)) (PP (JJ worth) (S (VP (VBG keeping) (NP (DT a) (NN look-out)) (PP (IN for))))))) (. .)))	(ROOT (S (NP (NNS Hierbas)) (VP (VBZ is) (NP (NP (DT a) (NN name)) (PP (JJ worth) (S (VP (VBG looking) (PRT (RP out)) (PP (IN for))))))) (. .)))	Hierbas, ans seco, ans dulce, and frigola are just a few names worth keeping a look-out for.	Hierbas is a name worth looking out for.
1	92164	92164	government	( ( ( The extent ) ( of ( the ( behavioral effects ) ) ) ) ( ( would ( ( depend ( in ( part ( on ( ( the structure ) ( of ( ( ( the ( individual ( account program ) ) ) and ) ( any limits ) ) ) ) ) ) ) ) ( on ( accessing ( the funds ) ) ) ) ) . ) )	( ( Many people ) ( ( would ( be ( very ( unhappy ( to ( ( loose control ) ( over ( their ( own money ) ) ) ) ) ) ) ) ) . ) )	(ROOT (S (NP (NP (DT The) (NN extent)) (PP (IN of) (NP (DT the) (JJ behavioral) (NNS effects)))) (VP (MD would) (VP (VB depend) (PP (IN in) (NP (NP (NN part)) (PP (IN on) (NP (NP (DT the) (NN structure)) (PP (IN of) (NP (NP (DT the) (JJ individual) (NN account) (NN program)) (CC and) (NP (DT any) (NNS limits)))))))) (PP (IN on) (S (VP (VBG accessing) (NP (DT the) (NNS funds))))))) (. .)))	(ROOT (S (NP (JJ Many) (NNS people)) (VP (MD would) (VP (VB be) (ADJP (RB very) (JJ unhappy) (PP (TO to) (NP (NP (JJ loose) (NN control)) (PP (IN over) (NP (PRP$ their) (JJ own) (NN money)))))))) (. .)))	The extent of the behavioral effects would depend in part on the structure of the individual account program and any limits on accessing the funds.	Many people would be very unhappy to loose control over their own money.
2	9662	9662	government	( ( ( Timely access ) ( to information ) ) ( ( is ( in ( ( the ( best interests ) ) ( of ( ( ( both GAO ) and ) ( the agencies ) ) ) ) ) ) . ) )	( It ( ( ( is ( in ( ( everyone 's ) ( best interest ) ) ) ) ( to ( ( have access ) ( to ( information ( in ( a ( timely manner ) ) ) ) ) ) ) ) . ) )	(ROOT (S (NP (NP (JJ Timely) (NN access)) (PP (TO to) (NP (NN information)))) (VP (VBZ is) (PP (IN in) (NP (NP (DT the) (JJS best) (NNS interests)) (PP (IN of) (NP (NP (DT both) (NNP GAO)) (CC and) (NP (DT the) (NNS agencies))))))) (. .)))	(ROOT (S (NP (PRP It)) (VP (VBZ is) (PP (IN in) (NP (NP (NN everyone) (POS 's)) (JJS best) (NN interest))) (S (VP (TO to) (VP (VB have) (NP (NN access)) (PP (TO to) (NP (NP (NN information)) (PP (IN in) (NP (DT a) (JJ timely) (NN manner))))))))) (. .)))	Timely access to information is in the best interests of both GAO and the agencies.	It is in everyone's best interest to have access to information in a timely manner.
3	5991	5991	travel	( ( Based ( in ( ( the ( Auvergnat ( spa town ) ) ) ( of Vichy ) ) ) ) ( , ( ( the ( French government ) ) ( often ( ( ( ( proved ( more zealous ) ) ( than ( its masters ) ) ) ( in ( ( ( suppressing ( civil liberties ) ) and ) ( ( drawing up ) ( anti-Jewish legislation ) ) ) ) ) . ) ) ) ) )	( ( The ( French government ) ) ( ( passed ( ( anti-Jewish laws ) ( aimed ( at ( helping ( the Nazi ) ) ) ) ) ) . ) )	(ROOT (S (PP (VBN Based) (PP (IN in) (NP (NP (DT the) (NNP Auvergnat) (NN spa) (NN town)) (PP (IN of) (NP (NNP Vichy)))))) (, ,) (NP (DT the) (JJ French) (NN government)) (ADVP (RB often)) (VP (VBD proved) (NP (JJR more) (NNS zealous)) (PP (IN than) (NP (PRP$ its) (NNS masters))) (PP (IN in) (S (VP (VP (VBG suppressing) (NP (JJ civil) (NNS liberties))) (CC and) (VP (VBG drawing) (PRT (RP up)) (NP (JJ anti-Jewish) (NN legislation))))))) (. .)))	(ROOT (S (NP (DT The) (JJ French) (NN government)) (VP (VBD passed) (NP (NP (JJ anti-Jewish) (NNS laws)) (VP (VBN aimed) (PP (IN at) (S (VP (VBG helping) (NP (DT the) (JJ Nazi)))))))) (. .)))	Based in the Auvergnat spa town of Vichy, the French government often proved more zealous than its masters in suppressing civil liberties and drawing up anti-Jewish legislation.	The French government passed anti-Jewish laws aimed at helping the Nazi.
...
```



> * test_matched.tsv数据样式说明:
> * test_matched.tsv中的数据内容共分为10列, 与train.tsv的前10列含义相同.



> * (MNLI/SNLI)数据集的任务类型:
> * 句子对多分类任务
> * 评估指标为: ACC



### 2.7 (QNLI/RTE/WNLI)数据集文件样式

- 数据集释义: 
  - QNLI(Qusetion-answering NLI，问答自然语言推断)，自然语言推断任务。QNLI是从另一个数据集The Stanford Question Answering Dataset(斯坦福问答数据集, SQuAD 1.0)[[3\]](https://zhuanlan.zhihu.com/p/135283598#ref_3)转换而来的.
  - RTE(The Recognizing Textual Entailment datasets，识别文本蕴含数据集)，自然语言推断任务，它是将一系列的年度文本蕴含挑战赛的数据集进行整合合并而来的.
  - WNLI(Winograd NLI，Winograd自然语言推断)，自然语言推断任务，数据集来自于竞赛数据的转换。
- 本质: QNLI是二分类任务.   RTE是二分类任务.  WNLI是二分类任务.

  * QNLI, RTE, WNLI三个数据集的样式基本相同.

```
- (QNLI/RTE/WNLI)/
        - dev.tsv
        - test.tsv
        - train.tsv
```



> * 文件样式说明:
> * 在使用中常用到的文件是train.tsv, dev.tsv, test.tsv, 分别代表训练集, 验证集和测试集. 其中train.tsv与dev.tsv数据样式相同, 都是带有标签的数据, 其中test.tsv是不带有标签的数据.




> * QNLI中的train.tsv数据样式:

```text
index	question	sentence	label
0	When did the third Digimon series begin?	Unlike the two seasons before it and most of the seasons that followed, Digimon Tamers takes a darker and more realistic approach to its story featuring Digimon who do not reincarnate after their deaths and more complex character development in the original Japanese.	not_entailment
1	Which missile batteries often have individual launchers several kilometres from one another?	When MANPADS is operated by specialists, batteries may have several dozen teams deploying separately in small sections; self-propelled air defence guns may deploy in pairs.	not_entailment
2	What two things does Popper argue Tarski's theory involves in an evaluation of truth?	He bases this interpretation on the fact that examples such as the one described above refer to two things: assertions and the facts to which they refer.	entailment
3	What is the name of the village 9 miles north of Calafat where the Ottoman forces attacked the Russians?	On 31 December 1853, the Ottoman forces at Calafat moved against the Russian force at Chetatea or Cetate, a small village nine miles north of Calafat, and engaged them on 6 January 1854.	entailment
4	What famous palace is located in London?	London contains four World Heritage Sites: the Tower of London; Kew Gardens; the site comprising the Palace of Westminster, Westminster Abbey, and St Margaret's Church; and the historic settlement of Greenwich (in which the Royal Observatory, Greenwich marks the Prime Meridian, 0° longitude, and GMT).	not_entailment
5	When is the term 'German dialects' used in regard to the German language?	When talking about the German language, the term German dialects is only used for the traditional regional varieties.	entailment
6	What was the name of the island the English traded to the Dutch in return for New Amsterdam?	At the end of the Second Anglo-Dutch War, the English gained New Amsterdam (New York) in North America in exchange for Dutch control of Run, an Indonesian island.	entailment
7	How were the Portuguese expelled from Myanmar?	From the 1720s onward, the kingdom was beset with repeated Meithei raids into Upper Myanmar and a nagging rebellion in Lan Na.	not_entailment
8	What does the word 'customer' properly apply to?	The bill also required rotation of principal maintenance inspectors and stipulated that the word "customer" properly applies to the flying public, not those entities regulated by the FAA.	entailment
...
```



> * RTE中的train.tsv数据样式:

```text
index	sentence1	sentence2	label
0	No Weapons of Mass Destruction Found in Iraq Yet.	Weapons of Mass Destruction Found in Iraq.	not_entailment
1	A place of sorrow, after Pope John Paul II died, became a place of celebration, as Roman Catholic faithful gathered in downtown Chicago to mark the installation of new Pope Benedict XVI.Pope Benedict XVI is the new leader of the Roman Catholic Church.	entailment
2	Herceptin was already approved to treat the sickest breast cancer patients, and the company said, Monday, it will discuss with federal regulators the possibility of prescribing the drug for more breast cancer patients.	Herceptin can be used to treat breast cancer.	entailment
3	Judie Vivian, chief executive at ProMedica, a medical service company that helps sustain the 2-year-old Vietnam Heart Institute in Ho Chi Minh City (formerly Saigon), said that so far about 1,500 children have received treatment.	The previous name of Ho Chi Minh City was Saigon.entailment
4	A man is due in court later charged with the murder 26 years ago of a teenager whose case was the first to be featured on BBC One's Crimewatch. Colette Aram, 16, was walking to her boyfriend's house in Keyworth, Nottinghamshire, on 30 October 1983 when she disappeared. Her body was later found in a field close to her home. Paul Stewart Hutchinson, 50, has been charged with murder and is due before Nottingham magistrates later.	Paul Stewart Hutchinson is accused of having stabbed a girl.	not_entailment
5	Britain said, Friday, that it has barred cleric, Omar Bakri, from returning to the country from Lebanon, where he was released by police after being detained for 24 hours.	Bakri was briefly detained, but was released.	entailment
6	Nearly 4 million children who have at least one parent who entered the U.S. illegally were born in the United States and are U.S. citizens as a result, according to the study conducted by the Pew Hispanic Center. That's about three quarters of the estimated 5.5 million children of illegal immigrants inside the United States, according to the study. About 1.8 million children of undocumented immigrants live in poverty, the study found.	Three quarters of U.S. illegal immigrants have children.	not_entailment
7	Like the United States, U.N. officials are also dismayed that Aristide killed a conference called by Prime Minister Robert Malval in Port-au-Prince in hopes of bringing all the feuding parties together.	Aristide had Prime Minister Robert Malval  murdered in Port-au-Prince.	not_entailment
8	WASHINGTON --  A newly declassified narrative of the Bush administration's advice to the CIA on harsh interrogations shows that the small group of Justice Department lawyers who wrote memos authorizing controversial interrogation techniques were operating not on their own but with direction from top administration officials, including then-Vice President Dick Cheney and national security adviser Condoleezza Rice. At the same time, the narrative suggests that then-Defense Secretary Donald H. Rumsfeld and then-Secretary of State Colin Powell were largely left out of the decision-making process.	Dick Cheney was the Vice President of Bush.	entailment
```



> * WNLI中的train.tsv数据样式:

```text
index	sentence1	sentence2	label
0	I stuck a pin through a carrot. When I pulled the pin out, it had a hole.	The carrot had a hole.	1
1	John couldn't see the stage with Billy in front of him because he is so short.	John is so short.	1
2	The police arrested all of the gang members. They were trying to stop the drug trade in the neighborhood.	The police were trying to stop the drug trade in the neighborhood.	1
3	Steve follows Fred's example in everything. He influences him hugely.	Steve influences him hugely.	0
4	When Tatyana reached the cabin, her mother was sleeping. She was careful not to disturb her, undressing and climbing back into her berth.	mother was careful not to disturb her, undressing and climbing back into her berth.	0
5	George got free tickets to the play, but he gave them to Eric, because he was particularly eager to see it.	George was particularly eager to see it.	0
6	John was jogging through the park when he saw a man juggling watermelons. He was very impressive.	John was very impressive.	0
7	I couldn't put the pot on the shelf because it was too tall.	The pot was too tall.	1
8	We had hoped to place copies of our newsletter on all the chairs in the auditorium, but there were simply not enough of them.	There were simply not enough copies of the newsletter.	1
```



> * (QNLI/RTE/WNLI)中的train.tsv数据样式说明:
> * train.tsv中的数据内容共分为4列, 第一列代表文本数据索引;  第二列和第三列数据代表需要进行'是否蕴含'判定的句子对; 第四列数据代表两个句子是否具有蕴含关系, 0/not_entailment代表不是蕴含关系, 1/entailment代表蕴含关系.



> * QNLI中的test.tsv数据样式:

```text
index	question	sentence
0	What organization is devoted to Jihad against Israel?	For some decades prior to the First Palestine Intifada in 1987, the Muslim Brotherhood in Palestine took a "quiescent" stance towards Israel, focusing on preaching, education and social services, and benefiting from Israel's "indulgence" to build up a network of mosques and charitable organizations.
1	In what century was the Yarrow-Schlick-Tweedy balancing system used?	In the late 19th century, the Yarrow-Schlick-Tweedy balancing 'system' was used on some marine triple expansion engines.
2	The largest brand of what store in the UK is located in Kingston Park?	Close to Newcastle, the largest indoor shopping centre in Europe, the MetroCentre, is located in Gateshead.
3	What does the IPCC rely on for research?	In principle, this means that any significant new evidence or events that change our understanding of climate science between this deadline and publication of an IPCC report cannot be included.
4	What is the principle about relating spin and space variables?	Thus in the case of two fermions there is a strictly negative correlation between spatial and spin variables, whereas for two bosons (e.g. quanta of electromagnetic waves, photons) the correlation is strictly positive.
5	Which network broadcasted Super Bowl 50 in the U.S.?	CBS broadcast Super Bowl 50 in the U.S., and charged an average of $5 million for a 30-second commercial during the game.
6	What did the museum acquire from the Royal College of Science?	To link this to the rest of the museum, a new entrance building was constructed on the site of the former boiler house, the intended site of the Spiral, between 1978 and 1982.
7	What is the name of the old north branch of the Rhine?	From Wijk bij Duurstede, the old north branch of the Rhine is called Kromme Rijn ("Bent Rhine") past Utrecht, first Leidse Rijn ("Rhine of Leiden") and then, Oude Rijn ("Old Rhine").
8	What was one of Luther's most personal writings?	It remains in use today, along with Luther's hymns and his translation of the Bible.
...
```



* (RTE/WNLI)中的test.tsv数据样式:

```text
index	sentence1	sentence2
0	Maude and Dora had seen the trains rushing across the prairie, with long, rolling puffs of black smoke streaming back from the engine. Their roars and their wild, clear whistles could be heard from far away. Horses ran away when they came in sight.	Horses ran away when Maude and Dora came in sight.
1	Maude and Dora had seen the trains rushing across the prairie, with long, rolling puffs of black smoke streaming back from the engine. Their roars and their wild, clear whistles could be heard from far away. Horses ran away when they came in sight.	Horses ran away when the trains came in sight.
2	Maude and Dora had seen the trains rushing across the prairie, with long, rolling puffs of black smoke streaming back from the engine. Their roars and their wild, clear whistles could be heard from far away. Horses ran away when they came in sight.	Horses ran away when the puffs came in sight.
3	Maude and Dora had seen the trains rushing across the prairie, with long, rolling puffs of black smoke streaming back from the engine. Their roars and their wild, clear whistles could be heard from far away. Horses ran away when they came in sight.	Horses ran away when the roars came in sight.
4	Maude and Dora had seen the trains rushing across the prairie, with long, rolling puffs of black smoke streaming back from the engine. Their roars and their wild, clear whistles could be heard from far away. Horses ran away when they came in sight.	Horses ran away when the whistles came in sight.
5	Maude and Dora had seen the trains rushing across the prairie, with long, rolling puffs of black smoke streaming back from the engine. Their roars and their wild, clear whistles could be heard from far away. Horses ran away when they came in sight.	Horses ran away when the horses came in sight.
6	Maude and Dora had seen the trains rushing across the prairie, with long, rolling puffs of black smoke streaming back from the engine. Their roars and their wild, clear whistles could be heard from far away. Horses ran away when they saw a train coming.	Maude and Dora saw a train coming.
7	Maude and Dora had seen the trains rushing across the prairie, with long, rolling puffs of black smoke streaming back from the engine. Their roars and their wild, clear whistles could be heard from far away. Horses ran away when they saw a train coming.	The trains saw a train coming.
8	Maude and Dora had seen the trains rushing across the prairie, with long, rolling puffs of black smoke streaming back from the engine. Their roars and their wild, clear whistles could be heard from far away. Horses ran away when they saw a train coming.	The puffs saw a train coming.
...
```



> * (QNLI/RTE/WNLI)中的test.tsv数据样式说明:
> * test.tsv中的数据内容共分为3列, 第一列数据代表每条文本数据的索引;  第二列和第三列数据代表需要进行'是否蕴含'判定的句子对.



> * (QNLI/RTE/WNLI)数据集的任务类型:
> * 句子对二分类任务
> * 评估指标为: ACC




## 3 小结

* 学习了GLUE数据集合的介绍:
    * GLUE由纽约大学, 华盛顿大学, Google联合推出, 涵盖不同NLP任务类型, 截止至2020年1月其中包括11个子任务数据集, 成为衡量NLP研究发展的衡量标准.

* GLUE数据集合包含以下数据集:
    * CoLA 数据集
    * SST-2 数据集
    * MRPC 数据集
    * STS-B 数据集
    * QQP 数据集
    * MNLI 数据集
    * SNLI 数据集
    * QNLI 数据集
    * RTE 数据集
    * WNLI 数据集