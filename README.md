# Atec-sim

## 仓库描述
这个仓库包含了参加[金融大脑-金融智能NLP服务](https://dc.cloud.alipay.com/index#/topic/intro?id=3)大赛的两个模型。目前这个比赛，已经开通了[学习赛](https://dc.cloud.alipay.com/index#/topic/intro?id=8)，可随时参加打擂，还有奖金。

## [赛题描述](https://dc.cloud.alipay.com/index#/topic/data?id=8)
>__问题相似度计算__，即给定客服里用户描述的两句话，用算法来判断是否表示了相同的语义。
>
>示例：
>1. “花呗如何还款” --“花呗怎么还款”：同义问句
>2. “花呗如何还款” -- “我怎么还我的花被呢”：同义问句
>3. “花呗分期后逾期了如何还款”-- “花呗分期后逾期了哪里还款”：非同义问句
>
>对于例子a，比较简单的方法就可以判定同义；对于例子b，包含了错别字、同义词、词序变换等问题，两个句子乍一看并不类似，想正确判断比较有挑战；对于例子c，两句话很类似，仅仅有一处细微的差别 “如何”和“哪里”，就导致语义不一致。

## 模型描述
### 先附上参考文献
 - Model 1: [A Decomposable Attention Model for Natural Language Inference](https://arxiv.org/abs/1606.01933)
 - Model 2: [Improving Language Understanding with Unsupervised Learning](https://blog.openai.com/language-unsupervised/)

### 具体实现
#### 词表示
我们现在的文字系统最早起源于古时候的象形文字，形状来自于人们眼睛所看到的事物，有一些“所见即所得”的意味。而如今的计算机应用，是基于二进制表达的算法、逻辑、图像、声音等的元素构成的。文字在这些系统里的表示，也仅仅是一个数字，不包含真实的语义。在传统的基于统计学的方法中，通过统计词频，计算一些数值，如tf-idf，我们可以得到一篇文档或者长文本的表示。不同主题的文档，得到的这种表示的分布是可以进行区分的。但是其中，并不包含具体的语义。但这种方案在处理短文本时，就显得捉襟见肘了。自然语言进入到深度学习时代，词语要以什么方式来构建神经网络输入层呢？因此，文字或者词语在计算机中如何进行表示，是利用深度学习方法进行自然语言处理首先要考虑的问题。

一个最简单的输入表示，就是把词语转换成one-hot向量。但是这么做的缺点是当面对特别大的词表时，one-hot向量的维度将会变的特别大，容易造成维数灾难。并且，表示成one-hot，我们也没有办法计算相似词语之间的语义或语法近似程度。因此，Hinton大佬在1986年的论文[_Learning Distributed Representations of Concepts_](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.408.7684&rep=rep1&type=pdf)中提出用一个固定长度的 __实值向量__ 来表示任意词。

在我们的任务钟，我们也使用了词向量来表示每一个词或字，然后把句子表示为向量序列输入到模型中。[data](https://github.com/zixuanweeei/atec-sim/tree/master/Decomposable%20Attention%20Model/data)文件夹中，包含了表示词所使用的词向量。
由于该赛题任务有一定的应用背景，在进行分词的时候，我们发现有些专有词汇，如花呗、借呗等，很难用现有的分词工具精确的识别出来。
因此，我们根据赛题提供的数据，建立了[自己的字典](https://github.com/zixuanweeei/atec-sim/blob/master/Decomposable%20Attention%20Model/data/user_dict.txt)。
但是，这个字典并不能覆盖所有的情况。
我们使用已经训练好的[中文词向量](https://github.com/Embedding/Chinese-Word-Vectors)作为一个比较权威的词库。
用赛题提供的语料进行分词，与上述词库进行对比，整理出了一份[新词表](https://github.com/zixuanweeei/atec-sim/blob/master/Decomposable%20Attention%20Model/data/new_words.txt)。
新词表中列出的词没有放到用户字典里进行分词，因为它们已经是通过分词工具确定出来的词语了。
基于上述分词和新词，我们尝试自己训练了一份词向量，其中包含11829个单词。
[训练所使用的语料](https://github.com/zixuanweeei/atec-sim/blob/master/Decomposable%20Attention%20Model/data/corpus.txt)仅包含204954条句子，数据量还是太少了。
不过针对特殊背景的语料，采用一些技术精调后，或许是有用的。(P.S. 如何精调还是个问题=, =)

#### 输入
##### 线下 - TFRecords
