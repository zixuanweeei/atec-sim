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
待续