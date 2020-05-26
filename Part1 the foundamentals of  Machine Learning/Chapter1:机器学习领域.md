# Chapter1机器学习领域

<!-- TOC -->

   * [Chapter1机器学习领域](#Chapter1机器学习领域)
      * [1.examples-of-ML-applications-和一些常见解决办法](##1.examples-of-ML-applications-和一些常见解决办法)
      * [2.机器学习的种类](##2.机器学习的种类)
         * [1.有监督学习/无监督学习/半监督学习/强化学习](###1.有监督学习/无监督学习/半监督学习/强化学习)
         * [2.批量学习/在线学习](###2.批量学习/在线学习)
         * [3.基于实例/基于模型](###3.基于实例/基于模型)
      * [3.机器学习的挑战](##3.机器学习的挑战)
      * [4.测试和验证Testing-and-Validating](##4.测试和验证Testing-and-Validating)
         * [1.超参数调整和模型选择Hyperparameter Tuning and Model Selection](###1.超参数调整和模型选择Hyperparameter-Tuning-and-Model-Selection)
         * [2.Data-Mismatch](###2.Data-Mismatch)
   * [PS:No-free-lunch-theorem](##No-free-lunch-theorem)
   
   
   
 <!-- /TOC -->
   
## 1.examples-of-ML-applications-和一些常见解决办法
1. 产品线上产品图像分类： image classification， CNNs。
 2. 检测大脑肿瘤：semantic segmentation， 要求更高，对每一个pixel像素进行分类， CNNs 。
 3. 自动分类新闻文章：natural language process（NLP)，文本分类，RNNs,CNNs, Transformers。
 4. 自动标记论坛差评：NLP。
 5. summarizing long documents automatically : text summarization (a branch of NLP)。
 6. 创造聊天机器人chatbot或者个人助理 ： 包含 many NLP components, including natural language understanding (NLU) and question-answering modules。
 7. 预测公司收入：回归问题（regression task）i.e. predicting values. 线性回归+多项式回归，回归支持向量机SVM，回归随机树，ANN（MLP）。同时RNNs,CNNs, Transformers 可以被用于处理过去的数据。
 8. 响应语音命令：speech recognition，语音识别，处理audio 音频 samples，更为复杂的sequences，RNNs,CNNs, Transformers。
 9. 信用卡欺诈：异常检测，anomaly detection。
 10. 根据顾客购买习惯，segment （分段，分类） 顾客，以便提供更好的市场策略：clustering 聚类。
 11. 处理展示复杂的高维度数据：数据可视化，维度降低技术。
 12. 根据顾客过去的购买数据，为顾客推荐产品：推荐系统，ANN，input是过去的购买数据，output是下一个想要买的产品。
 13. 智能游戏机器人：强化学习和深度学习的结合。

## 2.机器学习的种类
### 1.有监督学习/无监督学习/半监督学习/强化学习
`Whether or not they are trained with human supervision (supervised, nsupervised, semisupervised, and Reinforcement Learning)。`
**有监督学习**
In supervised learning, the training set you feed to the algorithm includes the desired solutions, called **labels**. 比如：
 - 分类：训练的数据集含有labels，已经被分好类class了。
 - 回归：预测一个target数值，训练的数据集包括labels和predictors（features = 属性attribute with values）。
 
回归可以被用于分类，vice versa。**逻辑回归经常被用于分类**，因为回归输出一个value, 和输入属于哪一个class的比例probability有关。**[常见的监督学习算法：• k-Nearest Neighbors • Linear Regression • Logistic Regression • Support Vector Machines (SVMs) • Decision Trees and Random Forests • Neural networks]**
**无监督学习**
**【常见的无监督学习算法：**
 - ***聚类clustering*** ：K-Means， DBSCAN， Hierarchical Cluster Analysis (HCA)。
 - ***异常和新奇检测Anomaly detection and novelty detection***：One-class SVM， Isolation Forest。
 - ***视觉化和维度降低Visualization and dimensionality reduction***：Principal Component Analysis (PCA)，Kernel PCA，Locally Linear Embedding (LLE)， t-Distributed Stochastic Neighbor Embedding (t-SNE)。
• ***关联规则学习Association rule learning***：Apriori， Eclat。**】**

**半监督学习**
例如 ：photo-hosting services, such as Google Photos
大多数半监督算法是有监督和无监督的组合。例如，**深度信念网络 deep belief networks (DBNs)** 是基于来自**无监督的受限玻尔兹曼机restricted Boltzmann machines (RBMs)** 。 RBMs are trained sequentially in an unsupervised manner, and then the whole system is fine-tuned using supervised learning techniques.
**强化学习**
（这一部分就比较特殊了，原书说RL 是十分不同的野兽，需要单独拿出来入门）
### 2.批量学习/在线学习
`Whether or not they can learn incrementally on the fly (online versus batch learning)`
**批量学习，batch learning：**
也叫offline learning，就是说系统根据已有的数据available data 去训练，之后就不在学习了。例如，如果出现了新的数据，就要和之前的旧的数据一起重新放进网络进行学习。耗时。
可以用持续学习的算法进行改进， 比如强化学习。
**在线学习，online learning：**
就是持续不停地灌入新数据进行学习， 可以单独的用数据，也可以以**min-batch**的形式用数据。（学习感悟：所以说看书还是要从头看，尤其是概念性的，这次总算看懂这个min-batch了）。实际应用：股票价格。
online learning 也可以被用于大数据中，就是特别多特别多的dataset，机器一次处理不过来了。然后系统就以min-batch的形式，一次又一次地重复训练过程，期间不断修正参数权值之类的网络参数，直到训练完所有的数据。这个也叫`out-of-core learning`，我们可以发现，某种意义上，这个过程不是一个offline学习吗？这样online learning 就十分confusing了，**Think of it as incremental learning.**

在线学习的一个重要参数是**learning rate 学习率**，就是说系统应该以多快的速度去适应这些持续改变的数据。原文是：One important parameter of online learning systems is how fast they should adapt to changing data: this is called the learning rate. 
如果LR过高，系统会很快的适应新的数据，但是同时也会很快的忘记旧的数据。
如果LR过低，系统会产生惯性，以至于学习越来越慢，但是受新数据中noise或非代表性数据（比如异常值outliers）的影响就会比较小。

在线学习的一个挑战是是否把糟糕的数据喂进了网络。
### 3.基于实例/基于模型
`Whether they work by simply comparing new data points to known data points, or instead by detecting patterns in the training data and building a predictive model, much like scientists do (instance-based versus model-based learning)`
**基于实例：**
对训练集中的每一个具体例子进行学习，当推广到新cases的时候，通过相似度`similarity measure`与学习到的数据进行比较。 `classification`

**基于模型：**
这里就稍微有点点复杂多话了。这里有点`prediction`的意思。对训练数据构造一个模型，然后通过预测的方式推广到新数据。

【例子：训练的数据集有X和Y，然后model selection 了一个线性linear model：Y=m+tX。之后，需要确定你的两个参数m 和t，如何确实m和t呢？这里就需要一个`performance measure`，你可以定义一个`fitness function`（也叫utility function，用来评价模型用这套参数的话，会表现的有多好），或者也可以定义一个==cost function==(用来评价模型用这套参数的话，会表现的有多差）。然后你就开业把你的训练数据放进模型不断的training（在python代码里，会是 **.fit(训练数据)** 的形式）模型，知道找到使cost function 最低的那套参数值。然后就可以预测啦，新数据a（和X是一个类型的数据），然后去预测b（和Y是一个类型的数据）。】

summary一下：

 1. 要研究的数据
 2. 选择一个模型
 3. 训练模型（最好的fitness function or utility function； 最差的cost function）
 4. 应用模型，进行预测，这叫`inference`
## 3.机器学习的挑战
**算法好不好，数据好不好**
|no.|`bad data`|details|
|--|:--|:--|
| 1 | 训练数据的量不够多。 |  insufficient quantity。|
|2|训练的数据是否具有代表性。|比如，采样偏差 sampling bias。|
| 3 | 质不好的数据，poor quality。 | 比如离群值outliers, 直接丢弃discard，或者manually 修复错误。比如缺少特征值features，就要考虑还用不用这个数据，或者换成中间值之类的，或者换个不用这个值的模型。 |
|4|不相关Irrelevant 的Features。|解决方法-*feature engineering*: feature selection-选择最有用的features。feature extraction-组合选择的features，产生一些更有意义的feature数据，比如维度降低。feature creating-通过整合上一步的新数据，create新的一些features。|

|no.|`bad 算法`|details|
|--|:--|:--|
| 1 | overfitting过拟合。  人类世界里就是Overgeneralizing过度概况的感觉 | 解决方法-比如简化模型，用一些参数少的模型。比如减少训练集的attributes属性。比如`约束模型`。比如收集更多的训练数据。比如减少训练集里的噪音。|
|2|underfitting欠拟合|解决方法-比如选择更powerful的模型，用一些参数多的模型。比如证据更好的数据features，就是feature engineering。比如减少模型约束，减少正则超参数。|

 `约束模型，也叫Regularization constraint（正则化约束）`
The amount of regularization to apply during learning can be controlled by a **hyperparameter**.
这里有个具体说明： [https://www.cnblogs.com/fcfc940503/p/10966034.html](https://www.cnblogs.com/fcfc940503/p/10966034.html)
## 4.测试和验证Testing-and-Validating
把数据分为training set 和 testing set ， error rate叫做`generalization error`，也叫out of sample error。如果training error 低，但是generalization error高，说明模型对training set 过拟合了。(regularization could avoid overfitting)

### 1.超参数调整和模型选择Hyperparameter Tuning and Model Selection
**Hyperparameter Tuning** 
holdout validation （holdout 直译是坚持的意思，validation  验证）：从训练集里hold out 一部分数据集叫validation set，然后原始training set 减去validation set得到新的training set，用这个新的训练集来训练各个候选模型with各个参数，然后用在validation set上表现最好的那个模型，选择这个模型，之后在full training set （including validation set）训练，得到最后的模型，最后在原始的testing set 上评价这个最后的模型，得到最后的generalization error。
**这个new held out set 叫做validation set, 也叫做development set，dev set**
validation set 选的过大过小都不合适，为了解决这个问题，使用**cross validation**--用很多小的验证集。Each model is evaluated once per validation set after it is trained on the rest of the data. By averaging out all the evaluations of a model, you get a much more accurate measure of its performance.There is a drawback, however: the training time is multiplied by the number of validation sets.
### 2.Data-Mismatch
就是想说 训练的数据集和test的数据集直接的差距有点大，比如一个来自网站， 一个来自APP， 那么这样你用来自网站的数据训练出来的模型怎么可能在来自APP的数据集上有好的表现呢？这里就是说，一般就是找到一个量足够打的数据集，然后分成一个train和一个test,或者shuffle them 。（！！等实际应用到的时候再来看看）

# PS:No-free-lunch-theorem
NFL理论<br>
这里就是想说模型一开始就是根据实际经验给出一个assumption模型，然后在这个模型的基础上进行优化。














