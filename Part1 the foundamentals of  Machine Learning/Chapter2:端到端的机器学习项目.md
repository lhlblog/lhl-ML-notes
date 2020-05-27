<a name="no"></a>
# Chapter2:end-to-end machine learning project

  - [Chapter2:end-to-end machine learning project](#no)
    - [1.working with real data](#no1)
    - [2.look at the big picture](#no2)
      - [2.1.Frame the problem](#no21)
      - [2.2.选择一个评价指标 select a performance measure](#no22)
      - [2.3.check the assumpthions](#no23)


<a name="no1"></a>
## 1.working with real data
一些常见的获得数据集的网站
- Popular open data repositories
  - [UC Irvine Machine Learning Repository](http://archive.ics.uci.edu/ml/index.php)
  - [Kaggle datasets](https://www.kaggle.com/datasets)
  - [Amazon’s AWS datasets](https://registry.opendata.aws/)
- Meta portals (they list open data repositories)
  - [Data Portals](http://dataportals.org/)
  - [OpenDataMonitor](https://opendatamonitor.eu/frontend/web/index.php?r=dashboard%2Findex)
  - [Quandl](https://www.quandl.com/)
- Other pages listing many popular open data repositories
  - [Wikipedia’s list of Machine Learning datasets](https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research)
  - [Quora.com](https://www.quora.com/Where-can-I-find-large-datasets-open-to-the-public)
  - [The datasets subreddit](https://www.reddit.com/r/datasets/)


<a name="no2"></a>
## 2.look at the big picture
<a name="no21"></a>
### 2.1.Frame the problem
确定objective,目的会帮助确定最后的算法selection， 用来评价模型的performance measure， 花费多少经历进行调整how much effort you will spend tweaking it.这里有一个概念词是`data pipeline`, 官方解释是 *A sequence of data processing components, and components typically run asynchronously*

<a name="no22"></a>
### 2.2.选择一个评价指标 select a performance measure
`a typical performance measure for regression problems is Root Mean Square Error`=**RMSE**       
<a href="https://www.codecogs.com/eqnedit.php?latex=\operatorname{RMSE}(\mathbf{X},&space;h)=\sqrt{\frac{1}{m}&space;\sum_{i=1}^{m}\left(h\left(\mathbf{x}^{(i)}\right)-y^{(i)}\right)^{2}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\operatorname{RMSE}(\mathbf{X},&space;h)=\sqrt{\frac{1}{m}&space;\sum_{i=1}^{m}\left(h\left(\mathbf{x}^{(i)}\right)-y^{(i)}\right)^{2}}" title="\operatorname{RMSE}(\mathbf{X}, h)=\sqrt{\frac{1}{m} \sum_{i=1}^{m}\left(h\left(\mathbf{x}^{(i)}\right)-y^{(i)}\right)^{2}}" /></a>
- m 是说数据集里面例子的个数
- <a href="https://www.codecogs.com/eqnedit.php?latex=x^{(i)}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?x^{(i)}" title="x^{(i)}" /></a>和<a href="https://www.codecogs.com/eqnedit.php?latex=y^{(i)}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?y^{(i)}" title="y^{(i)}" /></a> 其中i是只第i个例子，是序号；x是一个vector向量，里面包含所有的feature values， 经纬度也是OK的。y是label，是desired的output,一般就是一个值。
- X 这个在前面的大写加粗的X是所有x的集合
- h 是prediction function 也叫做`hypothesis`, <a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y}^{(i)}=h(x^{(i)})" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\hat{y}^{(i)}=h(x^{(i)})" title="\hat{y}^{(i)}=h(x^{(i)})" /></a>,y-hat 是系统预测出来的值。
- RMSE(X,h) 是成本函数， `cost function`,h 就是你的预测系统。<br>
`mean absolute error` = `average absolute deviation` = **MAE**, deviation 是偏差的意思<br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\operatorname{MAE}(\mathbf{X},&space;h)=\frac{1}{m}&space;\sum_{i=1}^{m}\left|h\left(\mathbf{x}^{(i)}\right)-y^{(i)}\right|" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\operatorname{MAE}(\mathbf{X},&space;h)=\frac{1}{m}&space;\sum_{i=1}^{m}\left|h\left(\mathbf{x}^{(i)}\right)-y^{(i)}\right|" title="\operatorname{MAE}(\mathbf{X}, h)=\frac{1}{m} \sum_{i=1}^{m}\left|h\left(\mathbf{x}^{(i)}\right)-y^{(i)}\right|" /></a> <br>
**RMSE 和 MAE 都是计算两个向量直接的距离，就是预测值向量和目标值向量直接的距离**<br>
> - Various distance measures or norms are possible:       
>   - RMSE corresponds to 对应的是 Euclidean Norm, l2。       
>   - MAE 对应的是l1 norm, 也叫做Manhattan norm。               
> - **note:**       
>The higher the norm index, the more it focuses on large values and neglects small ones. This is why the RMSE is more sensitive to outliers than the MAE. But when outliers are exponentially rare (like in a bell-shaped curve), the RMSE performs very well and is generally preferred.
<a name="no23"></a>
### 2.3.check the assumpthions
你假设的到底是分类问题还是预测问题

<a name="no3"></a>
## 3.get the data
通过函数得到数据 见jupyter
























