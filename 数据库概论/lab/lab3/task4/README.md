任务四：基于SQL实现各种损失函数 
同样，损失函数是机器学习算法的核心，各种损失函数的定义以及如何用 SQL 实现它
们，同学们可以参照“用 SQL 实现机器学习中的基础概念-损失函数”文档中的内容。 
我们的实习任务是 “世界幸福指数数据集” 实现上面的损失函数，同时选择合适的某个
损失函数，实现一个机器学习算法，比如逻辑回归、决策树、或者聚类算法。也可以选择线
性回归算法，看看应用不同损失函数的效果如何。 

基于SQL的多元线性回归 

定义如下幸福指数数据表 happyness(Overall_rank, Country, Score, GDP_per_capita, 
Social_support, Healthy_life_expectancy , Freedom_to_make_life_choices Generosity, 
Perceptions_of_corruption)。
请以 score 作为因变量，GDP_per_capita, Social_support, Healthy_life_expectancy, Freedom_to_make_life_choices, 
Perceptions_of_corruptio 作为自变量，用 SQL完成多元线性回归算法。 


基于SQL的决策树 
Generosity, 
定义如下幸福指数数据表 happyness(Overall_rank, Country, Score, GDP_per_capita, 
Social_support, Healthy_life_expectancy , Freedom_to_make_life_choices,  
Generosity, 
Perceptions_of_corruption)，score 被划分为 high, middle, low 三个类别（这个请自己选择
划分区间）。请以score作为分类属性，用SQL完成决策树分类算法。 


基于SQL的聚类 
定义如下幸福指数数据表happyness(Overall_rank, Country, Score, GDP_per_capita,
Social_support, Healthy_life_expectancy , Freedom_to_make_life_choices , Generosity, 
Perceptions_of_corruption)。 
请选择属性 Score, GDP_per_capita, Social_support, Healthy_life_expectancy , 
Freedom_to_make_life_choices , Generosity, Perceptions_of_corruption，对它们进行归一化
处理，并用归一化之后的结果来计算样本之间的距离，用SQL完成k-means算法，指定k=3。