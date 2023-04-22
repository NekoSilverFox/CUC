<div align="center">
  <img width="250px" src="https://github.com/NekoSilverFox/NekoSilverfox/blob/master/icons/logo_building_spbstu.png?raw=true" align="center" alt="ogo_building_spbstu" />
  </br>
  <b><font size=3>Санкт-Петербургский государственный политехнический университет</font></b>
  </br>
  <b><font size=2>Институт компьютерных наук и технологий</font></b>
  </br>
  <b><font size=2>Высшая школа программной инженерии</font></b>
</div>


</br>
</br>
</br>
</br>

<div align="center">
<p align="center"><b><font size=12>编码单元分类器</font></b></br></p>
</div>
</br>
</br>
</br>
</br>
</br>
</br>


| Выполнил студенты гр. 3530904/90102 | Мэн Ц.            |
| :---------------------------------- | ----------------- |
| Руководитель                        | Черноруцкий И. Г. |

</br>
</br>

<div align="center">
  <p align="center">Санкт-Петербург</p>
  <p align="center">2023</p>
</div>

<div align=left>

> 作者：孟嘉宁，圣彼得堡彼得大帝理工大学，计算机科学与技术学院
>
> Мэн Цзянин, Санкт-Петербургский политехнический университет Петра Великого, Институт компьютерных наук и технологий

<p align="center"><b><font size=6>摘要</font></b></br></p>

> 关键词：机器学习，分类算法，分类器，分类预测，监督学习，算法，编码单元

本文提出了一种新的算法——编码单元分类算法 (CUCL - Coding unit classification algorithm), 用于解决机器学习中的分类预测问题。该算法的灵感来自于**高效率视讯编码[^1]**（High Efficiency Video Coding，简称 HEVC，又称为H.265）中的[编码树单元](https://zh.wikipedia.org/wiki/編碼樹單元)(Coding Tree Unit, CTU)。

编码树单元是HEVC的基本编码单元，它取代了过往中使用的 16×16 像素[宏区块](https://zh.wikipedia.org/wiki/宏區塊)。编码树单元通过使用可变大小的区块结构，将图片细分为可变大小的像素块，提升通常会提高时编码的效率。

借助于编码树中编码单元的思想，我们开发了编码单元分类器（Coding unit classifier， CUC）。编码单元分类器中包含的编码单元是通过“预分割”、“细化分割”和“感染”三个步骤构建的，这些编码单元也组成了预估器。对于输入到编码单元分类器中的新样本点，将通过其应该所属的编码单元进行目标值预测。此外，我们还提出了一种方法，可以通过配置CUC的超参数来消除数据集中的噪声，或减少异常值对分类器性能的影响。

在最终的测试中，CUC 分类器对于不同的样本点数量和样本点分布的数据集都能够十分迅速的构建预估器并且对测试集达到 >0.95 的准确率。

在本项工作中，使用了以下库和工具 drawio, git, numpy、pandas、Matplotlib 和 scikit-learn，开发语言为Python 3.9，并使用PyCharm作为开发和测试环境。



[toc]

# 简介

在机器学习中，分类器是一种用于分类任务的算法或模型。分类器使用给定的数据集来构建一个分类模型，该模型可以将新数据分为不同的类别。分类器通常使用已知类别的训练数据来构建模型，并使用该模型对未知类别的测试数据进行分类。常见的机器学习分类器包括决策树、朴素贝叶斯分类器、支持向量机、K最近邻分类器等。这些分类器在不同的应用场景中具有不同的优缺点，选择适合的分类器需要考虑数据集的特征和目标。

机器学习分类器广泛应用于各种领域，例如医疗诊断、金融风险评估、图像识别、自然语言处理等。在医疗领域，分类器可以通过对患者的生理数据和症状进行分类来帮助医生诊断疾病。在金融领域，分类器可以通过对客户的信用记录和财务数据进行分类来评估客户的信用风险。在图像识别领域，分类器可以通过对图像进行分类来识别图像中的物体或场景。在自然语言处理领域，分类器可以通过对文本进行分类来实现情感分析、垃圾邮件过滤、文本分类等任务。因此，机器学习分类器在实际应用中具有广泛的应用前景。

在实际应用中，数据集的样本点分布通常比较复杂，这使得传统的分类器往往无法取得很好的分类效果。所以在本项工作中，我们将开发一个能够适用于大部分场景，特别是数据集中具有复杂样本点分布的机器学习分类器（编码单元分类器），以解决实际应用中的分类问题。编码单元分类器是一种新型的机器学习分类器，其基于自HEVC编码树单元的思想，并结合了分类器的特性。

目前，所提出的编码单元分类器已经被开发出来，并在多个具有不同样本点分布的数据集上进行了实验，实验的结果表明编码单元分类器具有更强的适应性和鲁棒性，可以在复杂的数据集中取得很好的分类效果。



> 在数据集上的动态演示：
>
> ![DoubleHelix](doc/pic/DoubleHelix.gif)
>
> ![fourclass](doc/pic/fourclass.gif)



# 文献回顾

## 机器学习的发展

机器学习是人工智能的一个分支，其发展历程从以“推理”为重点，到以“知识”为重点，再到以“学习”为重点，并已成为一门多领域交叉学科，涵盖多门学科，如概率论、统计学、逼近论、凸分析、计算复杂性理论等。机器学习理论主要探讨让计算机自动“学习”的算法，即从数据中自动分析获得规律，并利用规律对未知数据进行预测的算法。[^2]机器学习已广泛应用于诸多领域，如数据挖掘、计算机视觉、自然语言处理、生物特征识别等。

根据不同的学习方式，机器学习可分为监督学习、无监督学习、半监督学习和增强学习。监督学习的训练集包括由人标注的输入和目标输出，常见的算法有回归分析和统计分类；无监督学习的训练集没有人为标注的结果，常见的算法有生成对抗网络（GAN）和聚类；半监督学习介于监督学习和无监督学习之间；增强学习机器为了达成目标，随着环境的变动，而逐步调整其行为，并评估每一个行动之后所到的回馈是正向的或负向的。

许多算法被提出用于解决机器学习中的分类预测问题，并取得了较好的结果（T.Cover, P.Hart, 1967[^1-3]; Harry Zhang, 2004[^1-4]; Kevin P. Murphy, 2006[^1-5]; Leo Breiman, 1984[^2-5]; Quinlan, J. R. 1986[^2-6]; Quinlan, J. R. 1993[^2-7]; Freund, Y., Mason, L. 1999[^2-9]; Tin Kam Ho, 1995[^2-11], Leo Breiman, 2001[^2-12]; David Cox, 1958[^2-22]; V.N. Vapnik，A.Y. Chervonenkis，C. Cortes等人, (1964)[^2-14]）。下一章将讨论其中的一些算法。

## 分类算法

在介绍本文中提出的算法之前，我们将介绍以前为解决计机器学习分类预测方面所做的一些工作。

- 1967年，T.M.COVER和P.E.HART提出了kNN算法(k-NearestNeighbor, kNN)[^1-2]。其中k指的是某个样本的k个最近的邻居，也就是说每个样本都可以用它最接近的k个邻居来代表或预测。 kNN算法的核心思想是如果一个样本在特征空间中的k个最相邻的样本中的大多数属于某一个类别，则该样本也属于这个类别，并具有这个类别上样本的特性。该方法在确定分类决策上只依据最邻近的一个或者几个样本的类别来决定待分样本所属的类别。

    > https://bbs.huaweicloud.com/blogs/251434
    
    

- 在（Harry Zhang, 2004[^1-3]; Kevin P. Murphy, 2006[^1-4]）中提出了朴素贝叶斯分类器。该分类器模型会给问题实例分配用特征值表示的类标签，类标签取自有限集合。它不是训练这种分类器的单一算法，而是一系列基于相同原理的算法：所有朴素贝叶斯分类器都假定样本每个特征与其他特征都不相关。朴素贝叶斯自1950年代已广泛研究，在1960年代初就以另外一个名称引入到文本信息检索界中[^2-1]，并至今仍然是文本分类的一种热门方法，文本分类是以词频为特征判断文件所属类别或其他（如垃圾邮件、合法性、体育或政治等等）的问题。通过适当的预处理，它可以与这个领域更先进的方法（包括支持向量机）相竞争。它在自动医疗诊断中也有应用。

    
  
- 决策树 - 决策树是一类树模型的统称，其最早可以追溯到1948年左右，当时克劳德·香农介绍了信息论[^2-2]，这是决策树学习的理论基础之一。随后，1963年，Morgan和Sonquist开发出第一个回归树[^2-3]，他们提出了一种分析调查数据的方法，它不对交互影响施加任何限制，侧重于减少预测误差，顺序操作，并且不依赖于分类中的线性程度或解释变量的排列顺序，当时他们起名为交互检测（AID）模型。由于AID没有真正考虑到数据固有的抽样变异性，Messenger和Mandell在1972提出了THAID树以弥补这一缺陷，1980年，Gordon V. Kass开发出CHAID算法[^2-4]，这是AID和THAID程序的正式扩展。

    随后出现的是CART树——分类和回归树（简称CART）——是Leo Breiman引入的术语[^2-5]，指用来解决分类或回归预测建模问题的决策树算法。CART模型包括选择输入变量和那些变量上的分割点，直到创建出适当的树。使用贪婪算法（greedy algorithm）选择使用哪个输入变量和分割点，以使成本函数（cost function）最小化。

    1986年，Quinlan开发出ID3算法[^2-6]，并于随后几年提出了C4.5算法[^2-7]。之后为了提高它的运算效率，Loh和Shih于1997年开发出QUEST[^2-8]。

    1999年，Yoav Freund和Llew Mason提出AD-Tree[^2-9]，在此之前，决策树已经在正确率上取得了很不错的成就，但是当时的决策树在结构上依旧太大，而且不容易理解，所以AD-Tree在节点上做了改变。该算法不仅对数进行投票，也对决策节点进行投票。并且每个决策节点都是有语义信息的，支持图形化，使人更容易理解。

    目前流行的决策树算法包括ID3、CHAID、CART、QUEST和C4.5。

    决策树可以看成为一个if-then规则的集合，即由决策树的根节点到叶节点的每一条路径构建一条规则，路径上内部节点的特征对应着规则的条件，而叶节点的类对应于规则的结论。因此决策树就可以看作由条件if（内部节点）和满足条件下对应的规则then（边）组成。

    决策树的工作方式是以一种贪婪（greedy）的方式迭代式地将数据分成不同的子集。其中回归树（regression tree）的目的是最小化所有子集中的MSE（均方误差）或MAE（平均绝对误差）；而分类树（classification tree）则是对数据进行分割，以使得所得到的子集的熵或基尼不纯度（Gini impurity）最小。[^2-10]
    
    
    
- Tin Kam Ho 于 1995 年[^2-11]提出了随机森林。随后由Leo Breiman于2001年[^2-12]在一篇论文中提出了一种结合随机节点优化和bagging，利用类CART过程构建不相关树的森林的方法。此外，本文还结合了一些已知的、新颖的、构成了现代随机森林实践的基础成分，特别是

    1. 使用out-of-bag误差来代替泛化误差
    2. 通过排列度量变量的重要性

    随机森林也是集成学习方法的一种。在机器学习中，随机森林是一个包含多个决策树的分类器，并且其输出的类别是由个别树输出的类别的众数而定。RF 模型已被证明是小样本和高维数据的可靠预测器（Biau 和 Scornet，2016[^2-13]）。



- 逻辑回归分类器 (Logistic regression, LR) 是另一种基本方法,虽然名字中带有“回归”一次，但它是一个分类器而不是回归方法，最初由David Cox在1958年提出[^2-22]，它建立了一个Logistic模型（也被称为Logit模型）。它最显著的优点是，它既可以用于分类，也可以用于类别概率估计，因为它与对数数据分布相联系。它采取线性组合的特征，并对其应用非线性的sigmoid函数。在逻辑回归的基本版本中，输出变量是二进制的，然而，它可以扩展到多个类别（那么它被称为多进制逻辑回归(multinomial logistic regression)）。二元逻辑模型将样本分为两类，而多指标逻辑模型则将其扩展为任意数量的类，而不对其进行排序。[^2-23]

    

- 支持向量机（Support Vector Machine, SVM）被V.N. Vapnik，A.Y. Chervonenkis，C. Cortes 等人于 1964 年[^2-14]提出，1964年，Vapnik和Alexey Y. Chervonenkis对广义肖像算法进行了进一步讨论并建立了硬边距的线性SVM[^2-15]。此后在二十世纪70-80年代，随着模式识别中最大边距决策边界的理论研究[^2-16]、基于松弛变量（slack variable）的规划问题求解技术的出现[^2-17]，和VC维（Vapnik-Chervonenkis dimension, VC dimension）的提出[^2-18]，SVM被逐步理论化并成为统计学习理论的一部分。1992年，Bernhard E. Boser、Isabelle M. Guyon和Vapnik通过核方法得到了非线性SVM[^2-19]。1995年，Corinna Cortes和Vapnik提出了软边距的非线性SVM并将其应用于手写字符识别问题 [^2-20]。

    SVM 是一类按监督学习方式对数据进行二元分类的广义线性分类器（generalized linear classifier），其决策边界是对学习样本求解的最大边距超平面（maximum-margin hyperplane）[^2-21]。SVM 使用铰链损失函数（hinge loss）计算经验风险（empirical risk）并在求解系统中加入了正则化项以优化结构风险（structural risk），是一个具有稀疏性和稳健性的分类器。SVM可以通过核方法（kernel method）进行非线性分类，是常见的核学习（kernel learning）方法之一。



# 所提出的方法

## 模型中使用的变量和参数

| **变量/参数** | **定义**                                                  |
| ------------- | --------------------------------------------------------- |
| $x$           | 数据集中的当前样本点                                      |
| $x^{\prime}$  | 数据预处理后的当前样本点                                  |
| $y$           | 样本点或编码单元的目标值（target）                        |
| $x_{min}$     | 数据集某维度（列）中的最小值                              |
| $x_{max}$     | 数据集某维度（列）中的最大值                              |
| $I_{min}$     | 在数据预处理阶段，数据被映射到的最小值                    |
| $I_{max}$     | 在数据预处理阶段，数据被映射到的最大值                    |
| $N_i$         | 数据集中的第 i 个样本点（数据集的第 i 行）                |
| $D_i$         | 数据集中的第 i 个维度（数据集的第 i 列）                  |
| $\Delta L$    | 编码单元的边长                                            |
| $P_{start}$   | 编码单元的起始点                                          |
| $P_{end}$     | 编码单元的结束点                                          |
| $p$           | 编码单元内，占比最高的样本点的比例                        |
| $t$           | 超参数：预分割阶段确定是否应该暂停分割的阈值（threshold） |
| $C_{re}$      | 超参数：编码单元细化分割的次数                            |
| $ρ$           | 编码单元的感染力度（）                                    |
| $V$           | 编码单元的容积                                            |





## 数据预处理

在数据集中，我们将一行数据称为一个样本点或者一个粒子，而数据集中除目标值的某一列被称为某一维度 $D_{i}$。

得到特征抽取后的数据集后，我们需要对其中的数据进行预处理。在本文提出的分类器 CUC 中，引入数据预处理是为了方便后续数据可视化，并消除当数据集维度过大时可能导致的计算数据溢出或值过小时的精度损失。通常情况下，我们可以使用归一化和标准化两种方法。这两种方法也是大多数机器学习算法中常用的数据预处理方法。

### 归一化

在归一化中,我们将特征抽取之后的数据进行变换，把数据映射到固定区间$[I_{min}, I_{max}]$之内。归一化的在数学上表述如下：
$$
x^{\prime}={\frac{x-x_{min}}{x_{max}-x_{min}}} \cdot {(I_{max}-I_{min})}+I_{min}
$$
其中，

- $x$ - 为数据集中的当前样本点
- $x_{min}$ - 为该特征值对应维度中的最小值
- $x_{max}$ - 为该特征值对应维度中的最大值
- $I_{min}$ - 指定区间值的最小值
- $I_{max}$ - 指定区间值的最大值

进行归一化的优点是未改变数据分布的形状，但缺点是如果出现缺失值或者异常值（最大值和最小值是一个非常大的数），归一化会非常收到异常点的影响。所以这种方法的鲁棒性较差。

### 标准化

> https://www.zhihu.com/question/46500878

标准化（Standardization），也被称为Z值归一化（Z-Score Normalization），其作用是将每一维特征都调整为均值为0，方差为1。如果数据集中的数据比较嘈杂，我们建议使用标准化来进行特征预处理，标准化后数据集中每个维度的特征都具有相似的尺度和平均值，使得在数据处理中更加稳定和统一。标准化的数学表示如下：

首先计算均值与方差
$$
\begin{aligned}

\mu & =\frac{1}{N} \sum_{n=1}^N x \\

\sigma^2 & =\frac{1}{N} \sum_{n=1}^N\left(x-\mu\right)^2

\end{aligned}
$$
然后将特征 $x$ 减去均值，并除以标准差，得到新的特征值 $x^{\prime}$
$$
x^{\prime}=\frac{x-\mu}{\sigma}
$$
标准差 $\sigma$ 不能为零，否则说明这一维的特征没有任何区分性。

其中，

- $x$ - 为数据集中的当前样本点
- $\mu$ - 该特征值对应维度的平均值
- $\sigma$ - 该特征值对应维度标准差

## 编码单元

在本章节中我们将介绍编码单元的概念和其性质。

编码单元（Coding unit, CU）为编码单元分类器 (Coding unit classifier, CUC) 的最基本单位，其在空间中的形态由数据集中的维度数量 $D$ 决定，CU 可以视为一个在 $D$ 维空间下的超立方体（hypercube）。在二维的空间下，编码单元表现为正方形；在三维空间下，编码单元表现为立方体；在四维及更高维度的空间下，编码单元表现为超立方体。下图展示了在不同维度空间下空间下，编码单元的形态。

<img src="doc/pic/各维度下的超立方体.jpg" alt="各维度下的超立方体" style="zoom: 25%;" />

编码单元具有两个重要性质：起始点（$P_{start}$）和边长 （$\Delta L$）。在任何维度 $D_{i}$ 下，单个编码单元的边长都相等，并且边在该维度下与坐标轴平行。这使得我们仅通过编码单元的 $P_{start}$ 和 $\Delta L$ 便可以确定它的体积和在空间中所有顶点的位置。

对于在 $D$ 维空间下的一个编码单元 $CU$，它的容积 $V$ 可以通过下面的公式计算：
$$
V = {\Delta L}^{D}
$$
对于编码单元在维度 $D_i$ 下的结束点 $P_{i,\ end}$，我们可以通过一下公式确定：
$$
P_{i,\ end} = P_{i,\ start} + \Delta L
$$
在 $D$ 维数据集下，对于某个编码单元 $CU$ 和数据集中的一个点 $x_i$，如果在所有维度 $D_j$ 下，满足 $P_{j,\ start}<x_{i,\ j}<P_{j,\ end}$，则可以判断该点属于此编码单元，否则不属于。
$$
\left\{\begin{array}{}
x \in  CU,\ if \ P_{start}<x<P_{end}  \\
x \not\in  CU,\ otherwise
\end{array}\right.
$$
如果一个编码单元中不包含任何样本点，我们将其称之为空白编码单元（$CU_{EMPTY}$）。

## 构建编码单元

在本章节，我们将介绍构建编码单元的方法。构建最终的编码单元主要涉及两个阶段：“分割”和“感染”，其中“分割”阶段又分为“预分割”和“细化分割”。

![构建编码单元的阶段](doc/pic/构建编码单元的阶段.png)

### 分割

在这一小节中，我们将详细介绍编码单元的分割方法。下图展示了一个二维编码单元被分割，并不断的再对其右上角编码单元分割的过程。

<img src="doc/pic/编码单元分割示例.png" alt="编码单元分割示例" style="zoom:50%;" />

而在下图中，展示了一个被均匀分割了 3 次的编码单元。均匀分割 - 即对后续产生的小编码单元也再进行分割。

<img src="doc/pic/均匀分割示例.png" alt="均匀分割示例" style="zoom:50%;" />

可以看出，在分割的过程中，单个编码单元的容积 $V$ 和其中包含的粒子数量逐渐减少（如果包含了粒子），其效果类似于一个不断递归的正方形。

在分割过程中：

- 每进行一次分割，编码单元的边长将变为原来的 $1/2$。所以对于一个由 $n$ 次分割得到的编码单元，其边长 ${\Delta L}^{\prime}$ 可由以下公式确定
    $$
    {\Delta L}^{\prime} = \Delta L /{2^{n}}
    $$
    
- 对于 $D$ 维的编码单元，进行一次分割后，它将被更小的 $2^D$ 个编码单元所替代。所以对于一个被均匀分割（均匀分割即对后续产生的小编码单元也再进行分割）了 $n$ 次的编码单元，其产生的小编码单元数量可由以下公式确定：
    $$
    number\ CU\ after\ n\ splits = 2^{nD}
    $$

- 对于一个由 $n$ 次分割得到的编码单元，在某一维度上起始点和结束点 $(P_{start}, P_{end})$ 的数学表述如下：
    $$
    (P_{start}, P_{end}) = (I_{min}+\frac{I_{max}-I_{min}}{2^{n}} \cdot k,\ I_{min}+\frac{I_{max}-I_{min}}{2^{n}} \cdot {(k+1)}), \qquad
     k=0\ or\ 1\ \ or\ ...\ or\ 2_{n}-1
    $$

为了确定在 $D$ 维空间下编码单元 $CU$ 在经过一次分割后，产生的所有新编码单元顶点在空间中的位置，我们可以构建一个“二进制数值表”（Binary Value Table 或 Bit table），其大小为 $[{2^D},\ D]$，其中每一行对应一个 $D$ 位二进制数，其取值范围为 ${[0, 2^{D-1}]}$。然后将其每个维度 $D_{i}$ 上的新 $P_{i,\ start}$ 和 $P_{i,\ end}$ 映射（map）到对应的二进制位上。下图和算法 1 中展示了其构建过程和方法。

![二进制数值表](doc/pic/二进制数值表.png)

```
INPUT: D, 当前 CU 的起始点和 DL

{\Delta L}^{\prime} = \Delta L /{2}

计算每个维度（D_i）下的新起始点和结束点：
	P_{i, start} = P_{i, start}
	P_{i, end} = P_{i, start} + {\Delta L}^{\prime}

构建大小为 [{2^D},\ D] 的二进制表
将每个维度（D_i）下的新起始点和结束点映射到二进制表
```



#### 预分割

> /Users/fox/雪狸的文件/【资料】学习中/【优质-完整】机器学习讲义/机器学习(算法篇1).pdf
>
> 避免分类器过多的强调了训练集的准 确率甚至于对一些错误/异常的数据也进行了学习，而正确的数据却无法覆盖整个特征空间。为此，这样得到的分类器 在对新数据进行预测时将会出现错误。这种现象称之为过拟合，同时也是维灾难的直接体现。

本阶段将介绍编码单元的预分割原则和方法。

在开始时，数据集中的所有样本点都包含在一个大的编码单元中。在“预分割”阶段，需要将初始的编码单元分割为更小的单元，直到满足其中一个条件：

- 编码单元中不存在任何样本点

- 在当前编码单元中，占比最高的某种样本点的比例 $p$ 大于等于指定的阈值 $t$（threshold）。

    这里，阈值 $t$ 是作为编码单元分类器 CUC 的超参数出现的，在初始化 CUC 估计器时，用户需要指定具体值，并且 $0 \leq t \leq 1$。通过判断 $t \leq p$ 的目的是为了避免过拟合（overfitting），减少数据集中的异常值对 CUC 分类性能的影响。
    
    在编码单元内，占比最高的某种样本点的比例 $p$，可以用以下公式计算：
    $$
    p = \frac{在CU中占比最多的某种样本点的数量}{此CU中所有样本点的数量}
    $$

在预分割结束后，

- 不包含任何样本点的编码单元我们将其目标值 $y_{CU}$ 设置为空白编码单元（$CU_{EMPTY}$），

- 将包含有样本点的编码单元的目标值 $y_{CU}$ 设置为与该单元中最大占比样本点的目标值 $y_x$ 相同，并且将不等于该编码单元目标值的样本点标记为异常点，不再对其进行后续处理。
    $$
    y_{CU_{i}} =
    \left\{\begin{array}{}
    CU_{EMPTY},\ if \ no\ point\ in\ current\ CU\\
    y_x,\ points\ with\ the\ highest\ percentage
    \end{array}\right.
    $$
    

预分割的主要步骤如算法2所示：

```pascal
INPUT: 初始编码单元的 $P_{start}$, $\Delta L$

FOR 所有没有最终目标值$y_{CU_{i}}$ 的 $CU_{i}$
 	IF(当前编码单元中不存在任何样本点):
 		$y_{CU_{i}} = CU_{EMPTY}$
 		BREAK
 		
  IF($t \leq p$:)
  	$y_{CU_{i}} = y_x$
  	处理异常值
  	BREAK
  	
  继续分割 CU_{i} 为更小的 CU，记录产生的新 CU 的 $P_{start}$ 和 $\Delta L$
  	
```



#### 细化分割（refinement split）

细化分割是在预分割基础上所进行的，其目的是为了提高后期预测的准确率，避免在后期“感染”过大的编码单元时造成欠拟合（underfitting）。细化分割的次数 $C_{re}$ 是作为编码单元分类器 CUC 的超参数出现的，具体次数应该由用户指定。细化分割的具体步骤如下：

1. 对已经预分割结束后获得的所有单元进行 $C_{re}$ 次均匀分割

2. 由于经过 $C_{re}$ 次均匀分割后产生了更小的编码单元，其中可能会产生新的不包含任何样本点的空白编码单元，这些新产生的空白编码单元的目标值被重新标记为 $CU_{EMPTY}$。

    下图中展示了一个当细化分割次数 $C_{re}=1$ 和 $C_{re}=2$ 时对某个编码单元的效果。可以看到，当经过 2 次均匀分割后，产生了新的不包含任何样本点的空白编码单元，因此我们需要对其进行重新标注。

    <img src="doc/pic/细化分割示例.png" alt="细化分割示例" style="zoom:50%;" />

细化分割的主要步骤如算法3所示：

```
INPUT: 预分割结束后产生的所有 CU 的起始点和边长, C_{re}
OUTPUT: 最终的编码单元

对预分割结束后产生的所有 CU 再进行 $C_{re}$ 次均匀分割

FOR every new CU:
	IF (这个 CU 中没有样本点):
		$y_{CU_{i}} = CU_{EMPTY}$
```



### 示例

假设我们拥有一个包含红色和蓝色两种目标值的二维数据集，他们在数据预处理后的分布如下图所示：

![分割示例-粒子分布](doc/pic/分割示例-粒子分布.png)

CUC 中的超参数：

| 超参数   | 值   |
| -------- | ---- |
| $t$      | 0.99 |
| $C_{re}$ | 1    |

**预分割：**

1. 首先我们要构建一个和数据映射区间 $[I_{min}, I_{max}]$ 相同大小的编码单元，这个编码单元包含了数据集中的所有样本点。如下图所示：

    ![分割示例-编码单元1](doc/pic/分割示例-编码单元1.png)

2. 经过第一次分割后，根据编码单元预分割原则：

    - 第 1、3、4 号编码单元中所有样本点所对应的目标值唯一（$p_{1, 3, 4} \geq t$），因此我们将 $y_{CU1, 3, 4} = y_{x\ in \ CU_{1, 3, 4}}$
    - 而 2 号编码单元中占比最多的蓝色样本点的 $p_{2} < t$，所以我们需要其进行进一步的分割

    第一次分割后的效果如下图所示：

    ![分割示例-编码单元2](doc/pic/分割示例-编码单元2.png)

3. 经过第二次分割后，我们可以看到，原本编号为 2 的编码单元被分割成了更小的4、5、6、7号编码单元。第二次分割后的效果如下图所示：

    ![分割示例-编码单元3](doc/pic/分割示例-编码单元3.png)

    并且此时每个编码单元都满足 $p_{i} \geq t$ 或者不包含任何样本点，预分割阶段结束。



**细化分割阶段：**

5. 细化分割的次数指定为了 $C_{re}=1$，所以经过一次均匀分割后后，效果如下图所示：

    ![细化分割-示例](doc/pic/细化分割-示例.png)

6. 最终，我们将新产生的空白编码单元的目标值重新标记为 $CU_{EMPTY}$。

    ![重新标记空白单元](doc/pic/重新标记空白单元.png)

至此，分割阶段结束。开始感染阶段。



### 感染

> 流行病学 https://doi.org/10.1016/B978-0-323-90385-1.00004-2
>
> 最终，在“感染”阶段，那些空白编码单元将会通过计算被编码为目标值的一种，以用于分类器后续的预测。

在本阶段，将介绍感染的原理和方法。

该阶段的原理是根据病毒传播学和流行病学[^gr-1]中的 SIR 模型[^gr-2]（S，I，R 分别代表易感者、感染者和康复者的数量）。当族群中存在某种可传播性病毒时，该病毒在某族群中传染的力度 $ρ$ (infectivity) 由该族群中感染者的数量 $N_{I}^{SIR}$ 和区域的面积 $A$ 共同决定。
$$
ρ = \frac{N_{I}^{SIR}}{A}
$$
编码单元分类器的感染阶段就是受 SIR 模型的启发，我们可以将每个具有样本点的编码单元视为一个族群，并且与编码单元目标值相同的样本点的数量即为感染者的数量 $N_{I}^{SIR}$ ，而编码单元的容积 $V$ 即为 SIR 模型中族群所在的区域面积 $A$。在“分割”阶段后，每个编码单元的感染力度可以用以下公式表示：
$$
ρ = \frac{N_{I}^{SIR}}{\Delta L /{2^{n}}}=\frac{N_{I}^{SIR}}{V}
$$
由公式可知，如果某个编码单元的传染力度 $ρ$ 为 0，则说明其内部没有任何样本点存在（即空白编码单元），如上面图片中编号为 1、6、8、13 等单元。感染的目的就是将这些空白编码单元标记为数据集中目标值的某一种。

在计算出所有编码单元的传染力度 $ρ$ 后，感染阶段开始。

1. 拥有最大感染力度且没有感染过其他编码单元的编码单元（感染者， S），应当去感染其临近编码单元（被感染者，I）

    - 如果相邻的编码单元为空白编码单元，则将其感染为相同目标值，并执行步骤2
    - 如果相邻的编码单元具有相同目标值，则直接执行步骤2。
    - 如果相邻的编码单元具有不同目标值，则它们无法被感染，不会改变其目标值

2. 感染相邻编码单元之后，它们可以被视为一个融合在一起的，经过扩张的编码单元。经过扩张的编码单元，容积增大，感染力度下降。数学上的表述如下：
    $$
    ρ^{\prime}=\frac{\sum_{i=1}^n {N_{I}^{SIR}}}{\sum_{i=1}^n V}
    $$
    其中，$n$ - 为本次感染中感染者和被感染者的数量

3. 重复第一步，直到所有的编码单元都被感染。



感染阶段的主要步骤如算法4所示：

```
INPUT: 分割阶段后的所有 CU的起始点

WHILE (仍有未被感染的CU):
	感染者 = 拥有最大感染力度且没有感染过其他编码单元的编码单元
	感染者感染其临近的编码单元：
		IF (相邻的编码单元为空白编码单元):
			则将其感染为相同目标值，并标记为被感染者
    ELSE IF (相邻的编码单元具有相同目标值):
    	标记为被感染者
    ELSE IF (相邻的编码单元具有不同目标值):
    	不进行感染
    
	更新 感染者 和被感染者的 ρ
```



#### 示例

我们在分割阶段得到的结果的基础上进行，感染阶段的示例如下。

1. 通过计算可知，在 $ρ\neq0$ 的编码单元中，26 号的感染力度最大。所以应当去感染临近的 25 号编码单元

    ![感染示例 1](doc/pic/感染示例 1.png)

    

2. 之后，第二大 $ρ$ 的 4 号单元应当去感染 13 和 15 号。

    ![感染示例2](doc/pic/感染示例2.png)

3. 以此类推，得到最终的编码单元分类器

    ![感染示例3](doc/pic/感染示例3.png)

## 预测

对于一个输入CUC分类器中的新样本点 $x$，我们可以根据它所属编码单元的种类进行预测。比如在下图中展示了两个新粒子 $x_{new1}$ 和 $x_{new2}$，他们将会分别被预测为红色和蓝色。

![预测](doc/pic/预测.png)

预测阶段的主要步骤如算法5所示：

```
INPUT: 新的样本点

FOR CU_i 所有 CU 中:
	IF (新的样本点 x \in CU_i):
		RETURN y_{CU_{i}}
```



# 实验

> 准确率(Accuracy), 精确率(Precision), 召回率(Recall)和F1-Measure
>
> https://blog.csdn.net/JohinieLi/article/details/80672593

## 实验条件

为了评估所提出的算法的有效性和超参数对编码单元分类器性能的影响，我们在不同的数据规模和数据分布的数据集上进行了测试。这些数据集包括来自fourclass数据集[^fourclass]以及四组具有不同分布的二维数据集，分别为对角正态分布、Checkerboard、DoubleHelix 和锯齿(Sawtooth)。我们的衡量标准包括训练模型的时间以及对测试集的预测结果的准确度（Accuracy）。

为了保证结果的公平性，在对测试阶段使用的数据集进行特征工程时，均采用以下参数：

| 参数       | 说明                                                         |
| ---------- | ------------------------------------------------------------ |
| 特征变换   | 归一化，映射到区间 [0, 10]                                   |
| 数据集划分 | 使用 sklearn 模型选择模块下的 train_test_split。采用 75% 训练集，25% 测试集，随机数种子 90102 |

为了测试 CUC 分类器两个超参数 $C_{re}$ 和 $t$ 对性能的影响，他们的取值如下：

| 超参数   | 取值                    |
| -------- | ----------------------- |
| $C_{re}$ | 0 到 4，间隔为 1        |
| $t$      | 0.7 到 1.0，间隔为 0.01 |

试验所使用的计算机配备了 macOS13 64位操作系统，Apple M1 Pro 8 核心CPU，以及统一内存架构的16GB内存。



## 实验结果

### fourclass

fourclass 数据集拥有862个样本点，其中包括两种特征值和两种目标值。在样本点分布上，它具有部分嵌套结构。fourclass数据集常备用作测试 SVM[^t-1]和神经网络[^t-2]。它样本点的分布如图22所示。

<img src="doc/pic/dataset-fourclass.png" alt="dataset-fourclass" style="zoom:50%;" />

在 forclasd 数据集上的实验结果如表-4 所示。超参数 $C_{re}$ 和 $t$ 在取不同值时对预测准确率的影响，其结果如图-1所示；对训练模型所用的时间的影响，其结果如图-2 中所示。

![res-score-fourclass](doc/pic/res-score-fourclass.png)

![res-time-fourclass](doc/pic/res-time-fourclass.png)

并且为了方便评估，我们将 $t = 0.99$ ，$C_{re}$ 取不同值时，CUC 估计器的构造在图-16 表示出来。

![超参数对准确性的影响](doc/pic/超参数对准确性的影响.png)

通过所得到的结果我们可以看到，到当 $C_{re}=3$ 且 $t>0.95$ 时准确率均大于 0.95。且随着 $C_{re}$ 的增加不同区域的边界逐渐变得顺滑，异常值的影响的区域也逐渐减小。但是由于编码单元分割的性质，随着 $C_{re}$ 的逐渐增加，训练模型所用的时间也急剧增加。

|      | 0       | 1       | 2       | 3       |
| ---- | ------- | ------- | ------- | ------- |
| 0.70 | 0.80841 | 0.81308 | 0.85047 | 0.88785 |
| 0.71 | 0.80841 | 0.81308 | 0.84112 | 0.88318 |
| 0.72 | 0.80841 | 0.80841 | 0.85047 | 0.88785 |
| 0.73 | 0.87383 | 0.87850 | 0.92056 | 0.92056 |
| 0.74 | 0.86916 | 0.87850 | 0.91121 | 0.92056 |
| 0.75 | 0.87383 | 0.87850 | 0.92056 | 0.92056 |
| 0.76 | 0.86916 | 0.88318 | 0.92056 | 0.92056 |
| 0.77 | 0.87850 | 0.88318 | 0.91121 | 0.91589 |
| 0.78 | 0.86916 | 0.88318 | 0.92056 | 0.92056 |
| 0.79 | 0.87383 | 0.87850 | 0.91589 | 0.92056 |
| 0.80 | 0.87850 | 0.87850 | 0.92056 | 0.92523 |
| 0.81 | 0.89720 | 0.90187 | 0.92991 | 0.94393 |
| 0.82 | 0.90187 | 0.91121 | 0.93458 | 0.93925 |
| 0.83 | 0.90187 | 0.90187 | 0.93458 | 0.93925 |
| 0.84 | 0.91121 | 0.91121 | 0.93925 | 0.94393 |
| 0.85 | 0.90187 | 0.88785 | 0.92056 | 0.93925 |
| 0.86 | 0.90654 | 0.88785 | 0.92056 | 0.93925 |
| 0.87 | 0.90654 | 0.89252 | 0.92991 | 0.93925 |
| 0.88 | 0.92991 | 0.91121 | 0.94860 | 0.94393 |
| 0.89 | 0.92523 | 0.91121 | 0.94860 | 0.95327 |
| 0.90 | 0.92991 | 0.91589 | 0.94860 | 0.95327 |
| 0.91 | 0.92523 | 0.91589 | 0.93925 | 0.94860 |
| 0.92 | 0.93458 | 0.92056 | 0.93925 | 0.94860 |
| 0.93 | 0.92991 | 0.91589 | 0.94860 | 0.95327 |
| 0.94 | 0.93925 | 0.92523 | 0.95327 | 0.95794 |
| 0.95 | 0.95327 | 0.93925 | 0.94860 | 0.95794 |
| 0.96 | 0.95327 | 0.93458 | 0.94393 | 0.96262 |
| 0.97 | 0.94860 | 0.93925 | 0.94860 | 0.96262 |
| 0.98 | 0.94393 | 0.93458 | 0.94860 | 0.95327 |
| 0.99 | 0.95327 | 0.93925 | 0.95327 | 0.95794 |
| 1.00 | 0.94860 | 0.94393 | 0.95327 | 0.96262 |

|      | 0       | 1       | 2       | 3        |
| ---- | ------- | ------- | ------- | -------- |
| 0.70 | 0.16259 | 0.06305 | 0.13301 | 0.80318  |
| 0.71 | 0.05051 | 0.06038 | 0.13852 | 0.80607  |
| 0.72 | 0.05132 | 0.06295 | 0.13938 | 0.83982  |
| 0.73 | 0.09567 | 0.12362 | 0.36809 | 3.20759  |
| 0.74 | 0.09373 | 0.12568 | 0.35646 | 3.19990  |
| 0.75 | 0.09862 | 0.14906 | 0.35799 | 3.23578  |
| 0.76 | 0.09542 | 0.13095 | 0.35567 | 3.21960  |
| 0.77 | 0.09568 | 0.12831 | 0.36195 | 3.21483  |
| 0.78 | 0.12136 | 0.13072 | 0.35692 | 3.22730  |
| 0.79 | 0.09233 | 0.14821 | 0.35704 | 3.20135  |
| 0.80 | 0.09411 | 0.12176 | 0.38611 | 3.22092  |
| 0.81 | 0.11741 | 0.15925 | 0.50243 | 4.92818  |
| 0.82 | 0.12372 | 0.19768 | 0.53363 | 4.98051  |
| 0.83 | 0.12362 | 0.18549 | 0.50268 | 4.95876  |
| 0.84 | 0.20554 | 0.30040 | 1.38138 | 17.48304 |
| 0.85 | 0.32016 | 0.47259 | 2.80747 | 39.41223 |
| 0.86 | 0.31069 | 0.52308 | 3.01165 | 42.34097 |
| 0.87 | 0.30437 | 0.49908 | 2.98123 | 42.29985 |
| 0.88 | 0.33148 | 0.55461 | 3.35553 | 48.75857 |
| 0.89 | 0.37109 | 0.60752 | 3.77742 | 55.35713 |
| 0.90 | 0.34898 | 0.61034 | 3.83031 | 56.37722 |
| 0.91 | 0.34930 | 0.58464 | 3.80581 | 55.45983 |
| 0.92 | 0.37133 | 0.65142 | 4.23569 | 63.30174 |
| 0.93 | 0.36652 | 0.63345 | 4.30916 | 63.70213 |
| 0.94 | 0.38936 | 0.70433 | 4.71877 | 70.55132 |
| 0.95 | 0.41067 | 0.73092 | 5.21746 | 78.30348 |
| 0.96 | 0.41088 | 0.75078 | 5.25688 | 79.19870 |
| 0.97 | 0.40770 | 0.72674 | 5.28938 | 78.48920 |
| 0.98 | 0.45135 | 0.82231 | 6.31402 | 95.77734 |
| 0.99 | 0.44562 | 0.83152 | 6.35383 | 94.38570 |
| 1.00 | 0.43736 | 0.82263 | 6.35337 | 95.04048 |

### NormalDiagonal

对角线正态分布是一个简单的数据集，其中两类目标值的粒子群呈现正态分布的形态。使用的数据集样本分布如下图所示。

![dataset-NormalDiagonal](doc/pic/dataset-NormalDiagonal.png)

$C_{re}$ 和 $t$ 在取不同值时对预测准确率的影响，其结果如表-1 和图-1中所示；训练模型所用的时间，其结果在表-2和图-2 中所示。

![res-time-NormalDiagonal](doc/pic/res-time-NormalDiagonal.png)

图 24 中展示了$C_{re}$ 和 $t$ 在取不同值时CUC 估计器的形态。

![cuc-NormalDiagonal](doc/pic/cuc-NormalDiagonal.png)

根据结果显示，针对简单结构的数据集，CUC 能够实现高精准度分类。此外，通过调整超参数可以得到不同结构的分类器。

### Checkerboard

Checkerboard 是一种二维 4×4 数据集，常用于大规模评估。在本工作中，所使用的棋盘数据集包含 3600 个样本点，其中 400 个样本点为噪音（异常值）。样本的分布如图-3所示。训练和测试数据集是随机选择的。

<img src="doc/pic/dataset-checkerboard.png" alt="dataset-checkerboard" style="zoom:50%;" />

$C_{re}$ 和 $t$ 在取不同值时对预测准确率的影响，其结果如表-1 和图-1中所示；训练模型所用的时间，其结果在表-2和图-2 中所示。

![res-score-Checkerboard](doc/pic/res-score-Checkerboard.png)

![res-time-Checkerboard](doc/pic/res-time-Checkerboard.png)

当 $t = 0.85$ ，$C_{re}=0$ 时，CUC 估计器的构造如图-16 所示。

<img src="doc/pic/CUC-Cre-0-t-0.85-1.png" alt="CUC-Cre-0-t-0.85-1" style="zoom:67%;" />

通过所得到的结果我们可以看到，对于样本点不具有复杂嵌套结构的数据集，CUC 分类器能够识别且过滤噪声，并进行非常高效的分类。但由于数据集中噪声的存在，预分割阈值 $t$ 和细化分割 $C_{re}$ 的次数不应过高，否则会造成过拟合。并且由于细化分割产生的新编码单元数量是指数倍增长的，如果 $C_{re}$ 次数过大会造成模型训练时长的激增。

### DoubleHelix

在本实验中，所使用的DoubleHelix数据集包含随机生成[^t-dh]的 800 个样本点，并且 DoubleHelix 数据集中的样本点具有双螺旋嵌套结构。样本的分布如图-3所示。

<img src="doc/pic/dataset-DoubleHelix.png" alt="dataset-DoubleHelix" style="zoom: 50%;" />

$C_{re}$ 和 $t$ 在取不同值时对预测准确率的影响，其结果如表-1 和图-1中所示；训练模型所用的时间，其结果在表-2和图-2 中所示。

![res-score-DoubleHelix](doc/pic/res-score-DoubleHelix.png)

![res-time-DoubleHelix](doc/pic/res-time-DoubleHelix.png)

在下图中展示了当 $C_{re}=1$， $t = 0.70, 0.85, 1.00$ 时，CUC 估计器的构造。

![CUC-doublehelix](doc/pic/CUC-doublehelix.png)

根据结果显示，CUC 可对具有双螺旋嵌套结构的数据集进行高效分类，且在不考虑耗时的情况下，分类准确率可达 1.0；同时，在保证训练时长不超过 10 秒的情况下，准确率也能达到 0.99。图-N和表-N表明，在数据样本分布较为弯曲的情况下，提升 $C_{re}$ 和 $t$ 可显著提升准确率，有效避免欠拟合 (underfitting)。

### Sawtooth

Sawtooth 数据集包含两种目标值和两种特征值，共有 400 个样本点。值得注意的是，Sawtooth 数据集中两种样本之间的交接处较为模糊，样本分布如图-N 所示。

<img src="doc/pic/dataset-Sawtooth.png" alt="dataset-Sawtooth" style="zoom: 67%;" />

$C_{re}$ 和 $t$ 在取不同值时对预测准确率的影响，其结果如表-1 和图-1中所示；训练模型所用的时间，其结果在表-2和图-2 中所示。

![res-score-Sawtooth](doc/pic/res-score-Sawtooth.png)

![res-time-Sawtooth](doc/pic/res-time-Sawtooth.png)

在下图中展示了当 $C_{re}=1$， $t = 0.80, 0.85, 0.90$ 时，CUC 估计器的构造。

![CUC-Sawtooth-1](doc/pic/CUC-Sawtooth-1.png)

在下图中展示了当 $C_{re}=0, 1, 2, 3$， $t =0.90$ 时，CUC 估计器的构造。

![CUC-Sawtooth-2](doc/pic/CUC-Sawtooth-2.png)

由得到的结果和 CUC 估计器的构造我们可以分析得到，在较小的数据集中，由于边界数据点的密度较低，一个点或几个点可能不足以提供该数据集分区的足够信息，这会导致结果中的准确率略有下降。

## 可视化

由于编码单元分类器中，每个编码单元都可以视为一个可变大小的像素，所以在数据集特征值数量为 2 的情况下能够对预估器的构造过程进行方便的可视化。下图中展示了 DoubleHelix 数据集在 $C_{re} = 2$ 和 $t = 0.9$ 时预估器的构造过程，里面灰色的方块代表空白编码单元。

<img src="doc/pic/可视化估计器构造过程.png" alt="可视化估计器构造过程" style="zoom: 33%;" />

# 结论

在本文中，我们提出了一种新的机器学习分类器——编码单元分类器。它的特点是算法易于实现，并且不涉及任何复杂的向量计算。当用于训练的数据集特征值的数量为 2 时，分类过程还能够进行可视化。

为了对所提出方法的有效性进行验证，我们在开发出CUC之后在多个具有不同大小和样本点分布的数据集上进行了实验。结果表明，编码单元分类器具有很强的适应性和鲁棒性，在复杂的数据集中可以取得很好的分类效果。其中，编码单元分类器的两个超参数 $C_{re}$ 和 $t$ 能够极大的影响分类器的性能，对于不同场景的分类问题我们可以通过配置这两个超参数来训练出适合于所需场景的分类器预估器。经过尝试，我们建议对于样本点具有螺旋、弯曲等复杂分布形态的数据集，$C_{re}$ 的取值应该为 1 或 2，$t$ 的取值应该大于 0.9。对于具有大量噪声或者异常值的数据集，$C_{re}$ 的取值不应大于 2，且 $t$ 不应过高，随着噪声占比的增大，$t$ 值应该下降，不然会造成模型的过拟合和模型训练时间过长。





本研究的结果意味着编码单元分类器能够适用于大部分场景，并能够有效解决实际应用中的分类问题。



---

# Q&A

Q: 与 SVM 相比有何优势

A: 算法简单，没有复杂的矩阵运算，速度快



Q:

A:



# 引用来源

[^1]:[Gary J. Sullivan, Woo-Jin Han, Thomas Wiegand, 2012. Overview of the High Efficiency Video Coding (HEVC) Standard , IEEE Transactions on circuits and systems for video technology, VOL.22, NO.12](https://iphome.hhi.de/wiegand/assets/pdfs/2012_12_IEEE-HEVC-Overview.pdf)
[^2]: [Machine Learning, wikipedia](https://zh.wikipedia.org/wiki/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0)
[^1-3]:[T.Co ver and P. Hart (1967). "Nearest neighbor pattern classification" in IEEE Transactions on Information Theory, vol.13, no.1, pp.21-27, doi: 10.1109/TIT.1967.1053964.](https://ieeexplore.ieee.org/abstract/document/1053964)
[^1-4]:[Harry Zhang (2004). The Optimality of Naive Bayes](http://www.cs.unb.ca/~hzhang/publications/FLAIRS04ZhangH.pdf)
[^1-5]:[Kevin P. Murphy (2006). Naive Bayes classifiers](https://www.ic.unicamp.br/~rocha/teaching/2011s1/mc906/aulas/naive-bayes.pdf)

[^2-1]:Russell, Stuart; Norvig, Peter. Artificial Intelligence: A Modern Approach. 2nd. Prentice Hall. 2003 [1995]. ISBN 978-0137903955
[^2-2]:Shannon, C. E. (1949). Communication Theory of Secrecy Systems. Bell System Technical Journal 28 (4): 656–715.
[^2-3]:Morgan. J. N. & Sonquist, J. A. (1963) Problems in the Analysis of Survey Data, and a Proposal, Journal of the American Statistical Association, 58:302, 415-434.
[^2-4]:Gordon V. K. (1980).An  Exploratory Technique for Investigating Large Quantities of Categorical Data, Applied Statistics. 29(2): 119–127.
[^2-5]:Breiman, L.; Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). Classification and regression trees. Monterey, CA: Wadsworth & Brooks/Cole Advanced Books & Software.
[^2-6]:Quinlan, J. R. (1986). Induction of Decision Trees. Mach. Learn. 1(1): 81–106
[^2-7]:Quinlan, J. R. (1993). C4.5: Programs for machine learning. San Francisco,CA: Morgan Kaufman.
[^2-8]:Loh, W. Y., & Shih, Y. S. (1997). Split selection methods for classification trees. Statistica sinica, 815-840.
[^2-9]:Freund, Y., & Mason, L. (1999, June). The alternating decision tree learning algorithm. In icml (Vol. 99, pp. 124-133).

[^2-10]:[Mos Zhang, Yuanyuan Li (2022). Decision tree learning](https://www.jiqizhixin.com/graph/technologies/80fbc146-bc42-4585-93e3-21c2dd5ca63f).
[^2-11]:Tin Kam Ho. [Random decision forests](https://web.archive.org/web/20210303185509/https://ieeexplore.ieee.org/document/598994/). Proceedings of 3rd International Conference on Document Analysis and Recognition (Montreal, Que., Canada: IEEE Comput. Soc. Press). 1995, **1**: 278–282 [2020-03-04]. [ISBN 978-0-8186-7128-9](https://zh.m.wikipedia.org/wiki/Special:网络书源/978-0-8186-7128-9). [doi:10.1109/ICDAR.1995.598994](https://dx.doi.org/10.1109%2FICDAR.1995.598994).
[^2-12]:Leo Breiman (2001).  Random forests.
[^2-13]:Biau, G., Scornet, E. A (2016). random forest guided tour. *TEST* **25**, 197–227. https://doi.org/10.1007/s11749-016-0481-7
[^2-14]:Vapnik, V.N. and Lerner, A.Y., 1963. Recognition of patterns with help of generalized portraits. Avtomat. i Telemekh, 24(6), pp.774-780.
[^2-15]:Vapnik, V. and Chervonenkis, A., 1964. A note on class of perceptron. Automation and Remote Control, 24.
[^2-16]:Cover, T.M., 1965. Geometrical and statistical properties of systems of linear inequalities with applications in pattern recognition. IEEE transactions on electronic computers, (3), pp.326-334.
[^2-17]: Smith, F.W., 1968. Pattern classifier design by linear programming. IEEE Transactions on Computers, 100(4), pp.367-372.
[^2-18]:Vapnik, V.N. and Chervonenkis, A.Y., 2015. On the uniform convergence of relative frequencies of events to their probabilities. In Measures of complexity (pp. 11-30). Springer, Cham.
[^2-19]:Boser, B.E., Guyon, I.M. and Vapnik, V.N., 1992, July. A training algorithm for optimal margin classifiers. In Proceedings of the fifth annual workshop on Computational learning theory (pp. 144-152). ACM.
[^2-20]:Cortes, C. and Vapnik, V., 1995. Support-vector networks. Machine learning, 20(3), pp.273-297.
[^2-21]:Li Hang．Statistical Learning Methods. Beijing: Tsinghua University Press, 2012: Chapter 7, pp. 95-135
[^2-22]:[Cox, D.R. (1958) The Regression Analysis of Binary Sequences. Journal of the Royal Statistical Society: Series B, 20, 215-242.](https://www.jstor.org/stable/2983890)
[^2-23]:[Stephanie Kay Ashenden (2021). The Era of Artificial Intelligence, Machine Learning, and Data Science in the Pharmaceutical Industry.](https://www.sciencedirect.com/book/9780128200452/the-era-of-artificial-intelligence-machine-learning-and-data-science-in-the-pharmaceutical-industry#book-info)

[^gr-1]:[Virus transmission and epidemiology (2023). Viruses (Second Edition), Understanding to Investigation, Pages 59-71](https://www.sciencedirect.com/science/book/9780323903851)

[^gr-2]:Hao Hu, Karima Nigmatulina, Philip Eckhoff, 2013. The scaling of contact rates with population density for the infectious disease models. [Mathematical Biosciences](https://www.sciencedirect.com/journal/mathematical-biosciences)[Volume 244, Issue 2](https://www.sciencedirect.com/journal/mathematical-biosciences/vol/244/issue/2), Pages 125-134
[^fourclass]: Tin Kam Ho and Eugene M. Kleinberg, 1996. Building projectable classifiers of arbitrary complexity. In Proceedings of the 13th International Conference on Pattern Recognition, pages 880-885, Vienna, Austria.
[^t-1]:Omid Naghash Almasi, Modjtaba Rouhani, 2021. A geometric-based data reduction approach for large low dimensional datasets: Delaunay triangulation in SVM algorithms, Machine Learning with Applications, Volume 4, 100025
[^t-2]:TaiYu Cheng, Yukata Masuda, Jun Chen, Jaehoon Yu, Masanori Hashimoto, 2020. Logarithm-approximate floating-point multiplier is applicable to power-efficient neural network training, Integration, Volume 74, Pages 19-31
[^t-dh]: Generating Spiral Dataset for Classifying in Python, GitHub Gist [Электронный ресурс]. URL: https://gist.github.com/45deg/e731d9e7f478de134def5668324c44c5





















