# 支持向量机（SVM）

支持向量机（Support Vector Machine，SVM）是 Vapnik 等人在 1964 年提出的一种广义线性分类器，支持向量机通过核方法可将非线性分类问题转换成线性分类问题，从而有着更快的运算速度和分类准确度。

## 支持向量机解决线性判别分类问题

此处基于两类线性可分的样本来探讨 SVM 的原理，其中两类样本线性可分，其分界面为线性分界面。支持向量机中确定的线性分界面如下图所示，图中 $H_1$ 为第一类样本所在平面， $H_2$ 为第二类样本所在平面，两平面平行，线性分界面 $H$ 为两平面的平分面。

<div align = 'center'><img src = '../Document\pic\屏幕截图 2023-11-29 215315.png' width = '300' title = 'SVM 两类线性可分样本'></div>

支持向量即处在隔离带的边缘上的样本点，它们决定了这个隔离带。由图可知，支持向量的选择有很多，那么其所在平面 $H_1$ 、 $H_2$ 也有很多，最终导致其平分面即线性分界面是不定的，这在识别分类器的设计中无疑是错误的。为了根据一定的标准选择支持向量从而确定线性分类面，Vapnik 等人选择最大间隔准则作为此标准。

训练样本集： $\{X_i,\ y_i \},\ i = 1,\dots ,N$ ，其中 $X_i$ 为 d 维特征向量； $y_i = {-1,+1}$ ，表示其类别。分界面 $H$ ： $\mathbf{W}^T X_i + w_0 = 0$ 。令 $\mathbf{W}^T X_i + w_0 \ge 0$ ，若 $y_i = +1$ ； $\mathbf{W}^T + w_0 = \le -1$ ，若 $y_i = -1$ 。对在 $H_1$ 和 $H_2$ 平面上的点，两式取等号。两式也可合并成 $y_i(\mathbf{W}^T X_i + w_0) \ge 1$ 。

$H_1$ 平面到坐标原点的距离为 $\frac{(-w_0 + 1)}{\left \| \mathbf{W}\right \|}$ ，而 $H_2$ 平面到坐标原点的距离为 $\frac{(-w_0 - 1)}{\left \| \mathbf{W}\right \|}$ 。故 $H_1$ 到 $H_2$ 的间隔为： $\frac{2}{\left \| \mathbf{W} \right \|}$ 。间隔最大的准则应使 $J(\mathbf{W}) = \left \| \mathbf{W} \right \|^2$ 最小。约束条件： $y_i(\mathbf{W}^T X_i + w_0) - 1\ge 0$ 。

采用采用拉格朗日乘子法求解，建立拉格朗日函数：

$$L(\mathbf{W},w_0,a) = \frac{1}{2}\left \| \mathbf{W} \right \|^2 - \sum_{i=1}^{N} a_i [y_i(\mathbf{W}^T X_i + w_0) - 1]$$

令

$$\begin{cases}
\frac{\partial}{\partial \mathbf{W}} L(\mathbf{W},a) &= \mathbf{W} - \sum_{i=1}^N a_iy_iX_i = 0 \\
\frac{\partial}{\partial w_0} &= \sum_{i=1}^N a_iy_i = 0
\end{cases}$$

有

$$\begin{cases}
\mathbf{W} &= \sum_{i=1}^N a_i y_i X_i \\
\sum_{i=1}^N &= 0
\end{cases}$$