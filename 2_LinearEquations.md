#! https://zhuanlan.zhihu.com/p/333864068

本文使用 jupyter notebook 和 Julia 写成，原始 notebook 可在 [github 仓库](https://github.com/HQJo/LinearAlgebraNotes) 找到。

## 线性方程组有解的条件

线性方程组与矩阵存在着密切的联系，大部分教科书都是从线性方程组引入矩阵的。一个包含 $m$ 个 方程和 $n$ 个变量的方程组，可以用一个 $m\times n$ 的矩阵与 $\mathbb{R}^{n}$ 的向量相乘来表示：

$$
\begin{cases}
    a_{11} x_1 + a_{12} x_2 + \cdots a_{1n} x_n = b_1 \\
    a_{21} x_1 + a_{22} x_2 + \cdots a_{2n} x_n = b_2 \\
    \cdots \\
    a_{m1} x_1 + a_{m2} x_2 + \cdots a_{mn} x_n = b_m
\end{cases} \quad \Rightarrow \quad A \mathrm{x} = \mathrm{b}
$$

$x$ 即我们欲求解的变量。

在上一节中，我们知道矩阵向量乘可以看成列向量的线性组合，线性组合的系数就是 $\mathrm{x}$。那么，解线性方程组的问题可以看成：**怎样对 $A$ 的列进行线性组合，才能得到 $\mathrm{b}$？**

定义 $A$ 的列空间为其列向量的张成（即所有线性组合组成的集合），那么:**方程组 $A\mathrm{x} = \mathrm{b}$ 有解等价于 $\mathrm{b}$ 在 $A$ 的列空间中；$A\mathrm{x} = \mathrm{b}$ 有解等价于 $\mathrm{b}$ 不在 $A$ 的列空间中。**

我们知道，列空间的大小即 $A$ 的列秩。如果 $\mathrm{b}$ 在 $A$ 的列空间中，那么将 $\mathrm{b}$ 加入 $A$ 的列不会改变其列秩；否则，秩比原来多 1。如果把 $b$ 加入 $A$ 的列得到的矩阵记为增广矩阵 $\tilde{A}$。那么：**方程组 $A\mathrm{x} = \mathrm{b}$ 有解等价于 $A$ 的列秩等于 $\tilde{A}$ 的列秩**

总结一下，我们得到了两个 $A\mathrm{x} = \mathrm{b}$ 有解的等价条件：

1. $\mathrm{b}$ 在 $A$ 的列空间中；
2. $A$ 的列秩等于增广矩阵 $\tilde{A}$ 的列秩；

## 线性方程组解的数目

上面介绍了线性方程组有解/无解的判断条件，但有解又分为有唯一解和无数解两种情况。因为如果有两个不同的解，那么它们的带权平均也是一个解，因此能得到无数个解。

$$
    A\mathrm{x}_1 = \mathrm{b},\, A\mathrm{x}_2 = \mathrm{b} \quad \Rightarrow \quad A(\lambda \mathrm{x}_1 + (1-\lambda) \mathrm{x}_2) = \mathrm{b}
$$

假设 $A\mathrm{x} = \mathrm{0}$ 有非零解，即存在一个 $\hat{\mathrm{x}}$，满足 $A\hat{\mathrm{x}} = \mathrm{0}$。那么假设 $A\mathrm{x} = \mathrm{b}$ 有一个解 $\mathrm{x}$，那么 $\mathrm{x}$ 加上任意倍的 $\hat{\mathrm{x}}$ 也是 $A\mathrm{x} = \mathrm{b}$ 的一个解。

$$
A(\mathrm{x} + \lambda \hat{\mathrm{x}}) = \mathrm{b} + \mathrm{0} = \mathrm{b}
$$

定义 $A$ 的核为 $A\mathrm{x} = 0$ 的解空间，若 $A\mathrm{x} = \mathrm{b}$ 有解，则**有唯一解等价于 $A$ 的核为 $\{0\}$，即 $A\mathrm{x} = \mathrm{0}$ 只有平凡解 $\mathrm{0}$；有无数解等价于 $A$ 的核维数大于 0，即 $A\mathrm{x} = \mathrm{0}$ 有非平凡解**。

又由于 $A$ 的核的维数与 $A$ 的列空间维数之和为 $n$，那么上述结论还可以写成：若 $A\mathrm{x} = \mathrm{b}$ 有解，则**有唯一解等价于 $A$ 列空间的秩为 $n$，即 $A$ 列满秩；有无数解等价于 $A$ 的列秩小于 $n$，即 $A$ 不列满秩**。

当 $m = n$ 时，$A$ 为方阵。此时，若方程组 $A\mathrm{x} = \mathrm{b}$ 行列式不为 0（满秩），则可以通过克莱姆法则得到唯一解；如果行列式为 0，说明 $A$ 列不满秩，此时需要判断 $\mathrm{b}$ 在不在 $A$ 的列空间中判定是有无数解还是无解。
