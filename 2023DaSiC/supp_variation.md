---
title: "機械学習からみた言語モデルの鏡 DaSiC7 (2023) 発表資料 (3)"
author: 浅川伸一
layout: default
codemirror_mode: python
codemirror_mime_type: text/x-cython
---
<link href="/assets/css/asamarkdown.css" rel="stylesheet">
[DaSiC 7 (2023)](https://sites.google.com/view/dasic7-2023) Linguistics and Data Science in Collaboration 発表資料

<div align='right'>

Copyright (C) 2023 Shinichi Asakawa<br/>
<a href='mailto:asakawa@ieee.org'>Shin Aasakawa</a>, all rights reserved.<br/>
https://opensource.org/license/mit/
</div>

# 変分問題と標準正則化


## 0.1 機械学習における正則化

$$
\begin{aligned}
\text{損失関数} &= \text{誤差関数} + \text{ペナルティ項}\\
\mathcal{L}\left(y,x;\theta\right) &=\left\|y-f(x;\theta)\right\|^2+\lambda\left\|P\theta\right\|^2
\end{aligned}
$$

損失関数は目標関数とも呼ばれる。誤差関数は最小自乗誤差 least squared error が用いられる(回帰との類推では)。
ニューラルネットワークによる画像分類[@2012AlexNet]，主成分分析[@1901Pearson_PCA]，標準正則化理論(画像復元)[@Poggio1985]，画像分類課題では次式交差エントロピー誤差[@1989Hinton]が用いられる。

$$
\text{CEE}=-\sum_{j,c}y_{j,c}\log_2\left(f(x_{j,c})\right)+\left(1-y_{j,c}\right)\log_2\left(1-f(x_{j,c})\right),
$$
<!--[@1995GirosiPoggio]
- See https://ermongroup.github.io/cs228-notes/inference/variational/-->

$$
H[f]=\sum_{i=1}^{N}
\left(y_i-f\left(\mathbf{x}_i\right)\right)^2+\lambda\left\|Pf\right\|^2
$$
ここで $P$ は制約演算子，$\left\|\cdot\right\|$ はノルムを表す $\lambda$ は正の有理数であり正則化パラメータと呼ばれる。
汎関数 $H$ の最小化は Euler-Lagrange 方程式として定式化される。
$P$ は事前知識を表す。 [GirosiPoggio1990]

$$
\hat{P}Pf\left(\mathbf{x}\right)=\frac{1}{\lambda}\sum_{i=1}^N\left(y_i-f\left(\mathbf{x}\right)\right)\,\delta\left(\mathbf{x}-\mathbf{x}_i\right),
$$
ここで $\hat{P}$ は微分演算子 $P$ の随伴演算子であり，微分方程式，右辺の

上記は偏微分方程式であり，その解は微分作用素 $G$ のグリーン関数で与えられるカーネルを持つ右辺の積分変換，すなわち以下の分布微分方程式を満たす関数 $G$ として書けることがよく知られている：
<!-- The above is a partial differential equation, and it is well known that its solution can be written as the integral transformation of its right side with a kernel given by the Green's function of the differential operator $\hat{P}P$, that is the function $G$ satisfying the following distributional differential equation: -->

$$\tag{4}
\hat{P}P\,G\left(\mathbf{x};y\right)=\delta\left(\mathbf{x}-y\right).
$$
(4) 式にデルタ関数が現れるので，積分変換は離散和になり，$f$ は次のように書ける:
<!-- Because of the delta functions appearing in (4) the integral transformation becomes a discrete sum and $f
$ can then be written as -->
$$
f(\mathbf{x})=\frac{1}{\lambda}\sum_{i=1}^N(y_i-f(\mathbf{x}_i))
G(\mathbf{x};\mathbf{x}_i).
$$
式 (5) は，正則化問題の解が滑らかな関数の空間の $N$ 次元部分空間にあることを示している。
この部分空間の基底は $N$ 個の関数 $G(\mathbf{x};mathbf{x}_j)$ によって与えられる。
以下では，$G(\mathbf{x};mathbf{x}_j)$ を点 $\mathbf{x}_i$ を「中心とする」グリーン関数，点$\mathbf{x}_i$ を展開の「中心」と呼ぶ。
この理由は，通常グリーン関数は遷移的に不変であり，$G=G(\mathbf{x}-\mathbf{x}_i)$ であり，この場合 $G(\mathbf{x})$ と $G(\mathbf{x}-\mathbf{x}_i)$ とは，原点に $\mathbf{x}_i$ を写す座標変換によって等価になることにある。
<!-- Equation (5) says that the sol ution of the regularization problem lies in an N-dimensional subspace of the space of smooth functions.
A basis for this subspace is given by the $N$ functions $G(\mathbf{x};\mathbf{x}_j)$.
In the following we will refer to $G(\mathbf{x}; \mathbf{x}_j)$ as to the Green's function "centered" at the point $\mathbf{x}_i$, and to the points $\mathbf{x}_i$ as to the "centers" of the expansion.
The reason for this lies in the fact that usually the Green's function is transitionally invariant, that is $G=G(\mathbf{x}-\mathbf{x}_i)$, and in this case $G(\mathbf{x})$ and $G(\mathbf{x}-\mathbf{x}_i)$ are equivalent  modulo a coordinates translation that maps $\mathbf{x}_i$ in the origin. -->

R. Courant and D. Hilbert, Methods of Mathematica/Physics, Vol. 2. Interscience, London, England, 1962.
where $c$, which is related to $P_d(d)$, depens only on $d$.

物理学の場合 $\lambda$ が定まる，あるいは意味を持つ場合があるが，機械学習，ニューラルネットワークの場合定まる
とはかぎらない。ハイパーパラメータとして扱われる場合が多い。

<center>
<img src="/figures/2018Tschannen_Fig1.svg" style="width:74%"><br/>
<img src="/figures/2018Tschannen_Fig2.svg" style="width:74%"><br/>
</center>


# 標準正則化理論と条件付き最適化

視覚情報処理の分野では，David Marr や Tomaso Poggio らによって視覚情報処理を定式化する研究が行われた。
以下に論文を引用する。

<center>
<img src="/figures/1985Poggio_2.svg" style="width:33%"><br/>
</center>

以下に上記引用部分の拙訳を付ける:

データ $y$ から $z$ を見つけ出す不良設定問題の正則化
$$
Az = y
$$
では，正則化項 $\left\|\cdot\right\|$ の選択と汎関数の安定化項 $\left\|Pz\right\|$ が必要となる。
標準正則化理論においては，$A$ は線形演算子，ノルムは 2 次，$P$ は線形である。
2 種類の方法が適用可能である。
すなわち
1. $\left\|Az-y\right\|\leqslant\epsilon$ を満たし，次式を最小化する $z$ を探す
$$
\left\|Pz\right\|^2
$$

2. 次式を最小化する $z$ を探す
$$
\left\|Az-y\right\|+\lambda\left\|Pz\right\|^2,
$$
ここで $\lambda$ はいわゆる正則化パラメータである。

最初の方法は，十分にデータを近似し，かつ，「基準」$\left\|Pz\right\|$ を最小化するという意味で「正則」な $z$ を探す方法である。
二番目の方法は，$\lambda$ が正則化の程度と解のデータへの近似とをコントロールする。
標準正則化理論は，最良の $\lambda$ を決定する手法を提供する。
標準正則化の手法は，上式に制約を導入することで変分原理の問題としている。
最小化するコストは物理的制約条件を満たす良い解を反映している。
すなわち，データへの近似もよく，かつ，正則化項 $\left\|Pz\right\|^2$ も小さいことを意味する。
$P$ は問題の物理的制約を表しており，2 次の変分原理であり，解空間内での唯一解が存在する。
標準正則化手法は，不良設定問題に対して注意深い分析が必要であることを注記しておく。
ノルム $\left\|\cdot\right\|$，正則化関数 $\left\|Pz\right\|$, および，汎関数空間の選択は数学的性質と，物理的説得性を有する必要がある。
これらにより，正しい正則化の詳細条件が定まる。

変分原理は物理学，経済学，工学，で幅広く用いられている。例えば物理学における基本法則は変分原理を用いて，
エネルギーやラグランジェ関数を用いて簡潔に表現されている。

<!--
- [上を訳してみました。github.io だと数式が表示されない場合があるため colab にしています](https://github.com/komazawa-deep-learning/komazawa-deep-learning.github.io/blob/master/notebooks/2020_0529Poggios_standard_regularization_translation.ipynb){:target="_blank"}
-->

様々な視覚課題に適用されていて，以下のようなリストが挙げられる：

<center>
<img src="/figures/1985Poggio_1.svg" style="width:34%">
<img src="/figures/1985Poggio_3math.svg" style="width:34%"><br/>
</center>

1. 縁検出 Edge detection $\displaystyle\int[(Sf-i)^2 +\lambda(f_{xx}^2)]dx$
1. 光学フローの計算 Computation of optical flow $\displaystyle\int[(V\cdot N - V^N )^2+\lambda(\partial/\partial_x)v^2]dx$
1. 表面の再構成 $\displaystyle\int[(S\cdot f - d)^2+\lambda(f_{xx}^2+2f_{xy}^2+f_{yy}^2)^2]dxdy$
1. 時空間近似 spatiotemporal approximation: $\displaystyle\int[(S\cdot i)^2+\lambda(\nabla fV+f_t)^2]dxdydt$
1. 色: $\displaystyle\|I^v-Az\|^2 +\lambda\|Pz\|^2$
1. 陰影からの形状復元 shape from shading: $\displaystyle\int[(E-R(f,g))^2+\lambda(f_x^2+f_y^2+g_x^2+g_y^2)]dxdy$
1. 立体視: $\displaystyle \int\left[\nabla^{2} G * \left(L(x,y) - R(x+ d(x,y),y)\right)^{2}+\lambda(\nabla d)^{2}\right] dxdy $
1. 時空間内挿，近似 Spatio-temporal interpolation and approximation $\displaystyle\int\left[(i_x+i,v+i)^2+\lambda(u_x^2+u_y^2+v_x^2+v_y^2)\right]dxdy$
1. 明度，環境光の計算 Computation of lightness and albedo
1. 輪郭線からの形状復元 Shape from contours
1. キメからの形状復元 Shape from texture
1. 陰影からの形状復元 Shape from shading
1. 両眼立体視 Binocular stereo matching
1. 運動からの形状復元 Structure from motion
1. 両眼立体視 Structure from stereo
1. 表面復元 Surface reconstruction
1. 表面色の計算 Computation of surface colour
