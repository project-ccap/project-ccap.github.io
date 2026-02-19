---
title: Bayes for CCAP
author: Shin Asakawa
---
<link href="asamarkdown.css" rel="stylesheet"></link>

$$
\newcommand{\bs}[1]{\boldsymbol{#1}}
\newcommand{\mb}[1]{\boldsymbol{#1}}
% \newcommand{\mb}[1]{\mathbf{#1}}
\newcommand{\Brc}[1]{\left(#1\right)}
\newcommand{\BRc}[1]{\left[#1\right]}
\newcommand{\Rank}{\text{rank}\;}
\newcommand{\Hat}[1]{\widehat{#1}}
\newcommand{\Prj}[1]{\mb{#1}\Brc{\mb{#1}^{\top}\mb{#1}}^{-1}\mb{#1}^{\top}}
\newcommand{\RegP}[2]{\Brc{\mb{#1}^{\top}\mb{#1}}^{-1}\mb{#1}^{\top}\mb{#2}}
\newcommand{\NSQ}[1]{\left|\mb{#1}\right|^2}
\newcommand{\Norm}[1]{\left|#1\right|}
\newcommand{\IP}[2]{\left({#1}\cdot{#2}\right)}
\newcommand{\Bar}[1]{\overline{\;#1\;}}
\newcommand{\of}[1]{\left(#1\right)}
\newcommand{\Of}[1]{\left[#1\right]}
\newcommand{\OF}[1]{\left\{#1\right\}}
\newcommand{\widebar}[1]{\overline{#1}}
$$

# ベイズ統計学とベイズ機械学習の相違

1. 推測統計学における帰無仮説検定 (NHST: Null Hypothesis Statistical Tests) と ベイズ統計学におけるベイズ因子
2. 回帰分析の違い
    1. 推測統計学における回帰分析
    2. ベイズ回帰分析
    3. ガウス過程
3. パス解析 (構造方程式モデル) とグラフィカルモデルとの相違

See also:

* [初めてのベイズ学習](2023_1123var_basic.md)
* [確率的機械学習と人工知能, Ghahramani+(2015)](2015Ghahramani_BayesianML_ja.md)
* [ベイズ推定のための変分近似法: EM アルゴリズム以降の生活, Tzikas+(2008)](2008Tzikas_Variational_Bayes_ja.md)
* [ベイズ統計学の基礎, Gelman+(2013)](2013Gelman_Bayesian_Statistics_ja.md)
