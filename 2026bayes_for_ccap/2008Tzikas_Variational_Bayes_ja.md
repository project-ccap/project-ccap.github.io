---
author: Dimitris G. Tzikas, Aristidis C. Likas, and Nikolaos P. Galatsanos
title: "Tzikas2008 ベイズ推論の変分アプローチ"
etitle: The Variational Approximation for Bayesian Inference
date: 2008
---
<link href="/asamarkdown.css" rel="stylesheet"></link>
<div class="title">

ベイズ推定のための変分近似法: EMアルゴリズム以降の生活<br/>
The Variational Approximation for Bayesian Inference.  Life after EM algorithm<br/>
Dimitris G. Tzikas, Aristidis C. Likas, and Nikolaos P. Galatsanos (2008)
</div>

図 1 の左上に示した Thomas Bayes (1701-1761) は，その名の通り，死後 3 年経った1764 年に発表した論文で初めてベイズの定理を発見した。
ただし，ベイズはその定理で，一様事前分布を用いている [1]。
図 1 の右下に示した Pierre-Simon Laplace (1749-1827) は，Bayes の研究 を知らなかったようだ。
だが 25 歳の時に書いた手記でより一般的な形で 同定理を発見し，その広い応用可能性を示した [2]。
これらの問題について S.M. Stiegler はこう書いている:
<!-- Thomas Bayes (1701–1761), shown in the upper left corner of Figure 1, first discovered Bayes’ theorem in a paper that was published in 1764 three years after his death, as the name suggests.
However, Bayes, in his theorem, used uniform priors[1].
Pierre-Simon Laplace (1749–1827), shown in the lower right corner of Figure 1, apparently unaware of Bayes’ work, discovered the same theorem in more general form in a memoir he wrote at the age of 25 and showed its wide applicability[2].
Regarding these issues S.M. Stiegler writes:-->

この手記が与えた影響は計り知れない。
Bayes 自身の論文は 1780 年まで無視され，20 世紀まで科学的議論において重要な役割を果たさなかったので，「Bayes 的」な考え方はここから数学界に初めて広まったのである。
また，現在も採用されている事後分布の漸近的解析の数学的手法を導入したのも，この Laplace の論文であった。
また，最適推定の最古の例として，事後期待損失を最小化する推定量の導出とその特徴づけが行われたのも，この論文である。
2 世紀以上の時を経て，我々数学者，統計学者は，この科学の傑作に自分たちのルーツを認めるだけでなく，そこから学ぶことができるのである [3]。
<!-- > The influence of this memoir was immense.
It was from here that “Bayesian” ideas first spread through the mathematical world, as Bayes’s own article was ignored until 1780 and played no important role in scientific debate until the 20th century.
It was also this article of Laplace’s that introduced the mathematical techniques for the asymptotic analysis of posterior distributions that are still employed today.
And it was here that the earliest example of optimum estimation can be found, the derivation and characterization of an estimator that minimized a particular measure of posterior expected loss.
After more than two centuries, we mathematicians, statisticians cannot only recognize our roots in this masterpiece of our science, we can still learn from it.[3]-->

<div class="figure figcenter">
<img src="../figures/2008Tzikas_fig1.jpg">
<div class="figcaption">

図 1.
Thomas Bayes (左上) と Pieere-Simon Laplace (右下) は 1700 年代に数学で同様の定理を発見し，新しい技術を数学界に広めたが，それは 2 世紀以上たった今でも使われている。
<!-- FIG 1.
Thomas Bayes (upper left) and Pierre-Simon Laplace (lower right) discovered similar theorems in mathematics in the 1700s, spreading new techniques throughout the mathematic world that are still used more than two centuries later. -->
</div></div>

最尤法 (ML) は，現代の統計的信号処理で最もよく使われる手法の 1 つである。
EM アルゴリズムは，ML 推定のための反復纂法であり，多くの利点を持ち，統計的信号処理の問題を解くための標準的な方法論となっている。
しかし EM アルゴリズムにはある種の要件があり，複雑な問題への適用を著しく制限している。
最近 EM アルゴリズムの制約条件を緩和した変分ベイズ推論と呼ばれる新しい手法が登場し，急速に普及しつつある。
さらに EM アルゴリズムがこの方法論の特殊な場合と見なすことができることを示すことができる。
本稿では，まず，信号処理のコミュニティを対象としたベイズ変分推論のチュートリアルの紹介を行う。
線形回帰と Gauss 混合モデリングを例として，EM アルゴリズムと比較して，Bayes 変分推論が提供する追加的な機能を示す。
<!-- Maximum likelihood (ML) estimation is one of the most popular methodologies used in modern statistical signal processing.
The expectation maximization (EM) algorithm is an iterative algorithm for ML estimation that has a number of advantages and has become a standard methodology for solving statistical signal processing problems.
However, the EM algorithm has certain requirements that seriously limit its applicability to complex problems.
Recently, a new methodology termed “variational Bayesian inference” has emerged, which relaxes some of the limiting requirements of the EM algorithm and is gaining rapidly popularity.
Furthermore, one can show that the EM algorithm can be viewed as a special case of this methodology.
In this article, we first present a tutorial introduction of Bayesian variational inference aimed at the signal processing community.
We use linear regression and Gaussian mixture modeling as examples to demonstrate the additional capabilities that Bayesian variational inference offers as compared to the EM algorithm. -->

# 1. はじめに<!-- # 1. INTRODUCTION-->

ML の方法論は，現代の統計的信号処理の基本的な定番の 1 つである。
EM アルゴリズムは，ML 推定値を得るために多くの利点を提供する反復アルゴリズムである。
1977 年に Dempster らによって正式に発表されて以来[4]，EM アルゴリズムは ML 推定の標準的な方法論となった。
IEEE のコミュニティでも，EM は着実に人気を集めており，応用例も増えてきている。
IEEE のジャーナルに EM アルゴリズムが初めて掲載されたのは 1988 年であり，光子制限画像の断層像再構成の問題を扱っていた[5, 6]。
以来 EM アルゴリズムは，画像や映像の復元やセグメンテーション，画像のモデリング，通信や音声認識における搬送波周波数同期やチャネル推定など，幅広い用途で用いられる統計的信号処理のツールとして人気を博してきた。
<!-- The ML methodology is one of the basic staples of modern statistical signal processing.
The EM algorithm is an iterative algorithm that offers a number of advantages for obtaining ML estimates.
Since its formal introduction in 1977 by Dempster et al.[4], the EM algorithm has become a standard methodology for ML estimation.
In the IEEE community, the EM is steadily gaining popularity and is being used in an increasing number of applications.
The first publications in IEEE journals making reference to the EM algorithm appeared in 1988 and dealt with the problem of tomographic reconstruction of photon limited images[5],[6].
Since then, the EM algorithm has become a popular tool for statistical signal processing used in a wide range of applications, such as recovery and segmentation of images and video, image modelling, carrier frequency synchronization, and channel estimation in communications and speech recognition. -->

EM アルゴリズムの背後にある概念は非常に直感的で自然なものである。
EM 的なアルゴリズムは[4] よりも前から統計学の文献に存在していたが，そのようなアルゴリズムは実際には特殊な文脈における EM アルゴリズムであった。
このような最初の文献は 1886 年に遡り，Newcomb が 2 つの一変量規範の混合物のパラメータの推定を考察している[7]。
しかし，そのようなアイデアが統合され，EM アルゴリズムの一般的な定式化が確立されたのは [4] であった。
[4] 以前の EM アルゴリズムの歴史に関する良いサーベイが [8] にある。
<!-- The concept behind the EM algorithm is very intuitive and natural.
EM-like algorithms existed in the statistical literature even before [4], however such algorithms were actually EM algorithms in special contexts.
The first known such reference dates back to 1886, when Newcomb considers the estimation of the parameters of a mixture of two univariate normals[7].
However, it was in [4] where such ideas were synthesized and the general formulation of the EM algorithm was established.
A good survey on the history of the EM algorithm before [4] can be found in [8].-->

本論文は EM アルゴリズムに関するチュートリアルではない。
このようなチュートリアルは 1996 年に IEEE Signal Processing Magazine に掲載された[9]。
本論文の目的は EM アルゴリズムの欠点を改善する統計的推測のための新しい方法論を提示することである。
この方法は変分近似と呼ばれ[10]，EMアルゴリズムが適用できない複雑なベイズモデルを解くために用いることができる。
変分近似に基づく Bayes 推定は，それが最初に導入された 1990 年代半ば以来，機械学習コミュニティによって広く利用されてきた。
<!-- The present article is not a tutorial on the EM algorithm.
Such a tutorial appeared in 1996 in IEEE Signal Processing Magazine [9].
The present article is aimed at presenting an emerging new methodology for statistical inference that ameliorates certain shortcomings of the EM algorithm.
This methodology is termed variational approximation [10] and can be used to solve complex Bayesian models where the EM algorithm cannot be applied.
Bayesian inference based on the variational approximation has been used extensively by the machine learning community since the mid-1990s when it was first introduced. -->

* [4] A. Dempster, N. Laird, and D. Rubin, “Maximum likelihood from incomplete data via the EM algorithm,”J. Roy. Statis. Soc. A, vol. 39, no. 1, pp. 1–38, 1977.
* [5] W. Jones, L. Byars, and M. Casey, “Positron emission tomographic images and expectation maximization: A VLSI architecture for multiple iterations per second,” IEEE Trans. Nuclear Sci., vol. 35, no. 1, pp. 620–624, 1988.
* [6] Z. Liang and H. Hart, “Bayesian reconstruction in emission computerized tomography,” IEEE Trans. Nuclear Sci., vol. 35, no. 1, pp. 788–792, 1988.
* [7] S. Newcomb, “A generalized theory of the combination of observations so as to obtain the best result,” Amer. J. Math., vol. 8, pp. 343–366, 1886.
* [8] G. McLachlan and T. Krishnan, The EM Algorithm and Extensions. New York: Wiley, 1997.
* [9] T.K. Moon, “The EM algorithm in signal processing,” IEEE Signal Process. Mag., vol. 13, no. 6, pp. 47–60, 1996.
* [10] V. Smídl and A. Quinn, The Variational Bayes Method in Signal Processing. New York: Springer-Verlag, 2005.

# 2. ベイズ推論の基本<!-- # 2. BAYESIAN INFERENCE BASICS-->

$x$ を観測値，$\theta$ を $x$ を生成したモデルの未知のパラメータと仮定する。
本稿では，**パラメータを指す場合は推定，確率変数を指す場合は推論という用語を厳密に使用する** ことにする。
<font style="color:teal; font-weight:900">推定という用語は，不完全で不確かで雑音の多いデータから，パラメータの値を計算で近似することを意味する。</font>
対照的に，<font style="color:blue; font-weight:900">推論という用語はベイズ推論を意味するものとして使われ，観察値 $x$ を与えられた確率変数 $\theta$ の事後確率 $p(\theta\vert x)$ を推論するために，事前証拠と観測が使われる過程を意味する</font>。
パラメータ推定のための最も一般的なアプローチの 1 つは ML である。
このアプローチによると，ML  推定値は次のように得られる:
<!-- Assume that x are the observations and θ the unknown parameters of a model that generated x.
In this article, the term estimation will be used strictly to refer to parameters and inference to refer to random variables.
The term estimation refers to the calculated approximation of the value of a parameter from incomplete, uncertain and noisy data.
In contrast, the term inference will be used to imply Bayesian inference and refers to the process in which prior evidence and observations are used to infer the posterior probability p(θ|x) of the random variables θ given the observations x.
One of the most popular approaches for parameter estimation is ML. According to this approach, the ML estimate is obtained as -->
$$
\hat{\theta}_{ML}=\arg\max_{\theta} p(x;\theta)\tag{tz1}
$$

ここで，$p(x;\theta)$  は，観測値 $x$ を生成した仮定モデルに基づく観測値とパラメータとの間の確率的関係を記述する。
このとき，$p(x;\theta)$  と $p(x\vert\theta)$ の表記の違いを明確にしたい。
$p(x;\theta)$ と書くと $\theta$ がパラメータであることを意味し，$\theta$ の関数として尤度関数と呼ばれる。
これに対して $p(x\vert\theta)$ と書くと，$\theta$ が確率変数であることを意味する。
<!-- where p(x;θ) describes the probabilistic relationship between the observations and the parameters based on the assumed model that generated the observations x.
At this point, we would like to clarify the difference between the notation p(x; θ) and p(x|θ).
When we write p(x; θ) we imply that θ are parameters and as a function of θ is called the likelihood function.
In contrast, when we write p(x; θ), we imply that θ are random variables.  -->

尤度関数 $p(x;\theta)$ の直接評価は複雑で，直接計算することも最適化することも難しいか不可能な場合が多く，興味深い。
このような場合，隠れ変数 $z$ を導入することで，この尤度の計算が非常に容易になる。
これらの確率変数は，Bayes 則によって観測値を未知のパラメータに接続するリンクとして機能する。
隠れ変数の選択は問題に依存する。
しかし，その名前が示すように，これらの変数は観測されず，条件付き確率 $p(x|z)$ が計算しやすいように，観測に関する十分な情報を提供する。
この役割とは別に，隠れ変数は統計的モデリングにおいて別の役割を果たす。
それは，観測値を生成したと仮定される確率的機構の重要な部分であり，「グラフィカルモデル」と呼ばれるグラフによって簡潔に記述することができる。
グラフィカルモデルの詳細については 「グラフィカル・モデル」の項で説明する。
<!-- In many cases of interest direct assessment of the likelihood function p(x;θ) is complex and is either difficult or impossible to compute it directly or optimize it.
In such cases the computation of this likelihood is greatly facilitated by the introduction of hidden variables z.
These random variables act as links that connect the observations to the unknown parameters via Bayes’ law.
The choice of hidden variables is problem dependent.
However, as their name suggests, these variables are not observed and they provide enough information about the observations so that the conditional probability p(x|z) is easy to compute.
Apart from this role, hidden variables play another role in statistical modeling.
They are an important part of the probabilistic mechanism that is assumed to have generated the observations and can be described very succinctly by a graph that is termed “graphical model.”
More details on graphical models is given in the section “Graphic Models.” -->

隠れ変数とその事前確率 $p(z;\theta)$ が導入されると，以下のように隠れ変数を積分 (周辺化) することによって，尤度または周辺尤度 (時折呼ばれる) を得ることができる。
<!-- Once hidden variables and a prior probability for them p(z; θ) have been introduced, one can obtain the likelihood or the marginal likelihood as it is called at times by integrating out (marginalizing) the hidden variables according to  -->

$$\tag{tz2}
p(x;\theta) = \int p(x;z,\theta)\,dz = \int p(x\vert z,\theta) p(z;\theta)\,dz.
$$

この一見単純な積分が Bayes 的手法の肝であり，この方法で尤度関数とベイズの定理による隠れ変数の事後分布を得ることができるからである。
<!-- This seemingly simple integration is the crux of the Bayesian methodology because in this manner we can obtain both the likelihood function, and by using Bayes’ theorem, the posterior of the hidden variables according to -->

$$
p(z\vert x,\theta)=\frac{p(x\vert z,\theta)\, p(z;\theta)}{p(x;\theta)}
\tag{tz3} $$

事後情報が得られれば，隠れ変数について上述のごとき推論も可能である。
上記の定式化は簡単であるが，ほとんどの場合 式 (2) の積分は閉形式で計算することが不可能か非常に困難である。
したがって，Bayes 推定では，この積分を迂回または近似的に評価する技術に主な取り組みが集中している。
<!-- Once the posterior is available, inference as explained above for the hidden variables is also possible.
Despite the simplicity of the above formulation, in most cases of interest the integral in (2) is either impossible or very difficult to compute in closed form.
Thus, the main effort in Bayesian Inference is concentrated on techniques that allow us to bypass or approximately evaluate this integral. -->

このような方法は，大きく 2 つに分類される。
1 つはモンテカルロ法とも呼ばれる数値サンプリング法，もう 1 つは決定論的近似法である。
この記事では，モンテカルロ法については一切触れない。
このような手法に興味のある読者は，例えば [11] や [12] のようなこのトピックに関する多くの書籍やサーベイ記事を参照されたい。
さらに ML 法を拡張した最大事後推定 (MAP) は，非常に粗いベイズ近似と考えることができる。
`Maximum A Posteriori: Poor Man's Bayesian Inference` 参照。
<!-- Such methods can be classified into two broad categories.
The first is numerical sampling methods also known as Monte Carlo techniques and the second is deterministic approximations.
This article will not address at all Monte Carlo methods.
The interested reader for such methods is referred to a number of books and survey articles on this topic, for example [11] and [12].
Furthermore, maximum posteriori (MAP) inference, which is an extension of the ML approach, can be considered as a very crude Bayesian approximation, see “Maximum A Posteriori: Poor Man’s Bayesian Inference.”-->

以下に示すように，EM アルゴリズムは，事後確率 $p(z\vert x;\theta)$ の知識を前提とし，明示的に計算することなく尤度関数を反復的に最大化するベイズ推論手法である。
この方法論の重大な欠点は，多くの興味深い事例においてこの事後情報が得られないことである。
しかし，最近の Bayes 推論の発展により，事後分布を近似することでこの難点を回避することができるようになった。
これらは「変分 Bayes」と呼ばれ，このチュートリアルの焦点となるものである。
<!-- As it will be shown in what follows, the EM algorithm is a Bayesian inference methodology that assumes knowledge of the posterior p(z|x; θ) and iteratively maximizes the likelihood function without explicitly computing it.
A serious shortcoming of this methodology is that in many cases of interest this posterior is not available.
However, recent developments in Bayesian inference allow us to bypass this difficulty by approximating the posterior.
They are termed “variational Bayesian” and they will be the focus of this tutorial.-->

<!-- <div class="code" style="background-color:#f7f6f1;"> -->
<div style="width:88%;background-color:#f7f6f1;">

### A. 最大事後確率法：貧乏人の Bayes 推論<!-- ### MAXIMUM A POSTERIORI: POOR MAN’S BAYESIAN INFERENCE-->

統計的信号処理の文献で最もよく使われる手法の 1 つが最大事後確率法 (MAP: Maximum A Posteriori 法) である。
MAP は，パラメータベクトル $\theta$ を確率変数と仮定し，$\theta$ に事前分布 $p(\theta)$ を課すことから，しばしばベイズ的と呼ばれる。
ここでは，MAP 推定とベイズ推定との類似点と相違点を明らかにする。
$x$ を観測値，$\theta$ を未知量とすると，MAP 推定は次のように定義される。
<!-- One of the most commonly used methodologies in the statistical signal processing literature is the maximum a posteriori (MAP) method.
MAP is often referred to as Bayesian, since the parameter vector θ is assumed to be a random variable and a prior distribution pθ is imposed on θ.
In this appendix, we would like to illuminate the similarities and differences between MAP estimation and Bayesian inference.
For x the observation and θ an unknown quantity the MAP estimate is defined as -->

$$ \hat{\theta}_ {\text{MAP}} = \arg\max_ {\theta} p(\theta\vert x)\tag{A.1} $$

Bayes 則を用いて， MAP 推定は次式より得られる:
<!-- Using Bayes’ theorem, the MAP estimate can be obtained from  -->

$$
\hat{\theta}_{\text{MAP}} = \arg\max_{\theta} p(x|\theta)\,d\theta
\tag{A.2}$$

ここで $p(x|\theta)$ は観測値の尤度である。
MAP 推定値は (A.1) よりも (A.2) から求めた方が簡単である。
ベイズの定理に基づく (A.1) の事後確率は次式で与えられる:
<!-- where p(x|θ) is the likelihood of the observations.
The MAP estimate is easier to obtain from (A.2) than (A.1).
The posterior in (A.1) based on Bayes’ theorem is given by -->

$$ p(\theta\vert x) = \frac{p(x\vert\theta)p(\theta)}{\int p(x\vert\theta)p(\theta) d\theta}\tag{A.3} $$

であり 式 (A.3) の分母にベイズ積分を計算し，$\theta$ を周辺化する必要がある。
<!-- and requires the computation of the Bayesian integral in the denominator of (A.3) to marginalize θ.-->

以上より，MAP 推定量も Bayes 推定量も $\theta$ を確率変数と仮定し，Bayes の定理を用いることは明らかであるが，その類似性はそこに止まっている。
ベイズ推定では事後分布を用いるため，$\theta$ を周辺化する必要がある。
これに対して，MAP では事後値の最頻値を用いる。
ベイズ推定は，MAP と異なり，$\theta$ に関する利用可能なすべての情報を平均化すると言える。
したがって MAP は「貧乏人の Bayes 推論」と言える。
<!-- From the above, it is clear that both MAP and Bayesian estimators assume that θ is a random variable and use Bayes’ theorem, however, their similarity stops there.
For Bayesian inference, the posterior is used and thus θ has to be marginalized.
In contrast, for MAP the mode of the posterior is used.
One can say that Bayesian inference, unlike MAP, averages over all the available information about θ.
Thus, it can be stated that MAP is more like “poor man’s” Bayesian inference. -->

EM は $\theta$ の MAP 推定値も得るために用いることができる。
Bayes の定理を用いて，次のように書くことができる:
<!-- The EM can be used to also obtain MAP estimates of θ.
Using  Bayes’ theorem we can write -->

$$ \begin{aligned}
\ln p(\theta\vert x) & = \ln p(x,\theta) − \ln p(x)\\
&= \ln p(x\vert\theta) + \ln p(\theta) − \ln p(x).
\end{aligned}\tag{A.4} $$

「EM アルゴリズムの別見解」節 の ML-EM の場合と同様の枠組みで，次のように書くことができる:
<!-- Using a similar framework as for the ML-EM case in the section “An Alternative View of the EM Algorithm,” we can write-->

$$\begin{aligned}
\ln p(\theta\vert x) &= F(q,\theta) + D_ {\text{KL}}(q\vert\vert p) + \ln p(\theta) − \ln p(x)\\
&\ge F(q,\theta) + \ln p(\theta) − \ln p(x),
\end{aligned}\tag{A.5}$$

ここで，$\ln p(x)$ は定数である。
(A.5) の右辺は，EM アルゴリズムと同様に交互に最大化することができる。
$q(z)$  に関して最適化すると，先に説明した ML の場合と同じ E-step が得られる。
$\theta$ に関して最適化すると，目的関数が $\ln p(\theta)$  の項を含むので，異なるM-step が得られる．
一般に MAP-EM アルゴリズムの M-ステップは，ML の場合よりも複雑であり，例えば  [30]  や [31] を参照されたい。
厳密に言えば，このようなモデルでは，MAP  推定は $\theta$ 確率変数のみに使用され，ベイズ推定は隠れ変数 $z$ に使用される。
<!-- where in this context ln p(x) is a constant.
The right-hand side of (A.5) can be maximized in an alternating fashion as in the EM algorithm.
Optimization with respect to q(z) gives an identical E-step as for the ML case previously explained.
Optimization with respect to θ gives a different M-step since the objective function now contains also the term ln p(θ).
In general, the M-step for the MAP-EM algorithm is more complex than in its ML counterpart, see for example [30] and [31].
Strictly speaking, in such a model MAP estimation is used only for the θ random variables, while Bayesian inference is used for hidden variables z.-->
</div>

# 3. グラフィカルモデル (p4L)
<!-- # GRAPHICAL MODELS-->

グラフィカルモデルは，統計モデリング問題の確率変数間の依存関係を表現するための枠組みを提供し，確率系に関わる項目間の相互作用をグラフィカルに表現する包括的でエレガントな方法を構成する。
グラフィカルモデルは，問題の確率変数に対応するノードと，変数間の依存関係を表すエッジからなるグラフである。
グラフのノード A からノード B への有向エッジは，変数 B が変数 A の値に確率的に依存することを表す。
グラフィカルモデルには有向グラフと無向グラフとがある。
後者の場合，マルコフ確率場 (13,14,15) とも呼ばれる。
このチュートリアルの残りの部分では，ベイジアンネットワークとも呼ばれる有向グラフィカルモデルに焦点を当てる。
さらに，有向グラフは無サイクル (すなわちサイクルを含まない) であると仮定する。
<!-- Graphical models provide a framework for representing dependencies among the random variables of a statistical modelling problem and they constitute a comprehensive and elegant way to graphically represent the interaction among the entities involved in a probabilistic system.
A graphical model is a graph whose nodes correspond to the random variables of a problem and the edges represent the dependencies among the variables.
A directed edge from a node A to a node B in the graph denotes that the variable B stochastically depends on the value of the variable A.
Graphical models can be either directed or undirected.
In the second case they are also known as Markov random fields [13],[14],[15].
In the rest of this tutorial, we will focus on directed graphical models also called Bayesian Networks, where all the edges are considered to have a direction from parent to child denoting the conditional dependency among the corresponding random variables. In addition we assume that the directed graph is acyclic (i.e., contains no cycles).-->

$G=(V, E)$ を有向無サイクルグラフとし，$V$ をノードの集合，
$E$ を有向エッジの集合とする。
また $x_ {s}$ はノード $s$ に関連する確率変数，$\pi(s)$ はノード $s$ の親の集合を表すとする。
また，各ノード $s$ には，その親変数の値が与えられたときの $x_ {s}$ の分布を定義する条件付き確率密度 $p(x_ {s}\vert x_ {\pi(s)})$  が関連付けられている。
したがって，グラフモデルを完全に定義するためには，グラフ構造とは別に，各ノードにおける条件付き確率分布も指定する必要がある。
これらの分布がわかれば，全変数の集合に対する結合分布は積として計算することができる。
<!-- Let G = (V, E) be a directed acyclic graph with V being the set of nodes and E the set of directed edges.
Let also xs denote the random variable associated with node s and π(s) the set of parents of node s.
Associated with each node s is also a conditional probability density p(xs|xπ(s)) that defines the distribution of xs given the values of its parent variables.
Therefore, for a graphical model to be completely defined, apart from the graph structure, the conditional probability distribution at each node should also be specified. Once these distributions
are known, the joint distribution over the set of all variables can be computed as the product:-->

$$ p(x)=\prod_{s} p(x_{s}\vert x_{\pi(s)}).\tag{tz4} $$

上式は，有向グラフモデル [13] を，上式で指定された方法で因数分解する確率分布の集まりとして正式に定義したものである (もちろん，これは基礎となるグラフの構造に依存する)。
<!-- The above equation constitutes a formal definition of a directed graphical model [13] as a collection of probability distributions that factorize in the way specified in the above equation (which of course depends on the structure of the underlying graph).-->

図 2 に有向グラフモデルの一例を示す。
ノードに描かれた確率変数は $a,b,c,d$ である。
各ノードは条件付き確率密度を表し，その親からのノードの依存性を定量化する。
ノードの密度は正確にはわからないかもしれないが，パラメータ $\theta_ {i}$ の集合によってパラメータ化することができる。
確率の連鎖則を用いて，結合分布は以下のように書ける:
<!-- In Figure 2 we show an example of a directed graphical model.
The random variables depicted at the nodes are a, b, c, and d.
Each node represents a conditional probability density that quantifies the dependency of the node from its parents.
The densities at the nodes might not be exactly known and can be parameterized by a set of parameters θi.
Using the chain rule of probability we would write the joint distribution as:-->
$$
p(a,b,c,d;\theta) = P(a;\theta_{1}) p(b\vert a;\theta_{2}) p(c\vert a,b;\theta_{3}) p(d\vert a,b,c;\theta_{4})\tag{tz5}
$$

<div class="figure figcenter">
<img src="../figures/2008Tzikas_fig2.svg" width="19%">
<div class="figcaption">

図 2．有向グラフモデルの例。
丸で示されたノードは確率変数に対応し，四角で示されたノードはモデルのパラメータに対応する。
二重丸のノードは観測された確率変数を表し，一重丸のノードは隠れた確率変数を表す。
<!-- FIG 2. Example of directed graphical model.
Nodes denoted with circles correspond to random variables, while nodes denoted with squares correspond to parameters of the model.
Doubly circled nodes represent observed random variables, while single circled nodes represent hidden random variables. -->
</div></div>

しかし，グラフ構造が意味する独立性を考慮することで，この式を簡略化することができる。
一般に，グラフィカルモデルでは，各ノードはその親を与えられた先祖から独立している。
つまり，ノード $c$ はノード $b$ があればノード $a$ に依存しないし，ノード $d$ はノード $b$ と $c$ があれば $a$ に依存しない。
したがって (tz4) 式からは次のように書くことができる:
<!-- However, we can simplify this expression by taking into account the independencies that the graph structure implies.
In general, in a graphical model each node is independent of its ancestors given its parents.
This means that node c does not depend on node a given node b, and node d does not depend on a given nodes b and c.
Thus, from (4) we can write:-->

$$ p(a,b,c,d;\theta) = p(a;\theta_{1}) p(b\vert a;\theta_{2}) p(c\vert a;\theta_{3}) p(d\vert b,c;\theta_{4}).\tag{tz6} $$

グラフィカルモデリングで生じるもう一つの有用な特徴は，通常データセットと呼ばれるいくつかの観測値が存在する場合，確率変数は観測値が存在する観測型 (または可視型) と直接観測値が利用できない非観測型に区別されることである。
観測されたデータは，グラフィカルモデル構造によって記述される生成機構によって生成されると仮定することが有用であり，それは中間サンプリングおよび計算ステップとしての隠れ変数を含む。
また，グラフィカルモデルにはパラメトリックとノンパラメトリックがあることに注意しなければならない。
パラメトリックモデルであれば，グラフの一部のノードで条件付確率分布にパラメータが現れる，すなわち，これらの分布はパラメータ化された確率モデルである。
<!-- Another useful characterization arising in graphical modeling is that in the presence of some observations, usually called dataset, the random variables can be distinguished as observed (or visible) for which there exist observations and hidden for which direct observations are not available.
A useful consideration is to assume that the observed data are produced through a generation mechanism, described by the graphical model structure, which involves the hidden variables as intermediate sampling and computational steps.
It must also be noted that a graphical model can be either parametric or nonparametric.
If the model is parametric, the parameters appear in the conditional probability distributions at some of the graph nodes, i.e., these distributions are parameterized probabilistic models.-->

グラフィカルモデルが完全に決定されると (すなわち，すべてのパラメータが指定されると)，確率変数の部分集合の周辺分布の計算，残りの変数の値を与えられた変数の部分集合の条件付き分布の計算，いくつかの以前の密度における最大点の計算など，いくつかの推論問題を定義することができる。
グラフィカルモデルがパラメトリックである場合，ある観測データセットが与えられたときに，パラメータの適切な値を学習する問題がある。
通常，パラメータ学習の過程では，いくつかの推論段階が含まれる。
<!-- Once a graphical model is completely determined (i.e., all parameters have been specified), then several inference problems could be defined such as computing the marginal distribution of a subset of random variables, computing the conditional distribution of a subset of variables given the values of the rest variables and computing the maximum point in some of the previous densities.
In the case where the graphical model is parametric, then we have the problem of learning appropriate values of the parameters given some dataset with observations.
Usually, in the process of parameter learning, several inference steps are involved.-->

# 4. EM アルゴリズムの別見解 (p5L)
<!-- # AN ALTERNATIVE VIEW OF THE EM ALGORITHM -->

本稿では [16] と [13] の EM の解説に従うことにする。
対数尤度は次のように書けることは簡単である。
<!-- In this article, we will follow the exposition of the EM in [16] and [13].
It is straightforward to show that the log-likelihood can be written as -->

* [13] C. Bishop, Pattern Recognition and Machine Learning. New York: Springer-Verlag, 2006.
* [16] R.M. Neal and G.E. Hinton, “A view of the EM algorithm that justifies incremental, sparse and other variants,” in Learning in Graphical Models, M.I. Jordan, Ed. Cambridge, MA: MIT Press, 1998, pp. 355–368.

$$
\ln p(x;\theta) = F(q,\theta)+ D_ {\text{KL}}(q\vert\vert p)\tag{tz7}
$$
ここで<!--with-->
$$
F(q,\theta) = \int q(z) \ln\left(\frac{p(x,z;\theta)}{q(z)}\right)\,dz\tag{tz8}
$$
であり，かつ，<!-- and -->
$$
D_{\text{KL}}(q\vert\vert p) = − \int q(z)\ln\left(\frac{p(z|x;\theta)}{q(z)}\right)\,dz\tag{tz9}
$$

ここで $q(z)$ は任意の確率密度関数である。
$D_{\text{KL}}(q||p)$ は  $p(z|x;\theta)$ と $q(z)$ の間の カルバック・ライブラー 情報量 (Kullback-Leibler divergence; あるいは相互情報量) である。
$D_{\text{KL}} (q||p) \ge 0$ なので $\ln p(x;\theta)\ge F(q,\theta)$ が成立する。
つまり $F(q,\theta)$ は対数尤度の下界である。
等式が成り立つのは $D_{KL}(q||p) = 0$ のときだけである。
これは $p(z|x;\theta)=q(z)$ を意味する。
EM アルゴリズムやベイズ推定の決定論的近似における最近のいくつかの進歩は，密度 $q$ とパラメータ $\theta$ に関する下界 $F(q,\theta)$ の最大化として (7) 式の分解に照らして見ることができる。
<!-- where q(z) is any probability density function. KL(q p) is the Kullback-Leibler divergence between p(z|x; θ) and q(z), and since KL(q p) ≥ 0, it holds that ln p(x; θ) ≥ F(q, θ).
In other words, F(q, θ) is a lower bound of the log-likelihood.
Equality holds only when KL(q p) = 0, which implies p(z|x; θ) = q(z).
The EM algorithm and some recent advances in deterministic approximations for Bayesian inference can be viewed in the light of the decomposition in (7) as the maximization of the lower bound F(q, θ) with respect to the density q and the parameters θ.-->

特に EM は下界 $F(q,\theta)$ を最大化する 2 段階の反復アルゴリズムである。
したがって EM アルゴリズムは，モデルの対数尤度を最大化する。
パラメータの現在値を $θ^{\text{OLD}}$ とする。
E-ステップでは，下界 $F(q, \theta^{\text{OLD}}$ は $q(z)$ に関して最大化される。
これは $D_ {\text{KL}}(q\vert\vert p) = 0$ のとき，換言すれば $q(z) = p(z\vert x;\theta^{\text{OLD}})$ のときに起こることは容易に理解できる。
この場合，下界は対数尤度に等しくなる。
続く M-step では $q(z)$ は固定され，下界 $F(q,\theta)$ は $\theta$ に関して最大化されて，ある新しい値 $\theta^{\text{NEW}}$ を与える。
これにより，下界は増加し，その結果，対応する対数尤度も増加する。
$q(z)$ は $\theta^{\text{OLD}}$ を用いて決定され，M ステップで固定されるため，新しい事後確率 $p(z|x;\theta^{\text{NEW}})$ と等しくはならず，したがって KL ダイバージェンスはゼロにはならない。
従って，対数尤度の増加は下界の増加より大きい。
$q(z) = p(z|x;\theta^{OLD})$ を下界に代入し，式 (8) を展開すると次のようになる。
<!-- In particular, the EM is a two step iterative algorithm that maximizes the lower bound F(q,θ) and hence the log-likelihood.
Assume that the current value of the parameters is θOLD.
In the E-step the lower bound F(q, θOLD) is maximized with respect to q(z). It is easy to see that this happens when KL(q p) = 0, in other words, when q(z) = p(z|x; θOLD).
In this case the lower bound is equal to the log-likelihood.
In the subsequent M-step, q(z) is held fixed and the lower bound F(q,θ) is maximized with respect to θ to give some new value θNEW.
This will cause the lower bound to increase and as a result, the corresponding log-likelihood will also increase.
Because q(z) was determined using θOLD and is held fixed in the M-step, it will not be equal to the new posterior p(z|x; θNEW) and hence the KL distance will not be zero.
Thus, the increase in the log-likelihood is greater than the increase in the lower bound. If we substitute q(z) = p(z|x; θOLD) into the lower bound and expand (8) we get-->

$$\begin{aligned}
F(q,\theta) &= \int p(z\vert x;\theta^{\text{OLD}}) \ln p(x,z;\theta) dz - \int p(z\vert x;\theta^{\text{OLD}}) dz\\
&= Q(\theta,\theta^{\text{OLD}})+\text{定数}\\
\end{aligned}\tag{tz10}$$

ここで定数は $\theta$ に依存しない $p(z\vert x;\theta^{\text{OLD}})$ のエントロピーを単純に表したものである。
また，以下の関数は M-ステップで最大化される完全データ (観測値＋隠れ変数) の対数尤度の期待値である:
<!-- where the constant is simply the entropy of p(z|x; θOLD) which does not depend on θ.
The function-->

$$\begin{aligned}
Q(\theta,\theta^{\text{OLD}}) &= \int p(z\vert x;\theta^{\text{OLD}})\ln p(x,z;\theta)\,dz\\
&= \left<\ln p(x,z;\theta\right>_{p(z\vert x;\theta^{\text{OLD}})}\\
\end{aligned}\tag{tz:11}$$

信号処理の文献で EM アルゴリズムを示す通常の方法は $Q(\theta,\theta^{\text{OLD}})$ 関数を直接使用することである。
([9] と [17] を参照)。
<!-- is the expectation of the log-likelihood of the complete data (observations + hidden variables) which is maximized in the M-step.
The usual way of presenting the EM algorithm in the signal processing literature has been via use of the Q(θ, θOLD) function directly, see for example [9] and [17].-->

要約すると EM アルゴリズムは，以下の 2 段階を含む反復算法である。
<!-- In summary, the EM algorithm is an iterative algorithm involving the following two steps:-->

$$\begin{aligned}
\text{E-step: Compute  }  & P(z\vert x;\theta^{\text{OLD}})\\
\text{M-step: Evaluate  } & \theta^{\text{NEW}}=\arg\max Q(\theta,\theta^{\text{OLD}})\\
\end{aligned}\tag{tz12,tz13}$$

さらに EM アルゴリズムでは，$p(z|x;\theta)$ が明示的に分かっているか，少なくともその十分統計量 $\ln p(z|x;\theta) p(z|x;\theta^{\text{OLD}})$ の条件付き期待値を計算できる必要があることを指摘したい (11 参照)。
換言すれば，EM アルゴリズムを使用するためには，オブザベーションが与えられたときの隠れ変数の条件付き pdf を知らなければならない．
$p(z|x;\theta)$ は一般的に $p(x;\theta)$ よりもずっと推論しやすいが，多くの興味深い問題ではこれは不可能で，したがって EM アルゴリズムが適用できない。
<!-- Furthermore, we would like to point out that the EM algorithm requires that p(z|x; θ) is explicitly known, or at least we should be able to compute the conditional expectation of its sufficient statistics  ln p(z|x; θ)  p(z|x;θOLD), see (11).
In other words, we have to know the conditional pdf of the hidden variables given the observations in order to use the EM algorithm.
While p(z|x; θ) is in general much easier to infer than p(x; θ), in many interesting problems this is not possible and thus the EM algorithm is not applicable.-->

# 5. 変分 EM の枠組み (p5R)
<!-- # THE VARIATIONAL EM FRAMEWORK-->

式 (eq:tz7) の分解において適切な $q(z)$ を仮定することで $p(z|x;\theta)$ を正確に知るという要件を回避することができる。
E ステップでは $\theta$ を固定したまま $F(q,\theta)$ を最大化するような $q(z)$ を見つける。
この最大化を行うためには $q(z)$ の特定の形式を仮定しなければならない。
特定のケースでは $q(z;\omega)$ の形式の知識を仮定することが可能です ($\omega$ はパラメータのセット)。
したがって，下界 $F(\omega,\theta)$ はこれらのパラメータの関数となり，E-step では $\omega$ に関して，M-step では $\theta$ に関して最大化される。
<!-- One can bypass the requirement of exactly knowing p(z|x; θ) by assuming an appropriate q(z) in the decomposition of (7).
In the E-step q(z) is found such that it maximizes F(q, θ) keeping θ fixed.
To perform this maximization, a particular form of q(z) must be assumed.
In certain cases it is possible to assume knowledge of the form of q(z;ω), where ω is a set of parameters.
Thus, the lower bound F(ω, θ) becomes a function of these parameters and is maximized with respect to ω in the E-step and with respect to θ in the M-step, see for example [13]. -->

しかし，一般的な形式では，下界 $F(q,\theta)$ は $q$ についての **汎関数** である。
換言すれば $q(z)$ の関数を入力として受け取り，関数の値を出力として返す写像である。
これは自然に汎関数の微分の概念につながり，関数の微分と同様，入力関数の無限小の変化に対する関数の変化を与える。
この分野の数学は **変分計算** と呼ばれ，流体力学，熱伝導，制御理論など，数学，物理科学，工学の多くの分野に応用されている。
<!--However, in its general form the lower bound F(q, θ) is a functional in terms of q, in other words, a mapping that takes as input a function q(z), and returns as output the value of the functional.
This leads naturally to the concept of the functional derivative, which in analogy to the function derivative, gives the functional changes for infinitesimal changes to the input function.
This area of mathematics is called calculus of variations [18] and has been applied to many areas of mathematics, physical sciences and engineering, for example fluid mechanics, heat transfer, and control theory.-->

変分理論には近似値はない。
だが，変分法はベイズ推論問題の近似解を求めるのに使用できる。
これは，最適化が行われる関数が特定の形を持っていると仮定することによって行われる。
例えば 2 次関数や，固定基底関数の線形結合である関数のみを仮定することができる。
ベイジアン推論において，大きな成功を収めている特定の形式は因子化された形式である ([19] と [20] を参照)。
この因子化近似のアイデアは，理論物理学に由来しており，それは **平均場理論 mean field theory** と呼ばれている。
<!-- Although there are no approximations in the variational theory, variational methods can be used to find approximate solutions in Bayesian inference problems.
This is done by assuming that the functions over which optimization is performed have specific forms.
For example, we can assume only quadratic functions or functions that are linear combinations of fixed basis functions.
For Bayesian inference, a particular form that has been used with great success is the factorized one, see [19] and [20].
The idea for this factorized approximation stems from theoretical physics where it is called mean field theory [21].-->

この近似によれば，隠れ変数 $z$ は $i=1,\ldots,M$ で $M$ 個の分割 $z_{i}$ に分割されると仮定される。
また $q(z)$ はこれらの分割に関して次のように因数分解すると仮定する。
<!-- According to this approximation, the hidden variables z are assumed to be partitioned into M partitions zi with i = 1,...,M.
Also it is assumed that q(z) factorizes with respect to these partitions as-->

したがって，下界 $F(q,\theta)$ を最大化する (tz14) の形の $q(z)$ を求めたい。
(tz14) を用いて，簡単のために $q_{j} (z_{j})=q_{j}$ とすると，次のようになる。
<!-- Thus, we wish to find the q(z) of the form of (14) that maximizes the lower bound F(q,θ).
Using (14) and denoting for simplicity qj (zj)=qj we have-->

$$
q(z)=\prod_ {i=1}^{M} q_ {i}(z_ {i}).\tag{tz14}
$$

したがって，下界 $F(q,\theta)$ を最大化する 式(14) の形の $q(z)$ を求めたい。
式 (14) を用いて，簡単のために $q_{j}(z_{j})=q_{j}$ とすると，次のようになる。
<!-- Thus, we wish to find the q(z) of the form of (14) that maximizes the lower bound F(q,θ).
Using (14) and denoting for simplicity qj(zj)=qj we have-->

$$\begin{aligned}
F(q,\theta) &= \int \prod_{i} q_ {i}\left[\ln q(x,z;\theta) -\sum_{i}\ln q_{i}\right]dz\\
&=\int \prod_{i} q_ {i}\ln p(x,z;\theta)\prod_{i} dz_{i} - \sum_{i}\int\prod_{j} q_{j}\ln q_{i} dz_{i}\\
&=\int q_{j}\left[\ln p(x, z;\theta) \prod_{i\ne j}(q_{i}, dz_{i})\right] dz_{j} -\int q_{j}\ln q_{j} dz_{j} -\sum_{i\ne j}\int q_{i} \ln q_{i} dz_{i}\\
&= \int q_ {j}\ln \hat{p}(x,z_ {j};\theta) dz_ {j} - \int q_{j} \ln q_ {j}\, dz_ {j} -\sum\int q_ {i}\ln q_ {i} dz_ {i}\\
&=D_{\text{KL}}(q_{i}\vert\vert \hat{p})-\sum_{i\ne j} \int q_{i}\ln q_{i}dz_{i}\\
\end{aligned}\tag{tz15}$$
ここで，
$$
\ln \hat{p}(x,z_ {j};\theta)=\left<\ln p(x,z;\theta)\right>_ {i\ne j}=\int \ln p(x,z;\theta)\prod_ {i\ne j}(q_ {i},dz_ {i}).
$$

明らかに 式 (15) の境界は，Kullback-Leibler 距離が 0 になったときに最大となり，それは $q_{j}(z_ {j})=\hat{p}(x,z_{j}; \theta)$ の場合である。
換言すれば最適分布 $q^ {∗}_{j}(z_{j})$ の式は以下の通りである:
<!-- Clearly the bound in (15) is maximized when the Kullback-Leibler distance becomes zero, which is the case for $q_ {j}(z_ {j})=\hat{p}(x,z_ {j};\theta)$, in other words the expression for the optimal distribution $q^ {∗}_ {j}(z_ {j})$ is:-->

$$
\ln q_{j}^ {\star} (z_{j})=\left< p(x,z;\theta)\right>_{i\ne j}+\text{const}.\tag{tz16}
$$

式 (16) の加法定数は正規化によって求めることができるので
<!-- The additive constant in (16) can be obtained through normalization, thus we have   -->

$$
q^{\star}_{z_{j}}=\frac{\exp^{\left< \ln p(x,z;\theta)\right>_{i\ne j}}}{\int \exp^{\left< \ln p(x,z;\theta)\right>_{i\ne j}}dz_{j}}.\tag{tz17}
$$

$j=1,\ldots,M$ に対する上式は，式 (14) の因数分解を前提とした下界の最大値に対する整合性条件の集合である。
これらは $i\ne j$ の他の因子 $q_{i}(z_{i})$ に依存するため，明示的な解を提供しない。
したがって，これらの因子を循環させ，それぞれを修正された推定値で順番に置き換えることにより，一貫した解を求めることができる。
<!-- The above equations for j=1,...,M are a set of consistency conditions for the maximum of the lower bound subject to the factorization of (14).
They do not provide an explicit solution since they depend on the other factors $q_ {i}(z_ {i})$ for $i\ne j$.
Therefore, a consistent solution is found by cycling through these factors and replacing each in turn with the revised estimate.-->

要約すると，変分 EM アルゴリズムは，以下の 2 つのステップで与えられる。
<!-- In summary, the variational EM algorithm is given by the following two steps:-->

* 変分 E-ステップ: 式 (17) を解いて，$F(q,\theta^{\text{OLD}})$ を最大化するために，$q^{\text{NEW}}(z)$ を評価する
* 変分 M-ステップ: $\theta^{\text{NEW}} = \arg\max F(q^{\text{NEW}},\theta)$ を探索する

<!-- & \text{ to maximize $F(q,\theta^{\text{OLD}})$ solving the system of (17).\\ -->

このとき，ベイズモデルが隠れ変数のみを含み，パラメータを含まない場合があることに注目する必要がある。
そのような場合，変分 EM アルゴリズムは，式 (17) を用いて $q(z)$ を得る E-ステップのみを有する。
この 関数 $q(z)$ は，隠れ変数の推論に使用できる $p(z|x)$ の近似を構成する。
<!-- At this point it is worth noting that in certain cases a Bayesian model can contain only hidden variables and no parameters.
In such cases the variational EM algorithm has only an E-step in which q(z) is obtained using (17).
This function q(z) constitutes an approximation to p(z|x) that can be used for inference of the hidden variables.-->

# 6. 線形回帰 Linear Regression (p6L)

本節では，線形回帰問題を例にして，前節までのベイズ推論の手法を示す。
線形回帰が選ばれた理由は，単純であり，入門的な例として優れているためである。
さらに，線形回帰は，デコンボリューション，チャネル推定，音声認識，周波数推定，時系列予測，システム同定に至るまで，多くの信号処理アプリケーションで発生する。
<!-- In this section, we will use the linear regression problem as an example to demonstrate the Bayesian inference methods of the previous sections.
Linear regression was selected because it is simple and constitues an excellent introductory example.
Furthermore, it occurs in many signal processing applications ranging from deconvolution, channel estimation, speech recognition, frequency estimation, time series prediction, and system identification.-->

この問題では，未知信号 $y(x) \in\mathbb{R}, x\in\Omega\subseteq \mathbb{R}^{N}$ を考え，任意の位置 $x_{\star}\in\Omega$ におけるその値 $t_{\star}=y(x_{\star})$ を予測したいとする。
$n=1,\ldots,N$, $x=(x_{1},\ldots,t_{N})$,$x_{n}\in\Omega$, $n=1,\ldots,N$ の位置で$t_{n}=y(x_{n}) +\epsilon_{n}$ の雑音付き観測値を $\mathbf{t}=(T_{1},\ldots,T_{N})$ ベクトルとすると，$n$ の雑音がある。
加法性雑音 $\epsilon_{n}$ は一般に独立，平均ゼロのガウス分布と仮定する。
<!-- For this problem, we consider an unknown signal $y(x)\in\mathbb{R}, x\in\Omega\subseteq \mathbb{R}^N$ and want to predict its value $t_ {*}=y(x_ {*})$ at an arbitrary location $x_ {*}\in\Omega$, using a vector $\mathbf{t}=(t_ {1},\ldots,t_ {N})$ of $N$ noisy observations $t_ {n}=y(x_ {n}) +\epsilon_ {n}$, at locations $x=(x_{1},\ldots,t_ {N})$, $x_ {n}\in\Omega$, $n=1,\ldots,N$.
The additive noise εn is commonly assumed to be independent, zeromean, Gaussian distributed:-->

$$
p(\epsilon) = \mathcal{N}(\epsilon\vert 0,\beta^{-1}\mathbf{I}),\tag{tz18}
$$

ここで，$\beta$ は，分散行列の逆行列 $\epsilon=(\epsilon_{1},\ldots,\epsilon_{N})^{\top}$ である。
<!-- where $\beta$ is the inverse variance and $\epsilon=(\epsilon_ {1},\ldots,\epsilon_ {N})^{\top}$ -->

信号 $y$ は一般に $M$ 個の基底関数 $\phi_ {m}(x)$ の線形結合としてモデル化される。
<!-- The signal $y$ is commonly modeled as the linear combination of $M$ basis functions $\phi_{m}(x)$: -->

$$y(x)=\sum_{m=1}^{M} w_{m}\phi_{m}(x),\tag{tz19}$$

ここで $w=(w_{1},\ldots, w_{M})^{\top}$ は線形結合の重みを表す。
設計行列 $\mathbf{\Phi}=( \phi_{1},\ldots, \phi_{M}$) で，$\phi_{m}=(\phi_{m}(x_ {1}),\ldots,\phi_{m}(x_{N})^{\top}$ を定義し，観測値 $t$ は次のようにモデル化される:
<!-- where $w=(w_ {1},\ldots,w_ {M})^\top$ are the weights of the linear combination.
Defining the design matrix $\mathbf{\Phi}=(\phi_ {1},\ldots,\phi_ {M}$, with $\phi_ {m}= (\phi_ {m}(x_ {1}),\ldots,\phi_ {m}(x_ {N})^{\top}$, the observations $t$ are modeled as  -->

$$\mathbf{t}=\mathbf{\Phi}\mathbf{w}+\epsilon\tag{tz20}$$

そして，尤度は次式で与えられる:<!-- and likelihood is  -->

$$
p(t;w,\beta) = \mathcal{N}(t\vert \Phi W,\beta^{-1}\mathbf{I}). \tag{tz21}
$$

以下では，前節までの理論を線形回帰問題に適用し，この線形モデルの未知の重み $w$ を計算する 3 つの方法論を示す。
まず，パラメータと仮定された重みの典型的な ML 推定を適用する。
これから示されるように，パラメータの数は観測値の数と同じであるため，ML 推定はモデルの雑音に非常に敏感で，観測値に過剰に適合してしまう。
そこで，この問題を改善するために，確率変数と仮定される重みに事前分布を課した。
まず，重みの定常ガウス事前分布に基づく単純なベイズモデルが使用される。
このモデルでは，EM アルゴリズムを用いてベイズ推定が行われ，結果として得られる解は雑音に対して頑健である。
しかしながら，このベイズモデルは非常に単純であり，局所的な信号の特性を捕らえる能力を持たない。
この目的のために，重みの非定常ガウス事前分布とハイパー事前確率に基づく，より洗練された空間的に変化する階層的モデルを導入することが可能である。
このモデルは，EM アルゴリズムで解くには複雑すぎる。
この目的のために `変分推論 EM の枠組み` 節で説明した変分ベイズ法が，このモデルの未知数の値を推論するために使用されている。
最後に，ベイズモデルの複雑性が解の質を向上させることを実証するために，3 つの方法を用いて簡単な人工信号の推定値を得る。
図 3 a, b, c は，線形回帰の 3 つのアプローチのグラフィカルモデルである。
<!-- In what follows we will apply the theory from earlier sections to the linear regression problem and demonstrate three methodologies to compute the unknown weights w of this linear model.
First, we apply typical ML estimation of the weights which are assumed to be parameters.
As it will be demonstrated, since the number of parameters is the same as the number of our observations, the ML estimates are very sensitive to the model noise and over fit the observations.
Subsequently, to ameliorate this problem a prior is imposed on the weights which are assumed to be random variables.
First, a simple Bayesian model is used which is based on a stationary Gaussian prior for the weights.
For this model, Bayesian inference is performed using the EM algorithm and the resulting solution is robust to noise.
Nevertheless, this Bayesian model is very simplistic and does not have the ability to capture the local signal properties. For this purpose it is possible to introduce a more sophisticated spatially varying hierarchical model which is based on a nonstationary Gaussian prior for the weights and a hyperprior.
This model is too complex to solve using the EM algorithm.
For this purpose, the variational Bayesian methodology described in the section "Variational EM Framework" is used to infer values for the unknowns of this model.
Finally, the three methods are used to obtain estimates of a simple artificial signal, in order to demonstare that the added complexity in the Bayesian model improves the solution quality.
In Figure 3(a), (b), and (c) we show the graphical models for the three approaches to Linear Regression. -->

線形モデルの重み $w$ の最も単純な推定値は，モデルの尤度を最大化することによって得られる。
この ML 推定は，図 3a のグラフィカルモデルに示すように，重み $w$ をパラメータと仮定している。
ML 推定値は尤度関数を最大化することにより得られる
<!-- The simplest estimate of the weights w of the linear model is obtained by maximizing the likelihood of the model.
This ML estimate assumes the weights w to be parameters, as shown in the graphical model of Figure 3(a).
The ML estimate is obtained by maximizing the likelihood function -->
$$p(t;w,\beta) =\left(2\pi\right)^{-\frac{N}{s}}\beta^{\frac{N}{2}}\exp\left(-\frac{\beta}{2}\left\|\mathbf{t}-\mathbf{\phi w}\right\|\right)$$
これは，$E_{LS(w)}=\left\|t-\mathbf{\phi}w\right\|^{2}$ を最小化することと等価である。
したがって，この場合，ML は最小二乗 (LS) 推定値と等価である
<!-- This is equivalent to minimizing ELS(w) =  t − w 2.
Thus, in this case the ML is equivalent with the least squares (LS) estimate -->
$$
w_{LS}={\arg\max}_{w} p(t;w,\beta)={\arg\min}_{w}\mathbb{E}_{LS(w)}=\left(\mathbf{\phi}^{\top}\mathbf{\phi}\right)^{-1}\mathbf{\phi}^{\top}t\tag{tz22}
$$

多くの状況で (そして使用される基底関数に依存して)，行列 $\mathbf{\phi}^{\top}\mathbf{\phi}$ は条件不一致で反転が困難な場合がある。
これは，雑音 $\epsilon$ が信号の観測値に含まれる場合，重みの推定値 $w_{LS}$ に大きく影響することを意味する。
したがって，ML 線形回帰を使用する場合，行列 $\mathbf{\phi}^{\top}\mathbf{\phi}$ が反転できるように基底関数を慎重に選択する必要がある。
これは一般に，基底関数が少ないスパースモデルを使用することで達成され，また推定しなければならないパラメータが少ないという利点もある。
<!-- In many situations (and depending on the basis functions that are used), the matrix phi^{top}phi may be ill-conditioned and difficult to invert.
This means that if noise ε is included in the signal observations, it will heavily affect the estimation wLS of the weights.
Thus, when using ML linear regression (MLLR), the basis functions should be carefully chosen to ensure that matrix mathbf{phi}^{top}\mathbf{\phi} can be inverted.
This is generally achieved by using a sparse model with few basis functions, which also has the advantage that only few parameters have to be estimated. -->

## 6.1 EM アルゴリズムに基づく，ベイジアン線形回帰 EM-Based Bayesian Linear Regression (p7L)

線形モデルのベイズ的な取り扱いは，まずモデルの重みに **事前分布** を割り当てることから始まる。
これは推定にバイアスをもたらすが，ML 推定の大きな問題であるその分散を大幅に減少させる。
ここでは，線形モデルの重みに独立で，かつ，平均ゼロのガウス型事前分布をよく選択することを考える。
<!-- A Bayesian treatment of the linear model begins by assigning **a prior distribution** to the weights of the model.
This introduces bias in the estimation but also greatly reduces its variance, which is a major problem of the ML estimate.
Here, we consider the common choice of independent, zero-mean, Gaussian prior distribution for the weights of the linear model: -->

$$
p(w;\alpha)= \prod_{m=1}^{M} \mathcal{N}(w_{m}\vert 0,\alpha^{-1}).\tag{tz23}
$$

これは定常的な事前分布であり，すべての重みの分布が同一であることを意味する。
この問題のグラフィカルモデルを 図 [@2008TzikasFig3(b)] に示す。
ここで，重み $w$ は隠れ確率変数であり，モデルパラメータは $w$ の事前分布のパラメータ $\alpha$ と加法性ノイズの逆分散 $\eta$ であることに注意されたい。
<!-- This is a stationary prior distribution, meaning that the distribution of all the weights is identical.
The graphical model for this problem is shown in Figure [@2008TzikasFig3](b).
Notice that here the weights $w$ are hidden random variables and the model parameters are the parameter $\alpha$ of the prior for $w$ and the inverse variance $\beta$ of the additive noise. -->

<div class="figure figcenter">
<img src="../figures/2008TzikasFig3a.svg" width="19%">
<img src="../figures/2008TzikasFig3b.svg" width="19%">
<img src="../figures/2008TzikasFig3c.svg" width="19%">
<div class="figcaption">

図 3. 線形回帰のグラフモデルは (a) 直接 ML 推定 (事前分布のないモデル) を用いて解く。
(b) EM (定常的な事前分布を持つモデル), および
(c) 変分 EM (階層的事前分布を持つモデル)
From [@2008Tzikas_VaBayes] Fig.3
<!-- Graphical models for linear regression solved using (a) direct ML estimation (model without prior),
(b) EM (model with stationary prior), and
(c) variational EM (model with hierarchical prior).  -->
</div></div>

ベイズ推論は，隠れ変数の事後分布を計算することで進められる。
<!-- Bayesian inference proceeds by computing the posterior distribution of the hidden variables: -->
$$p(w\vert t;\alpha,\beta) = \frac{p(t\vert w;\beta) p(w;\alpha)}{p(t;\alpha,\beta)}.\tag{tz24}$$

分母に現れる周辺尤度 $p(t;\alpha,\beta)$ は解析的に計算できることに注意。
<!-- Notice that the marginal likelihood $p\of{t;\alpha,\beta}$ that appears on the denominator can be computed analytically: -->
$$
p(t;\alpha,\beta) =\int p(t\vert w;\beta)\;p(w;\alpha)\;dw
= \mathcal{N}\left(t\vert 0,\beta^{-1}\mathbf{I}+\alpha^{-1}\mathbf{\Phi\Phi}^{\top}\right).\tag{tz25}
$$

このとき，隠れ変数の事後確率は:
<!-- Then, the posterior of the hidden variables is -->
$$p(w\vert t;\alpha,\beta)= \mathcal{N}(w\vert \mu,\sigma),\tag{tz26}$$
ここで<!-- with-->
$$
\begin{aligned}
\mathbf{\mu}    &=\beta\,\mathbf{\Sigma\Phi}^{\top} t.\\
\mathbf{\Sigma} &=\beta\,\mathbf{\Phi}^{\top} \mathbf{\Phi} + \alpha\mathbf{I}^{-1}.\\
\end{aligned}
\tag{tz27,tz28}
$$

モデルのパラメータは、周辺尤度 $p(t;\alpha,\beta)$ の対数を最大化することにより推定できる:
<!-- The parameters of the model can be estimated by maximizing the logarithm of the marginal likelihood $p(t;\alpha,\beta)$:-->

$$(\alpha_{ML},\beta_{ML}) =\arg\min_ {\alpha,\beta}
\left[\log\left|\beta^{-1}\mathbf{I} + \alpha^{-1}\mathbf{\Phi\Phi}^{\top}\right|\right]
+ \mathbf{t}^{\top}
\left[ \beta^{-1} \mathbf{I} + \alpha^{-1} \mathbf{\Phi\Phi}^{\top-1}t \right].\tag{tz29}
$$

式 (29) はパラメータ $(\alpha,\beta)$ に関する導関数が計算しにくいため，直接最適化を行うにはいくつかの計算上の困難が伴う。
また $(\alpha,\beta)$ の推定値は逆分散を表すため正でなければならず，制約付き最適化アルゴリズムが必要である。
その代わりに，先に述べた EM アルゴリズムにより $(\alpha,\beta)$ の推定値の取得と $w$ の値の推定を同時に行う効率的な枠組みを提供する。
EM アルゴリズムでは，周辺尤度 式(tz:25) の計算を行わないが，その局所最大値に収束することに注意すること。
パラメータ $\left\{\alpha^ {0},\beta^ {0}\right\}$ を適当な値に初期化した後，アルゴリズムは以下のステップを繰り返し実行することで進められる。
<!-- Direct optimization of Eq.(\ref{eq:tz29}) presents several computational difficulties, since its derivatives with respect to the parameters $\Brc{\alpha,\beta}$ are difficult to compute.
Furthermore, the problem requires a constrained optimization algorithm since the estimates of $\Brc{\alpha,\beta}$ have to be positive since they represent inverse variances.
Instead, the EM algorithm described earlier, provides an efficient framework to simultaneously obtain estimates for $\Brc{\alpha,\beta}$ and infer values for $w$.
Notice, that although the EM algorithm does not involve computations with the marginal likelihood Eq.(\ref{eq:tz25}), the algorithm converges to a local maximum of it.
After initializing the parameters to some values $\left{\alpha\of{0},\beta\of{0}\right}$, the algorithm proceeds by iteratively performing the following steps:-->

### 6.1.1 E step:
完全尤度対数の期待値を計算:
<!-- Compute the expected value of the logarithm of the complete likelihood :-->

$$
\begin{aligned}
Q^{(t)}(t,w;\alpha,\beta) &= \left<\log p(t,w;\alpha,\beta)\right>_{p(w\vert t;\alpha^{(t)},\beta^{(t)})}\\
& =\left<\log p(t\vert w;\alpha,\beta) p(w;\alpha,\beta)\right>_{p(w\vert t;\alpha^{(t)},\beta^{(t)})}.\\
\end{aligned}\tag{tz30}
$$

上式は，(21)，(23) 式を用いて計算され:
<!-- This is computed using (tz21) and (tz23) as -->

<!--
$$
\begin{aligned}
Q^{(t)}(t,w;\alpha,\beta) &=  \left<\frac{N}{2} \log\beta - \frac{\beta}{2}\right.\\
\end{aligned}
$$

$$
\begin{aligned}
\left(\left|\mathbf{t}-\mathbf{\Phi\mu}^{(t)}\right|^{2}\right).\tag{2}
\end{aligned}
$$

$$
\begin{aligned}
+ \frac{M}{2}\ln\alha -\frac{\alpha}{2}\left(\left\|w\right\|^ {2}\right)\right>+\text{const}\\
\end{aligned}
$$
-->

$$\begin{aligned}
Q^{(t)}(t,w;\alpha,\beta) &= \left<\frac{N}{2} \log\beta - \frac{\beta}{2}\left(\left\|\mathbf{t}-\mathbf{\Phi\mu}^{(t)}\right\|^{2}\right)\right> + \text{const}\\
&= \frac{N}{2} \log\beta - \frac{\beta}{2}\left(\left\|\mathbf{t}-\mathbf{\Phi\mu}^{(t)}\right\|^{2}\right)
+ \frac{M}{2}\log\alpha - \frac{\alpha}{2}\left(\left<\left\|\mathbf{w}\right\|^{2}\right>\right)+\text{const}\\
\end{aligned}
\tag{tz31}$$

これらの期待値は $p(w\vert t;\alpha^{t},\beta^{(t)})$ に対するもので，式 (26) から計算すると次のようになる:
<!-- These expected values are with respect to $p(w\vert t;\alpha^{t},\beta^{(t)})$ and can be computed from Eq.(tz26), giving -->

$$
Q^{(t)}(t,w;\alpha,\beta) = \frac{N}{2} \log\beta
- \frac{\beta}{2}\left(\left\|\mathbf{t}-\mathbf{\Phi\mu}^{(t)}\right\|^{2}
+ \text{tr}\left| \mathbf{\Phi}^{\top}\mathbf{\Sigma}^{(t)} \mathbf{\Phi} \right|\right)
+ \frac{M}{2} \log\alpha
- \frac{\alpha}{2}\left(\left|\mathbf{\mu}^{(t)}\right|^{2}
+ \text{tr}\left|\mathbf{\Sigma}^{(t)}\right|\right) + \text{const}
\tag{tz32}$$

ここで $\mathbf{\mu}^{(t)}$ と $\mathbf{\Sigma}^{(t)}$ は，パラメータ $\alpha^{(t)}$ と $\beta^{(t)}$ の現在の推定値を使用して計算される:
<!-- where $\mathbf{\mu}^{(t)}$ and $\mathbf{\Sigma}^{(t)}$ are computed using the current estimates of the parameters $\alpha^{(t)}$ and $\beta^{(t)}$: -->

$$\begin{aligned}
\mathbf{\mu}^{(t)}    &= \beta^{(t)} \mathbf{\Sigma}^{(t)} \mathbf{\Phi}^{\top} \mathbf{t},\\
\mathbf{\Sigma}^{(t)} &= \left(\beta^{(t)} \mathbf{\Phi}^{\top} \mathbf{\Phi} + \alpha^{(t) \mathbf{I}}\right)^{-1}.\\
\end{aligned}\tag{tz33,tz34}$$

### 6.1.2 M step:

$Q^{(t)}(t,w;\alpha,\beta)$ をパラメータ $\alpha$ と $\beta$ に関して最大化する。
<!-- Maximize $Q^{(t)}(t,w;\alpha,\beta)$ with respect to the parameters $\alpha$ and $\beta$: -->

$$\left(\alpha^{(t+1)},\beta^{(t+1)}\right)=\arg\max_{(\alpha,\beta)}Q^{(t)}\left(t,w;\alpha,\beta\right).\tag{tz35}$$

$Q^{(t)}(t,w;\alpha,\beta)$ のパラメータに関する導関数は次の通りである。
<!-- The derivatives of $Q^{(t)}(t,w;\alpha,\beta)$ with respect to the prameters are: -->

$$\begin{aligned}
\frac{\partial Q^{(t)}}{\partial\alpha} &=\frac{M}{2\alpha}-\frac{1}{2}\left(\left\|\mathbf{\mu}^{(t)}\right\|^{2}+\text{tr}\left|\Sigma^{(t)}\right|\right)\\
\frac{\partial Q^{(t)}}{\partial\beta}  &=\frac{N}{2\beta}-\frac{1}{2}\left(\left\|t-\mathbf{\phi\mu}^{(t)}\right\|^{2} +\text{tr}\left|\mathbf{\phi}^{\top}\mathbf{\Sigma}^{(t)}\mathbf{\phi}\right|\right).
\end{aligned}\tag{tz36,tz37}
$$

これらを 0 とすると，パラメータ $\alpha$, $\beta$ を更新する以下の式が得られる:
<!-- Setting these to zero, we obtain the following formulas to update the parameters α and β: -->

$$\begin{aligned}
\alpha^{(t+1)} &= \frac{M}{\left\|\mathbf{\mu}^{(t)}\right\|^{2}+\text{tr}\left|\mathbf{\Sigma}^{(t)}\right|}\\
\beta^{(t+1)}  &= \frac{N}{\left\|t-\mathbf{\phi\mu}^{(t)}\right\|^{2}+\text{tr}\left|\mathbf{\Phi}^{\top}\mathbf{\Sigma}^{(t)}\mathbf{\Phi}\right|}
\end{aligned}\tag{tz38,tz39}$$

数値最適化を必要とする (25) の周辺尤度の直接最大化とは対照的に，最大化ステップは解析的に実行できることに注意されたい。
さらに (38) と (39) はパラメータ $\alpha$ と $\beta$ の正の推定を保証しており，これらは逆分散パラメータを表しているので，これは必要条件である。
ただし，パラメータの初期化によっては，異なる局所最大値になる可能性があるため，初期化には注意が必要である。
E-step では事後統計量 $p(w|t;\alpha,\beta)$ が計算されているので，$w$ に対する推論が直接得られる。
この事後統計量の平均値 (33) は $w$ のベイズ線形最小平均二乗誤差 (LMMSE) 推論として用いることができる。
<!-- Notice that the maximization step can be analytically performed in contrast to direct maximization of the marginal likelihood in (25), which would require numerical optimization.
Furthermore, (38) and (39) guarantee that positive estimations for the parameters α and β are produced, which is a requirement since these represent inverse variance parameters.
However, the parameters should be initialized with care, since depending on the initialization a different local maximum may be attained.
Inference for w is obtained directly since the sufficient statistics of the posterior p(w|t; α, β) are computed in the E-step.
The mean of this posterior (33) can be used as Bayesian linear minimum mean square error (LMMSE) inference for w. -->


## 6.2 変分 EM に基づく Bayesian 線形回帰 (P9L)<!-- ## VARIATIONAL EM-BASED BAYESIAN LINEAR REGRESSION-->

前節で述べたベイズ的アプローチでは，線形モデルの重みに定常ガウス事前分布を用いるため，限界尤度の厳密な計算が可能で，ベイズ推定は解析的に行われる。
しかし，多くの場面で，信号の局所的な特性を柔軟にモデル化することが重要であり，これは単純な定常ガウス事前分布では不可能である。
このため，各重みに対して明確な逆分散 $\alpha_ {m}$ を持つ非定常ガウス事前分布が検討される。
<!-- In the Bayesian approach described in the previous section, due to the use of a stationary Gaussian prior distribution for the weights of the linear model, exact computation of the marginal likelihood is possible and Bayesian inference is performed analytically.
However, in many situations, it is important to allow the flexibility to model local characteristics of the signal, which the simple stationary Gaussian prior distribution is unable to do. For this reason, a non-stationary Gaussian prior distribution with a distinct inverse variance αm for each weight is considered: -->

$$ p\left(w\vert\mathbf{\alpha}\right) = \prod_ {m=1} ^ {M}\mathcal{N}\left(w_ {m}\vert 0, \alpha_ {m}^{-1}\right)\tag{tz40} $$

しかし，このようなモデルは，推定すべきパラメータとほぼ同数の観測値があるため，パラメータが過剰になる。
このため 精度パラメータ $\mathbf{\alpha}=\left(\alpha_ {1},\ldots,\alpha_ {M}\right)^{\top}$ を確率変数として扱い，事前ガンマ分布を課すことにより制約をかける。
<!-- However, such a model is over-parameterized since there are almost as many observations as parameters to be estimated.
For this purpose, the precision parameters $\mathbf{\alpha}=\left(\alpha_ {1},\ldots,\alpha_ {M}\right)^{\top}$ are constrained by treating them as random variables and imposing a Gamma prior distribution to them according to -->

$$ p\left(\mathbf{\alpha};a,b\right) = \prod_ {m=1}^ {M}\text{Gamma}(\alpha_ {m}\vert a,b)\tag{tz41} $$

この事前分布は Gaussian 分布と共役であることから選択された[13]。
さらに，ノイズの逆分散 $\beta$ の事前分布としてガンマ分布を仮定する
<!-- This prior is selected because it is conjugate to the Gaussian [13].
Furthermore, we assume a Gamma distribution as prior for the noise inverse variance β-->

$$ p(\beta;c,d)=\text{Gamma}(\beta\vert c, d).\tag{tz42} $$

この Bayes プローチのグラフィカルモデルは 図 3(c) のようになり，隠れ変数 $w$ の隠れ変数 $\alpha$ への依存性が明らかになる。
また，このモデルのパラメータ $a, b, c, d$  とそれらに依存する隠れ変数も明らかにされている。
<!-- The graphical model for this Bayesian approach is shown in Figure 3(c) where the dependence of the hidden variables w on the hidden variables α is apparent.
Also, the parameters a, b, c, and d of this model and the hidden variables that depend on them are also apparent.-->

Bayes 推論では，事後分布の計算が必要であり:
<!-- Bayesian inference requires the computation of the posterior distribution -->

$$
p(w,\alpha\beta\vert t) = \frac{p(t\vert w, \beta)p(w\vert\alpha)p(\alpha)p(\beta)}{p(t)}.\tag{tz43}
$$

しかし，周辺尤度 $\displaystyle p(t)=\int p(t\vert w,\beta)p(w\vert \alpha)p(\alpha)dw\;d\alpha\;d\beta$ は解析的に計算できないため，式(43) の正規化定数を計算することができない。
そこで，近似的な Bayes 推論手法，特に変分推論手法に頼ることになる。
重み $w$ と分散パラメータ $\alpha,\beta$ の間に事後独立性があると仮定する。
<!-- However, the marginal likelihood p(t)=in p(t|w, β)p(w|α)p(α) p(β)dwdαdβ cannot be computed analytically, and thus the normalization constant in (43) cannot be computed.
Thus, we resort to approximate Bayesian inference methods and specifically to the variational inference methodology.
Assuming posterior independence between the weights w and the variance parameters α and β,-->

$$
p(w,\alpha,\beta\vert t; a, b, c, d)\approx q(w,\alpha,\beta) = q(w)q(\alpha)q(\beta),\tag{tz44}
$$

は (16) 式から次のように近似的な事後分布 $q$ を計算することができる。
$\ln q(w)$ のうち $w$ に依存する項だけを残して，次のようになる。
<!-- the approximate posterior distributions q can be computed from (16) as follows.
Keeping only the terms of ln q(w) that depend on w, we have-->

$$\begin{aligned}
\ln q(w) &=\left<\ln p(t,w,\mathbf{\alpha},\beta)_{q(\mathbf{\alpha})q(\beta)}\right>+\text{const}\\
&= \left<\ln p(t|w,\beta) p(w|\mathbf{\alpha})_{q(\mathbf{\alpha})q(\beta)}\right>+\text{const}\\
&= \left<\ln p(t|w,\beta) + \ln p(w|\mathbf{\alpha})_{q(\mathbf{\alpha})q(\beta)}\right>+\text{const}\\
&= \left<-\frac{\beta}{2}\left(t-\Phi w\right)^{\top}\left(t-\Phi w\right)-\frac{1}{2}\sum_{m=1}^{M}\alpha_{m}w_{m}^{2}\right>+\text{const}\\
&=\frac{\left<\beta\right>}{2}\left[t^{\top}t-2t^{\top}\Phi w+w^{\top}\Phi^{\top}\Phi w\right]
- \frac{1}{2}\sum_{m=1}^{M}\left<\alpha_{m}\right>w_{m}^{2}+\text{const}\\
&=-\frac{1}{2}w^{\top}\left(\left<\beta\right>\Phi^{\top}\Phi+\left<\mathbf{A}\right>\right)w-\left<\beta\right>w^{\top}\Phi^{\top}t+\text{const}\\
&=-\frac{1}{2}w^{\top}\Sigma^{-1}w - w^{\top}\Sigma^{-1}\mathbf{\mu}+\text{const}
\end{aligned}\tag{tz45}$$
ここで，$\mathbf{A}=\text{diag}\left(\alpha_{1},\ldots,\alpha_{M}\right)$.
<!-- where A = diag(α1, . . . , αM). -->

これは，平均 $\mu$ 分散行列 $\Sigma$ に従う指数 Gauss 分布であり，それぞれ次式に従う:
<!-- Notice that this is the exponent of a Gaussian distribution with mean μ and covariance matrix   given by -->

$$\begin{aligned}
\mathbf{\mu} &=\left<\beta\right>\mathbf{\Sigma\Phi}^{\top}t.\\
\mathbf{\Sigma} &= \left(\left<\beta\right>\mathbf{\Phi}^{\top}\mathbf{\Phi}+\left<A\right>\right)^{-1}.
\end{aligned}\tag{taz46,tz47}$$

それゆえ，$q(w)$ は，次式で与えられる:

$$q(w)=\mathcal{N}\left(w|\mathbf{\mu},\mathbf{\Sigma}\right).\tag{tz48}$$

事後確率 $q(\mathbf{\alpha})$ は同様に $\mathbf{\alpha}$ に依存する $\ln q(\mathbf{\alpha})$ を計算することで得られる。

$$\begin{aligned}
\ln q(\mathbf{\alpha}) &= \left<\ln p(t,w,\mathbf{\alpha},\beta\right>_{q(w) q(\beta)}\\
\end{aligned}\tag{tz49}$$

これは，パラメータ $\tilde{a},\tilde{b}_m$ を持つ $M$ 個の独立なガンマ分布の積の指数で，次式で与えられる:
<!-- This is the exponent of the product of M independent Gamma distributions with parametersa and bm, given by -->

$$\begin{aligned}
\tilde{a} &= a+\frac{1}{2}\\
\tilde{b}_{m} &= b + \frac{1}{2}\left<w_{m}^{2}\right>.
\end{aligned}\tag{tz50,tz51}$$
それゆえ $q(\mathbf{\alpha})$ は，次式で与えられ:
$$
q(\alpha)=\prod_{m=1}^{M}\Gamma(\alpha_{m}|\tilde{a},\tilde{b}_{m}).\tag{tz52}
$$
<!-- q(\alpha)=\prod_{m=1}^{M}\text{Gamma}(\alpha_{m}|\tilde{a},\tilde{b}_{m}).\tag{tz52} -->

雑音の逆分散の事後分布も同様に次のように計算される:
<!-- The posterior distribution of the noise inverse variance can be similarly computed as -->

$$\tag{tz53}
q(\beta)=\Gamma(\beta|\tilde{c},\tilde{d}_m),
$$
with
$$
\tag{tz54,tz55}
\begin{aligned}
\tilde{c} &= c+\frac{N}{2},\\
\tilde{d} &= d+\frac{1}{2}\left<\left\|\mathbf{t}-\mathbf{\Phi w}\right\|^{2}\right>.\\
\end{aligned}
$$

そして，(48), (52), (53) の近似事後分布は，互いの統計量に依存するため，収束するまで繰り返し更新される，詳細は [22] を参照のこと。
<!-- The approximate posterior distributions in (48), (52), and (53) are then iteratively updated until convergence, since they depend on the statistics of each other, see [22] for details. -->

ここで，重みの真の事前分布は，ハイパーパラメータ $\alpha$ を周辺化することで計算できることに注意。
<!-- Notice here, that the true prior distribution of the weights can be computed by marginalizing the hyperparameters alpha: -->

$$\tag{tz56}
\begin{aligned}
p(w,a,b) &= \int p(w|\alpha) p(\alpha;a,b)d\alpha\\
         &= \int \prod_{m=1}^{M} N\left(w_m|0,\alpha_m^{-1}\right)\Gamma (a_m|a,b)d\alpha_{m}\\
         &= \prod_{m=1}^{m} S t(w_m|\lambda,\nu)
\end{aligned}
$$

and is a student-t pdf,
$$
St(x|\mu,\lambda,\nu)=\frac{\Gamma((\nu+1)/2}{\Gamma(\nu/2)}\left(\frac{\lambda}{\pi\nu}\right)^{1/2}\times \left[1+\frac{\lambda(x-\mu)^{2}}{\nu}\right]^{-(\nu+1)/2},
$$

で 平均 $\mu=0$,  パラメータ $\lambda= a/b$, 自由度 $\nu=2a$。
この分布は自由度 $\nu$ が小さい場合，重尾分布となる。
したがって，この分布は，少数の基底関数だけを含み，残りの基底関数を対応する重みを非常に小さな値に設定することによって刈り込む，疎な解に有利である。
最終的なモデルで実際に使用される基底関数を関連性基底関数と呼ぶ。
<!-- with mean μ = 0, parameter λ = a/b and degrees of freedom ν = 2a.
This distribution, for small degrees of freedom ν, exhibits very heavy tails.
Thus, it favours sparse solutions, which include only few of the basis functions and prunes the remaining basis functions by setting the corresponding weights to very small values.
Those basis functions that are actually used in the final model are called relevance basis functions. -->

簡単のために，Student-t 分布のパラメータ a, b, c, d を固定とした。
実際には，これらのパラメータを非常に小さな値，すなわち，$a=b=c=d= 10^{-6}$ に設定することによって得られる非情報的な分布を仮定することによって，しばしば良い結果を得ることができる。
あるいは，変分 EM アルゴリズムを用いて，これらのパラメータを推定することもできる。
このようなアルゴリズムは，これらのパラメータに関して変分境界を最大化する M-ステップを，説明した方法に追加することになる。
しかし，ベイズモデリングにおける典型的なアプローチは，ハイパーパラメータを固定し，モデルの最上位レベルで非情報的なハイパー事前分布を定義することである。
<!-- For simplicity, we have assumed fixed the parameters a, b, c, and d of the student-t distributions.
In practice, we can often obtain good results by assuming uninformative distributions, which are obtained by setting these parameters to very small values, i.e., a = b = c = d = 10−6 .
Alternatively, we can estimate these parameters using a variational EM algorithm.
Such an algorithm would add an M-step to the described method, in which the variational bound would be maximized with respect to these parameters.
However, the typical approach in Bayesian modeling is to fix the hyperparameters to define uninformative hyperpriors at the highest level of the model. -->


## 6.3 線形回帰の例 (p10R)
<!-- ## 6.3 LINEAR REGRESSION EXAMPLES (p10R)-->

次に，先に説明した線形回帰モデルの特性を示す数値例を示す。
また，変分 Bayes 推論を用いることで得られる利点を示す。
人工的に生成された信号 $y(x)$ を用いるので，観測値を生成した元の信号は既知であり，したがって推定の品質を評価することができる。
信号の $N=50$ サンプルを得て，分散 $\sigma^{2}=4\times10^{-2}$ のガウス雑音を加え，SN 比 $\text{SNR}=6.6$ dB に相当させた。
$N$ 個の基底関数を用い，特に各信号観測位置を中心とした基底関数を 1 個用いた。
基底関数は以下の形のガウシアンカーネルである。
<!-- Next, we present numerical examples to demonstrate the properties of the previously described linear regression models.
We also demonstrate the advantages that can be reaped by using the variational Bayesian inference.
An artificially generated signal y(x) is used, so that the original signal which generated the observations is known and therefore the quality of the estimations can be evaluated. We have obtained N = 50 samples of the signal and added Gaussian noise of variance σ2 = 4 × 10−2 , which corresponds to signal to noise ratio SNR = 6.6 dB.
We used N basis functions and, specifically, one basis function centred at the location of each signal observation.
The basis functions were Gaussian kernels of the form  -->

$$\phi_{i}(x)=K(x,x_{i})=\exp\left(-\frac{1}{2\sigma^{2}_{\phi}}\left\|x-x_{i}\right\|^{2}\right).\tag{tz57}$$

この例では，非常に重い尾を持つ，情報量の少ない Student-t 分布を得るために $a=b=0$ と設定した。
<!-- In this example we set a = b = 0, in order to obtain a very heavy-tailed, uninformative Student-t distribution. -->

次に，この観測結果を用いて i) 最尤推定 (ML 推定) (22), ii) EM-に基づくベイズ推論 (33), iii) 変分 EM-に基づく ベイズ推論 (46) により、信号の出力を予測することができた。
結果は 図4(a) に示されている。
ML 推定は雑音の多い観測値に忠実に従うことに注目されたい。
したがって，平均二乗誤差の点では最悪である。
この定式化では，観測値と同じ数の基底関数を使用し，重みに制約がないため，これは予想されることである。
ベイズ法は，重みが事前分布によって制約されるため，この問題を克服している。
しかし，この信号には大きな分散を持つ領域と非常に小さな分散を持つ領域があるため，定常事前分布ではその局所的な振る舞いを正確にモデル化できないことは明らかである。
これに対して，階層的非定常事前分布はより柔軟で，より良い局所的な適合が得られる。
実際，後者の事前分布に対応する解は，基底関数の小さな部分集合しか使用しておらず，その位置は図 4 の丸で囲んだ観測値として示されている。
これは，我々は $a=b=0$ を設定したためで，情報量の少ない Student-t 分布を定義している。
したがって，ほとんどの重みは正確に 0 と推定され，信号推定に使用される基底関数はごくわずかである。
最終的なモデルで実際に使用されるこれらの基底関数は関連性基底関数と呼ばれ，それらが中心化されたベクトルは関連性ベクトル (RV: revelence vector) と呼ばれ，図 4 に示されている。
<!-- We then used the observations to predict the output of the signal, using i) ML estimation (22), ii) EM-based Bayesian inference (33), and iii) variational EM-based Bayesian inference (46).
Results are shown in Figure 4(a).
Notice that the ML estimate follows exactly the noisy observations.
Thus, it is the worst in terms of mean square error.
This should be expected, since in this formulation we use as many basis functions as the observations and there is no constraint on the weights.
The Bayesian methodology overcomes this problem since the weights are constrained by the priors.
However, since this signal contains regions with large variance and some with very small variance, it is clear that the stationary prior is not capable of accurately modeling its local behavior.
In contrast, the hierarchical non-stationary prior is more flexible and seems to achieve better local fit.
Actually, the solution corresponding to the latter prior, uses only a small subset of the basis functions, whose locations are shown as circled observations in Figure 4.
This happens because we have set a = b = 0, which defines an uninformative Student-t distribution.
Therefore, most weights are estimated to be exactly zero and only few basis functions are used in the signal estimation.
Those basis functions that are actually used in the final model are called relevance basis function and the vectors where they are centered are called relevance vectors (RV) and are shown in Figure 4. -->

<div class="figure figcenter">
<img src="../figures/2008Tzikas_fig4.svg" width="66%">
<div class="figcaption" style="width:49%">

図 4 最尤 (ML)推定，EM に基づく推論，変分 EM ベイズ推論で得られた線形回帰の解。
<!-- FIG. 4 Linear regression solutions obtained by ML estimation, EM-based Bayesian inference and variational-EM Bayesian inference. -->
</div>
</div>

この例と同じ精神で，画像復元問題，画像超解像問題，画像ブラインドデコンボリューション問題に対して，それぞれ [23], [24], [25] で階層的な非定常事前分布が提案されている。
画像再構成問題において，このような事前分布は，画像のエッジを保持すると同時に，画像の平坦部における雑音を抑制する能力を実証した。
さらに，このような事前分布は，下絵が未知の場合の電子透かし検出器の設計にも用いられ，成功を収めている[26]。
<!-- In the same spirit as this example, a hierarchical nonstationary prior has been proposed for the image restoration, image super resolution, and image blind de-convolution problems in [23], [24], and [25], respectively.
In image reconstruction problems, such priors demonstrated the ability to preserve image edges and at the same time suppress noise in flat areas of the image.
In addition, priors of this nature have also been used with success to design watermark detectors when the underlying image is unknown [26]. -->

# 7. Gauss 混合モデル (p10R)<!-- # 7. GAUSSIAN MIXTURE MODELS-->

Gauss 混合モデル (GMM) は，密度をモデル化するための貴重な統計ツールである。
任意の密度を高い精度で近似できる柔軟性があり，さらに，ソフトクラスタリング解として解釈することができる。
GMM は音声理解，画像モデリング，追跡，セグメンテーション，認識，電子透かし，ノイズ除去など様々な信号処理問題で広く用いられている。
<!-- Gaussian mixture models (GMM) are a valuable statistical tool for modeling densities.
They are flexible enough to approximate any given density with high accuracy and in addition, they can be interpreted as a soft clustering solution.
They have been widely used in a variety of signal processing problems ranging from speech understanding, image modeling, tracking, segmentation, recognition, watermarking, and denoising. -->

GMM はガウス分布の凸組み合わせとして定義され，単一の分布では十分でない場合にデータセットの密度を記述するために広く用いられている。
M 個の成分を持つ混合モデルを定義するには，各成分 $j$ の確率密度 $p_j(x)$ と，成分の混合重み $\pi_{j}(\pi_{j}\ge0$ と $\sum_{j=1}^{M}\pi_{j}=1$) を含む確率ベクトル $(\pi_{1},\ldots,\pi_{M})$ を指定しなければならない。
<!-- A GMM is defined as a convex combination of Gaussian densities and is widely used to describe the density of a dataset in cases where a single distribution does not suffice.
To define a mixture model with M components we have to specify the probability density pj (x) of each component j as well as the probability vector (π1, . . . , πM) containing the mixing weights πj of the components (πj ≥ 0 and  M j =1 πj = 1). -->

このような混合物を用いてデータセット X の密度をモデル化する場合の重要な仮定は，各データムが以下の手順で生成されていることである：
<!-- An important assumption when using such a mixture to model the density of a dataset X is that each datum has been generated using the following procedure: -->

1. 成分 k を確率ベクトル $\pi_1,\ldots,pi_M$ を用いてランダムにサンプリングする。
2. 成分 k の密度 $p_k(x)$ からサンプリングして観測を生成する。

<!-- 1. We randomly sample one component k using the probability vector $\pi_1,\ldots,\pi_M$.
2. We generate an observation by sampling from the density pk(x) of component k. -->

ここで，離散隠れ確率変数 $Z$ は，観測サンプル $x$ を生成するために，すなわち，観測確率変数 $X$ に値 $X=x$ を割り当てるために選択された成分を表す。
このグラフモデルでは，ノード分布は $P(Z=j)=\pi_j$ と $P(X=x|Z=j)=p_j(x)$ である。
X と Z との同時確率密度関数については，次式が成り立つ
<!-- The graphical model corresponding to above generation process is presented in Figure 5(b), where the discrete hidden random variable Z represents the component that has been selected to generate an observed sample x, i.e., to assign the value X=x to the observed random variable X.
In this graphical model, the node distributions are $P(Z=j)=pi_j$ and $P(X=x|Z=j)=p_j(x)$. -->
<!-- For the joint pdf of X and Z it holds that -->
$$
p(\mathbf{X},Z) = p(\mathbf{X}|Z) p(Z)\tag{tz58}
$$
そして，$Z$ を周辺化することにより，混合モデルのよく知られた公式が得られる。
<!-- and through marginalization of Z we obtain the well-known formula for mixture models -->
$$
p(\mathbf{X}=x)=\sum_{j=1}^{M}p(\mathbf{X}=x|Z=j)p(Z=j)=\sum_{j=1}^{M}\pi_{j}p_{j}(x).\tag{tz59}
$$

GMM の場合，各成分 $j$ の密度は $p_j(x)=N(x;\mu_j,\Sigma_j)$となり，$\mu_j\in\mathcal{R}^d$ は平均 $\Sigma_j$ は $d\times d$ の共分散行列を表す。
それゆえ，
<!-- In the case of GMMs, the density of each component j is p_j(x)=N(x;mu_j,Sigma_j) where mμ_jinmathcal{R}^d denotes the mean and Sigma_j is the d×d covariance matrix.
Therefore -->
$$
p(x)=\sum_{j=1}^{M}\pi_{j}N\left(x;\mathbf{\mu}_{j},\mathbf{\Sigma}_{j}\right).\tag{tz60}
$$

<center>
<img src="../figures/2008Tzikas_fig5.svg" style="width:49%">
<div class="figcaption" style="width:49%">

図 4. グラフィカルモデル (a) 単一ガウス成分，(b) ガウス金剛モデル
<!-- [FIG5] Graphical models (a) for a single Gaussian component, (b) for a Gaussian mixture model. -->
</div>
</center>

混合モデルにおける注目すべき利便性は，ベイズの定理を用いて，ある観察 $x$ が混合成分 $j$ の分布からサンプリングして生成されたという事後確率 $P(j|x)=p(Z=j|x)$ を計算するのが簡単であることである。
<!-- A notable convenience in mixture models is that using Bayes’ theorem it is straightforward to compute the posterior probability P(j|x)=p(Z=j|x) that an observation x has been generated by sampling from the distribution of mixture component j -->
$$
P(j|X)=\frac{p(x|Z=j)p(Z=j)}{p(x)}=\frac{\pi_j N(x|\mu,\Sigma_j)}{\sum_{l=1}^{M}\pi_l N(x|\mu_l,\Sigma_l)}\tag{tz61}
$$
この確率は，観察 $x$ を生成した成分 $j$ の責任と呼ばれることもある。
さらに，データ点 $x$ を最大事後確率を持つ成分に割り当てることで，データ集合 $X$ を$M$ 個のクラスタにクラスタリングすることが簡単にできる。
<!-- This probability is sometimes referred to as the responsibility of component j for generating observation x.
In addition, by assigning a data point x to the component with maximum posterior, it is easy to obtain a clustering of the dataset X into M clusters, with one cluster corresponding to each mixture component. -->

## 7.1 Gauss 混合モデルの訓練のための EM<!--EM FOR GMM TRAINING-->

$X=\left\{x_n|x_n\in\mathcal{R}^{d},n=1,\ldots,N\right\}$ を，$M$ 個の成分を持つ GMM を使ってモデル化されるデータ点の集合とする：
<!-- Let $X=\left\{x_n|x_n\in\mathcal{R}^{d},n=1,\ldots,N\right\}$ denote a set of data points to be modeled using a GMM with M components: -->
$p(x)=\sum_{j=1}^{M}\pi_j N(x_n|\mu_j,\Sigma_j)$.
成分数 $M$ はあらかじめ指定されているものとする。
推定する混合パラメータのベクトル $\theta$ は，混合重みと各成分のパラメータから構成される，すなわち $\theta=\left\{\pi_j,\mu_j,\Sigma_j|j=1,\ldots,M\right\}$.
<!-- We assume that the number of components $M$ is specified in advance.
The vector $\theta$ of mixture parameters to be estimated consists of the mixing weights and the parameters of each component, i.e., $\theta=\left\{\pi_j,\mu_j,\Sigma_j|j=1,\ldots,M\right\}$. -->

パラメータ推定は，対数尤度の最大化によって達成できる。
<!-- Parameter estimation can be achieved through the maximization of the log-likelihood -->
$$
\theta_{ML}=\arg\max_{\theta} \log p(\mathbf{X};\mathbf{\theta}),\tag{tz62}
$$
ここで，独立で同一分布の観察を仮定すると，尤度は次のように書ける。
<!-- where assuming independent and identically distributed observations the likelihood can be written as -->
$$
p(X;\theta) =\prod_{n=1}^{N}p(X_n;\theta)=\prod_{n=1}^{N}\prod_{j=1}^{M}\pi_{j}N\left(x_n;\mu_j,\Sigma_{j}\right).\tag{tz63}
$$

図 5(b) のグラフィカルモデルから，各観測変数 $x_{n}\in X$ は，$x_n$ を生成するために使用された成分を表す隠れ変数 $z_n$ に対応することが明らかである。
この隠れ変数は，$x_n$ が混合成分 $j$ から生成された場合に $z_{jn}=1$，そうでない場合に $z_{jn}=0$ となるような，$M$ 個の要素を持つ二値ベクトル $z_{jn}$ を用いて表現することができる。
$z_{jn}=1$ は確率 $\pi_{j}$ と $\sum_{j=1}^{M}\pi_{j}=1$ であるので，$z_n$ は多項分布に従う。
$Z=\left\{z_n,n=1,\ldots,N\right\}$ を隠れ変数の集合とする。
すると $(Z|\theta)$ は次のように書かける:
<!-- From the graphical model in Figure 5(b) it is clear that to each observed variable $x_n\in X$ corresponds a hidden variable $z_n$ representing the component that was used to generate $x_n$.
This hidden variable can be represented using a binary vector with M elements $z_{jn}$, such that $z_{jn}=1$ if $x_n$ has been generated from mixture component $j$ and $z_{jn}=0$ otherwise.
Since $z_{jn}=1$ with probability $\pi_{j}$ and $\sum_{j=1}^{M}\pi_{j}=1$, then $z_n$ follows the multinomial distribution.
Let $Z=\left\{z_n,n=1,\ldots,N\right\}$ denote the set of hidden variables.
Then $p(Z|\theta)$ is written -->
$$
p(Z;\theta)=\prod_{n=1}^{N}\prod_{j=1}^{M}\pi_j^{z_{jn}}\tag{tz64}
$$
かつ<!--and-->
$$
p(X|z;\theta)=\prod_{n=1}^{N}\prod_{m=1}^{M}N\left(x_n;\mu_j,\Sigma_j\right)^{z_{jn}}\tag{tz65}
$$

前述したように，混合モデルの便利な問題は，(61) 式を用いて観察を与えられた隠れ変数の正確な事後値 $p\left(z_{jn}=1|x_n;\theta\right)$ を簡単に計算できることである。
したがって，厳密 EM アルゴリズムの適用が可能である。
<!-- As previously noted, the convenient issue with mixture models is that we can easily compute the exact posterior $p(z_{jn}=1|x_n;\theta)$ of the hidden variables given the observations using (61).
Therefore application of the exact EM algorithm is feasible. -->

より具体的には，$\theta^{(t)}$ が EM 反復 $t$ におけるパラメータベクトルを表すとすると，隠れ変数 $z_{jn}$ の事後 $p(z|x;\theta^{(t)})$ の期待値は次式で与えられる:
<!-- More specifically, if $\theta^{(t)}$ denotes the parameter vector at EM iteration $t$, the expected value of the posterior $p(z|x;\theta^{(t)})$ of hidden variables $z_{jn}$ is given as -->
$$\tag{tz66}
\left<z_{jn}^{(t)}\right>=\frac{\pi_{j}^{(t)} N\left(\right)x_{n};\mu_{j}^{(t),\sigma_{j}^{(t)}}}{\sum_{j=1}^{M}\pi_{j}^{(t)} N\left(\right)x_{n};\mu_{j}^{(t),\sigma_{j}^{(t)}}}
$$

上式は，$j=1,\ldots,M$ および $n=1,\ldots N$ に対して E ステップで実行されるべき計算を指定する。
<!-- The above equation specifies the computations that should be performed in the E-step for $j=1,\ldots,M$ and $n=1,\ldots N$. -->

事後分布 $p(Z|X;\theta(t))$ に対する完全対数尤度 $\log P(X,Z)$ の期待値は次式で与えられる。
<!-- The expected value of the complete log-likelihood $\log P(X,Z)$ with respect to the posterior $p(Z|X;\theta(t))$ is given by -->
$$\tag{tz67}\begin{aligned}
Q\left(\theta,\theta^{(t)}\right) &= \left<\log p\left(X,Z;\theta\right)\right>\\
&=\left<\log p\left(X|Z;\theta\right)\right>+\left<\log p\left(Z;\theta\right)\right>_{p\left(z|x;\theta^{(t)}\right)}\\
&=\sum_{n=1}^{N}\sum_{j=1}^{M}\left<z_{jn}^{(t)}\right>\log\pi_{j} +\sum_{n=1}^{N}\sum_{j=1}^{M}\log N\left(x_{n};\mu_j,\Sigma_{j}\right)\\
\end{aligned}$$

M ステップでは，期待完全対数尤度 Q はパラメータ $\theta$ に関して最大化される。
対応する偏導関数をゼロに等しくし，制約条件 $\sum_{j=1}^{M}\pi_{j}=1$ に対してラLagrange 乗数を用いると，M ステップの更新について以下の式が導ける：
<!-- In the M-step the expected complete log-likelihood Q is maximized with respect to the parameters $\theta$.
Taking the corresponding partial derivatives equal to zero and  us ing a Lagrange multiplier s for the constraint $\sum_{j=1}^{M}\pi_{j}=1$, we can derive the following equations for the updates of the M-step: -->
$$\tag{68}
\pi_{j}^{(t+1)}=\frac{1}{N}\sum_{n=1}^{N}\left<z_{jn}^{(t)}\right>$$
$$\tag{69}
\mu_{j}^{(t+1)}=\frac{\sum_{n=1}^{N}\left<z_{jn}^{(t)}\right>x_n}{\sum_{n=1}^{N}\left<z_{jn}^{(t)}\right> }$$
$$\tag{70}
\Sigma_{j}^{(t+1)}=\frac{\sum_{n=1}^{N}\left<x_{jn}^{(t)}\right>\left(x_{n}-\mu_{j}^{(t)}\right)\left(x_{n}-\sum_{j}^{(t)}\right)^{\top}}{\sum_{n=1}^{N}\left<z_{jn}^{(t)}\right>}$$

GMM 訓練のための上記の更新式は非常に簡潔で実装が簡単である。
これらは，EM の採用が尤度最大化問題の解決をどのように促進するかについての注目すべき例である。
<!-- The above update equations for GMM training are quite simple and easy to implement.
They constitute a notable example on how the employment of EM may facilitate the solution of likelihood maximization problems. -->

上記のアプローチで起こりうる問題は，図 6 に示す例のように，共分散行列が特異になる可能性があるという事実に関連している。
この図は，2 次元 (2-D) データ集合に対して，EM を適用して 20 個の成分を持つ GMM を訓練したときに得られた解の等高線プロットである。
GMM 成分の一部が特異であること，すなわち，それらの密度がデータ点の周りに集中し，いくつかの主軸に沿った分散が 0 になる傾向があることは明らかである。
GMM 学習のための典型的な ML アプローチのもう 1 つの欠点は，モデル選択，すなわち成分数の決定に使えないことである。
これらの問題に対する解決策は，Bayes GMM を使うことで得られるかもしれない。
<!-- A possible problem of the above approach is related to the fact that the covariance matrices may become singular, as shown in the example presented Figure 6.
This figure provides the contour plot of a solution obtained when applying EM to train a GMM with 20 components on a two-dimensional (2-D) dataset.
It is clear that some of the GMM components are singular, i.e., their density is concentrated around a data point and their variance along some principal axis tends to zero.
Another drawback of the typical ML approach for GMM training is that it cannot be used for model selection, i.e., determination of the number of components.
A solution to those issues may be obtained by using Bayesian GMMs. -->

## 7.2 変分 GMM 訓練<!--VARIATIONAL GMM TRAINING-->

### 7.2.1 完全 Bayesian GMM<!--FULL BAYESIAN GMM-->

$X=\left\{x_{n}\right\}$ を N 個の観察集合とし，各 $x_{n}\in\mathcal{R}^{d}$ を特徴ベクトルとする。
また，$p(x)$ を $M$ 個のガウス成分を持つ混合とする
<!-- Let $X=\left\{x_{n}\right\}$ be a set of N observations, where each $x_{n}\in\mathcal{R}^{d}$ is a feature vector.
Let also p(x) be a mixture with M Gaussian components -->
$$\tag{71}
p(x)=\sum_{j=1}^{M}\pi_{j} N(x;\mu_{j},T_{j})$$
ここで $\pi=\left\{\pi_{j}\right\}$  は混合係数 (重み)，$\mu=\left\{\mu_j\right\}$ は成分の平均 (中心)，$T=\left\{T_j\right\}$ は精度 (逆共分散) 行列である (Bayes GMM では共分散行列の代わりに精度行列を使用する方が便利であることに注意)。
<!-- where $\pi=\left\{\pi_{j}\right\}$ are the mixing coefficients (weights), $\mu=\left\{\mu_j\right\}$ the means (centers) of the components, and $T=\left\{T_j\right\}$ the precision (inverse covariance) matrices (it must be noted that in Bayesian GMMs it is more convenient to use the precision matrix instead of the covariance matrix). -->

Bayes 型 Gauss 混合モデルは，モデルパラメータ $\pi,\mu,T$ に事前分布を与えることで得られる。
典型的には共役事前分布が用いられ，$\pi$ に対しては Dirichlet, $(\mu,T)$ に対しては Gauss-Wishart 分布である。
パラメータ $\left\{\alpha_{j}\right\}$ を持つ $\pi$ の Dirichlet 事前分布は次式で与えられる。
<!-- A Bayesian Gaussian mixture model is obtained by imposing priors on the model parameters $\pi,\mu$ and $T$.
Typically conjugate priors are used, that is Dirichlet for $\pi$ and Gauss-Wishart for $(\mu,T)$.
The Dirichlet prior for π with parameters $\left\{\alpha_{j}\right\}$ is given by -->
$$
\text{Dir}\left(\pi|\alpha_1,\ldots,\alpha_{M}\right)=\frac{\Gamma\left(\sum_{j=1}^{M}\alpha_{j}\right)}{\prod_{j=1}^{M}\gamma\left(\alpha_{j}\right)}\prod_{j=1}^{M}\pi_{j}^{\alpha_{j}-1},
$$
ここで $\Gamma(x)$ はガンマ関数を表す。
通常，$\alpha_j$ はすべて等しい，すなわち $\alpha_j=\alpha_0, j=1,\ldots,M$.
<!-- where (x) denotes the Gamma function.
Usually, we assume that all αj are equal, i.e., αj = α0, j= 1, . . . , M. -->

$(\mu,T)$ の Gauss-Wishart 事前分布は $p(\mu,T)=\prod_{j=1}^{M}p(\mu_j,T_j)=p(\mu_j |T_j)p(T_j)$ であり，$p(\mu_j|T_j=N(\mu_j;\mu_0,\beta_0T_j)$ (パラメータ $\mu_0$ と $\beta_0$) であり，$p(T_j )$ は Wishart 分布である。
<!-- The Gauss-Wishart prior for (μ,T) is p(μ T) = Mj=1 p(μj,Tj)=Mj=1 p(μj |Tj )p(Tj ) , where p(μj |Tj =N(μj;μ0, β0 Tj ) (with parameters μ0 and β0) and p(Tj ) is the Wishart distribution -->
$$
W(T_j|\nu,V)=\frac{\left|T_j\right|^{(v-d-1)/2}\exp\text{tr}\left\{-\frac{1}{2}VT_j\right\}}{2^{vd/2}\pi^{d(d-1)/4}\left|V\right|^{-n/2}\prod_{i=1}^{d}\Gamma((v+1-i)/2)},
$$
パラメータ $\nu$ と $V$ はそれぞれ自由度とスケール行列を表す。
Wishart 分布は，Gamma 分布の多次元一般化であることに注意。
線形回帰では，Gauss-Gamma 事前分布を用い，独立の事前分布 $\alpha_i$ を仮定し，したがって，それらに独立の Gamma 事前分布を割り当てる。
しかし，ここでは，データ間に有意な相関があるかもしれないので，これらの相関を捕捉するために Wishart 事前分布を使うことができる。
<!-- with parameters ν and V denoting the degrees of freedom and the scale matrix respectively.
Notice that the Wishart distribution is the multidimensional generalization of the Gamma distribution.
In linear regression, we used the Gauss-Gamma prior, assuming independent precisions αi and thus assigning them independent Gamma prior distributions.
Here, however, because there may be significant correlations between data, we could use the Wishart prior to capture these correlations. -->

この Bayes GMM に対応するグラフィカルモデルを 図 7(a) に示す。
これは完全 Bayes  GMM であり，すべてのハイパーパラメータ (すなわち，事前分布のパラメータ $\alpha,\mu_0,\beta_0,\nu,V$) が事前に指定されている場合，モデルは推定されるべきパラメータを含まず，データが与えられた事後値 $p(h|x)$ を計算しなければならない隠れ確率変数 $h=(Z,\pi,\mu,T)$ だけを含む。
この場合，事後値が解析的に計算できないことは明らかであり，したがって，変分平均場 (16) を特定の Bayes モデルに適用することによって近似 $q(h)$ が計算される[27]。
<!-- The graphical model corresponding to this Bayesian GMM is presented in Figure 7(a).
This is a full Bayesian GMM and if all the hyperparameters (i.e., the parameters α,μ0, β0, ν and V of the priors) are specified in advance, then the model does not contain any parameter to be estimated, but only hidden random variables h=(Z,π,μ,T) whose posterior p(h|x) given the data must be computed.
It is obvious that in this case the posterior cannot be computed analytically, thus an approximation q(h) is computed by applying the variational mean field (16) to the specific Bayesian model [27]. -->

<div class="figcenter">
<img src="../figures/2008Tzikas_fig6.svg" width="39%">
<img src="../figures/2008Tzikas_fig7.svg" width="49%">
<div class="figcaption">

左 図 6. 20 の Gauss 成分を用いた EM ベースの GMM 訓練。
右 図 7.
(a) 完全ベイズ GMM のグラフィカルモデル。
(b) [27] のベイズ GMM のグラフィカルモデル。
2 つのモデルにおける $\pi$ の役割の違いに注目。
また，$\pi,\mu,\Sigma$ の事前分布のパラメータは固定されているので，それらは示されていない。
<!-- Figure 6. EM-based GMM training using 20 Gaussian components.
Figure 7.
(a) Graphical model for the full Bayesian GMM.
(b) Graphical model for the Bayesian GMM of [27].
Notice the difference in the role of π in the two models.
Also, the parameters of the priors on pi,mu,Sigma are fixed, thus they are not shown. -->
</div></div>

平均場近似では，$q$ は次の形式の積であると仮定する。
<!-- The mean-field approximation assumes q to be a product of the form -->
$$\tag{72}
q(h)=q_{z}(Z) q_{\pi}(\pi) q_{\mu T}(\mu, T).
$$
となり，解は (16) で与えられる。
必要な計算を行った結果，次のような密度の集合が得られる：
<!-- and the solution is given by (16).
After performing the necessary calculations, the result is the following set of densities: -->
$$\tag{73}
q_z(Z)=\prod_{n=1}^{N}\prod_{j=1}^{M}r_{jn}^{z_{jn}}$$

$$\tag{74}
q_{\pi}(\pi)=\text{Dir}\left(\pi|\left\{\lambda_{j}\right\}\right)$$

$$\tag{75}
q_{\mu T}(\mu,T)=\prod_{j=1}^{M}q_{\mu}(q_{\mu}\left(\mu_j|T_{j}\right) q_{T}\left(T_{j}\right)$$

$$\tag{76}
q_{\mu}(\mu_j|T)=\prod_{j=1}^{M}N\left(\mu_j;m_j,\beta_jT_j\right)
$$

$$\tag{77}
q_{T}=\prod_{j=1}^{M}W\left(T_j;n_j,U_j\right)
$$
そして，密度のパラメータ $(r_{jn},\lambda_j,m_j,\beta_j,\nu_j,U_j)$ を更新するための詳細な公式は [27] にある。
簡単な反復更新手順を用いて上記の方程式系を解くことにより，平均場制約の下での真の事後 $p(h|x)$ に対する最適な近似 $q(h)$ が得られる。
<!-- and the detailed formulas for updating the parameters (rjn, λj,mj, βj, ηj,Uj ) of the densities can be found in [27].
By solving the above system of equations using a simple iterative update procedure, we obtain an optimal approximation q(h) to the true posterior p(h |x) under the mean-field constraint. -->

Bayes モデリングにおける典型的なアプローチは，モデルのハイパーパラメータ $\alpha,\nu,V,\mu_0,\beta_0$ を指定し，情報量のない事前分布が定義されるようにすることである。
これらのパラメータを調整するために，アルゴリズムに M ステップを組み込むことは可能であるが，我々はこのアプローチに従う。
しかし，これは通常行われない。
<!-- The typical approach in Bayesian modeling is to specify the hyperparameters α, ν, V, μ0, and β0 of the model so that uninformative prior distributions are defined.
We follow this approach, although it would be possible to incorporate an M step to the algorithm, in order to adjust these parameters.
However, this is usually not followed. -->

完全 Bayes GMM の利点の 1 つは，事前分布を持たない GMM と比較して，Gauss 成分が 1 つのデータ点を担当するようになる ML アプローチでしばしば生じる特異解を許さないことである。
2 番目の利点は，交差検証法などの手法に頼ることなく，最適な成分数を直接決定するためにベイズ GMM を使用できることである。
しかしながら，Direchlet 事前分布は，成分の混合重みがゼロになって混合から除去されることを許さないので，この問題では完全 Bayes 混合分布の有効性は制限される。
また，この場合，最終的な結果は，事前に指定されなければならない事前分布のハイパーパラメータ (特に Direchlet 事前分布のパラメータ) に大きく依存する[13]。
特定のハイパーパラメータの集合に対して，混合成分の数 $M$ のいくつかの値に対して変分アルゴリズムを実行し，変分下界の最良の値に対応する解を保持することが可能である。
<!-- One advantage of the full Bayesian GMM compared to GMM without priors is that it does not allow the singular solutions often arising in the ML approach where a Gaussian component becomes responsible for a single data point.
A second advantage is that it is possible to use the Bayesian GMM for directly determining the optimal number of components without resorting to methods such as cross-validation.
However, the effectiveness of the full Bayesian mixture is limited for this problem, since the Dirichlet prior does not allow mixing weight of a component to become zero and to be eliminated from the mixture.
Also, in this case the final result depends highly on the hyperparameters of the priors (and especially of the parameters of the Dirichlet prior) that must be specified in advance [13].
For a specific set of hyperparameters, it is possible to run the variational algorithm for several values of the number M of mixture components and keep the solution corresponding to the best value of the variational
lower bound. -->

### 7.2.2 混合重みからの事前分布除去<!--REMOVING THE PRIOR FROM THE MIXING WEIGHTS-->

[28] では，Bayes GMM モデルのもう 1 つの例が提案されており，これは混合重み $\left\{\pi_j\right\}$ の事前分布を仮定せず，混合重みは確率変数ではなくパラメータとして扱われる。
このアプローチのグラフモデルは図 7(b) に描かれている。
<!--In [28], another example of a Bayesian GMM model has been proposed that does not assume a prior on the mixing weights {πj }, which are treated as parameters and not as random variables.
The graphical model for this approach is depicted in Figure 7(b). -->

このベイズ GMM (本研究の 2 人の著者の頭文字をとって CB モデルと呼ぶ)
 では，$\mu$ と $T$ に対してそれぞれ Gauss 事前分布と Wishart 事前分布が仮定される
<!-- In this Bayesian GMM, which we will call CB model from the initials of the two authors of this work, Gaussian and Wishart priors are assumed for μ and T, respectively -->

$$\tag{78}
p(\mu)=\prod_{j=1}^{M}N(\mu_j|0,\beta\mathbf{I})
$$

$$\tag{79}
p(T)=\prod_{j=1}^{M}W(T_j|\nu,\mathbf{V})
$$

この Bayes モデルは，(ある程度) 最適な成分数を推定することができる。
これは，隠れ変数 $h=\left\{Z,\mu,T\right\}$ を積分することによって得られる周辺尤度 $p(X;\pi)$ の最大化によって達成される。
<!-- This Bayesian model is capable (to some extent) to estimate the optimal number of components.
This is achieved through maximization of the marginal likelihood p(X;π) obtained by integrating out the hidden variables h = {Z,μ, T} -->
$$\tag{80}
p(X;\pi)=\int p\left(X,h;\pi\right)\,dh$$
パラメータとして扱う混合重み $\pi$ に関して。
変分近似は，対数周辺尤度 $$\tag{81}$$ の下界の最大化を提案する。
ここで $q(h)$ は事後 $p(h|X)$ を近似する任意の分布である。
注目すべき性質は，$F$ を最大化するとき，いくつかの成分がデータ空間の同じ領域に入る場合，この領域のデータが少ない成分で十分に説明できるようになると，モデルには冗長な成分を排除する (つまり，それらの $\pi_j$ を 0 にする) 強い傾向があることである。
その結果，混合成分間の競合は，モデル選択問題に対処するための自然なアプローチを示唆する： 多数の成分で初期化された混合を適合させ，競合によって冗長な成分を除去させる。
<!-- with respect to the mixing weights π that are treated as parameters.
The variational approximation suggests the maximization of a lower bound of the logarithmic marginal likelihood $$\tag{81}$$
where q(h) is an arbitrary distribution approximating the posterior p(h|X).  -->
<!-- A notable property is that during maximization of F, if some of the components fall in the same region in the data space, then there is strong tendency in the model to eliminate the redundant components (i.e., setting their πj equal to zero), once the data in this region are sufficiently explained by fewer components. Consequently, the competition between mixture components suggests a natural approach for addressing the model selection problem: fit a mixture initialized with a large number of components and let competition eliminate the redundant. -->

変分法に従って，我々の目的は対数周辺尤度 $\log p(X;\pi)$ の下界 $F$ を最大化することである。
<!-- Following the variational methodology, our aim is to maximize the lower bound F of the logarithmic marginal likelihood log p(X;π) -->
$$\tag{82}
F\left[q,\pi\right]=\sum_{z}\int q(Z,\mu,T)\log\frac{p(X,Z,\mu,T)}{q(Z,\mu,T)}du\,dT.
$$
ここで $q$ は事後分布 $p(Z,\mu,T|X;\pi)$ を近似する任意の分布である。
$F の最大化は，変分 EM アルゴリズムを用いて反復的に実行される。
各反復で 2 つのステップが行われる：まず $q$ に関する境界の最大化，次に $\pi$ に関する境界の最大化。
<!-- where q is an arbitrary distribution that approximates the posterior distribution p(Z,μ, T|X;π).
The maximization of F is performed in an iterative way using the variational EM algorithm.
At each iteration two steps take place: first maximization of the bound with respect to q and, subsequently, maximization of the bound with respect to π. -->

$q$ に関する最大化を実行するために，$q$ が次の形式の積であると仮定する平均場近似が採用されている(14)。
<!-- To implement the maximization with respect to q, the mean-field approximation has been adopted (14) that assumes q to be a product of the form  -->
$$
q(h)=q_z(Z)\,q_\mu(\mu)\,q_T(T).\tag{83}
$$.
(16)で必要な計算を行った結果，以下のような密度が得られた：
<!-- After performing the necessary calculations in (16), the result is the following set of densities: -->
$$
q_z(Z)=\prod_{n=1}^{N}\prod_{j=1}^{M}r_{jn}^{z_{jn}}\tag{84}$$
$$
q_\mu(\mu)=\prod_{j=1}^{M}N(\mu_j|m_j,S_j)\tag{85}$$
$$
q_T(T)=\prod_{j=1}^{M}W(T_j|n_j,U_j)\tag{86}$$
ここで，密度のパラメータは次のように計算できる。
<!-- where the parameters of the densities can be computed as -->
$$\begin{aligned}
r_{jn} &=\frac{\tilde{r}_{jn}}{\sum_{k=1}^{M}\tilde{r}_{jn}} &&& (tz:87)\\
\tilde{r}_{jn} &=\pi_{j}\exp\left\{\frac{1}{2}\left<\log\left|T_j\right|\right>-\frac{1}{2}\text{tr}\left\{\left<T_j\right>\left(x_nx_n^{\top}-x_n\left<u_j\right>^{\top}+
\left<u_j\right>x_n^{\top}+\left<\mu_j\mu_j^{\top}\right>\right)\right\}
\right\} &&& (tz:88)\\
m_J &= S_j^{-1}\left<T_j\right>\sum_{n=1}^{N}\left<z_{jn}\right>x_n &&& (tz:89)\\
S_j & =\beta\mathbf{I}+\left<T_j\right>\sum_{n=1}^{N}\left<z_{jn}\right> &&& (tz:90)\\
n_J & =\nu+\sum_{n=1}^{N}\left<z_{jn}\right> &&& (tz:91)\\
U_j&=V+\sum_{n=1}^{N}\left<z_{jn}\right>\left(x_nx_n^{\top}-x_n\left<u_j\right>^{\top}+\left<u_j\right>x_n^{\top}+\left<u_ju_j^{\top}\right>\right)&&&(tz:92)\\
\end{aligned}
$$

<!-- $$
r_{jn}=\frac{\tilde{r}_{jn}}{\sum_{k=1}^{M}\tilde{r}_{jn}}\tag{87}
$$
$$
\tilde{r}_{jn}=\pi_{j}\exp\left\{\frac{1}{2}\left<\log\left|T_j\right|\right>-\frac{1}{2}\text{tr}\left\{\left<T_j\right>\left(x_nx_n^{\top}-x_n\left<u_j\right>^{\top}+
\left<u_j\right>x_n^{\top}+\left<\mu_j\mu_j^{\top}\right>\right)\right\}
\right\}
\tag{88}$$
$$
m_J=S_j^{-1}\left<T_j\right>\sum_{n=1}^{N}\left<z_{jn}\right>x_n\tag{89}$$
$$
S_j=\beta\mathbf{I}+\left<T_j\right>\sum_{n=1}^{N}\left<z_{jn}\right>\tag{90}$$
$$
n_J=\nu+\sum_{n=1}^{N}\left<z_{jn}\right>\tag{91}$$
$$
U_j=V+\sum_{n=1}^{N}\left<z_{jn}\right>\left(x_nx_n^{\top}-x_n\left<u_j\right>^{\top}+\left<u_j\right>x_n^{\top}+\left<u_ju_j^{\top}\right>\right)\tag{92}$$ -->

上式で用いた $q(h)$ に関する期待値は，次式を満たす：  
$\left<T_j\right>=\eta_jU^{-1}_j,\left<\log\left|T_j\right|\right>=\sum_{i=1}^{d}\Psi(0.5(\eta_j+1-i)) +d\ln 2 - \ln\left|U_j\right|,\left<\mu_j\right>=m_j,\left<\mu_j\mu^{\top}_j\right>=S^{-1}_{j}+m_jm^{\top}_j$ [ここで，$\psi$ は次式で定義されるディガンマ関数を示す。
$d/dx\ln\Gamma(x)=\Gamma^{\prime}(x)/\Gamma(x)$] かつ，$z_{jn}=r_{jn}$。
密度が期待値を通して結合していることがわかるので，パラメータの反復推定が必要である。
しかし実際には，変分 E ステップでは 1 回のパスで十分である。
<!-- The expectations with respect to q(h) used in the above equations satisfy the equations:  
Tj = ηjU−1j, log |Tj| = di=1 ψ(0.5(ηj +1 − i)) +d ln 2 − ln |Uj |,  μj = mj,  μjμTj =S−1j+ mjmTj [Here ψ denotes the digamma function, defined as
d/dxln  (x) =   (x)/ (x)] and  zjn  = rjn. It can be observed that the densities are coupled through their expectations, thus an iterative estimation of the parameters is needed.
However, in practice a single pass seems to be sufficient for the variational E-step -->

$q$ に関して $F$ を最大化した後，訓練法の各反復の第 2 ステップでは，$\pi$ に関して $F$ を最大化する必要があり，次のような変分 M ステップの簡単な更新式が導かれる：
<!-- After the maximization of F with respect to q, the second step of each iteration of the training method requires maximization of F with respect to π, leading to the following simple update equation for the variational M-step: -->
$$\tag{93}
\pi_j=\frac{\sum_{n=1}^{N}r_{jn}}{\sum_{k=1}^{M}\sum_{n=1}^{N}r_{kn}}$$
上記の変分 EM 更新方程式は反復的に適用され，変分境界の局所最大値に収束する。
最適化中に混合係数の一部がゼロに収束し，対応する成分が混合物から除去される。
このようにして複雑さの制御が達成される。
これは，$\mu$ と $T$ に関する事前分布が重複する成分に罰則を課しているからである。
定性的には，変分境界は 2 つの項の和として書くことができる。
1 つは尤度項 (データ適合の質に依存する) で，もう 1 つは複雑なモデルに罰則を与える事前分布による罰則項である。
<!-- The above variational EM update equations are applied iteratively and converge to a local maximum of the variational bound.
During the optimization some of the mixing coefficients converge to zero thus the corresponding components are eliminated from the mixture.
In this way complexity control is achieved.
This happens because the prior distribution on μ and T penalizes overlapping components.
Qualitatively speaking, the variational bound can be written as a sum of two terms: the first one is a likelihood term (that depends on the quality of data fitting) and the other is a penalty term due to the priors that penalizes complex models. -->

図 8 は，すでに図 6 で示した 2 次元データセットを用いたこの手法の性能の例示である。
この手法は 20 成分から始まり，反復回数が増えるにつれて，成分数は徐々に減少し(いくつかの $\pi_j$ はゼロになる)，最終的にこのデータセットに対する良好な GMM モデルが達成される。
また，共分散行列に事前分布が存在することで，図 6 に示された事前分布なしの GMM 解とは対照的に，特異解に到達しないことが観察される。
<!-- Figure 8 provides an illustrative example of the performance on this method using the 2-D dataset already presented in Figure 6.
The method starts with 20 components and, as the number of iterations increases, the number of components gradually decreases (some πj become zero) and, finally, a good GMM model for this dataset is attained.
It can also be observed, that the existence of the prior on the covariance matrices, does not allow to reach singular solutions in contrast to the GMM solution without priors presented in Figure 6. -->

<div class="figcenter">
<img src="../figures/2008Tzikas_fig8.svg">
<div class="figcaption">

図 8 [28] のモデルを用いた変分ベイズ GMM 学習。
(a) 20 ガウス成分による初期化，(b), (c) EM 反復中のモデル進化，(d) 最終解。
特異点の回避に注目。
<!-- Fig. 8 Variational Bayesian GMM training using the model presented in [28].
(a) Initialization with 20 Gaussian components, (b), (c) model evolution during EM iterations, and (d) final solution.
Notice the avoidance of singularities. -->
</div></div>

一般に，CB は，成分がよく分離されている場合に良好な性能を示す効果的な手法である。
しかし，その性能は，精度行列に課される Wishart 事前分布のスケール行列 V の指定に敏感である。
上記の混合モデルを構築するための漸進的手法が [29] で提案されている。
各ステップにおいて，学習は特定の混合成分jが占めるデータ領域に制限されるため，精度行列 $T_j$ に基づいて局所的な精度事前分布を指定することができる。
この動作を実現するために，図 7 の生成モデルに対して，成分の下位集合でのみ競合を制限する修正を加えた。
<!-- In general, the CB constitutes an effective method exhibiting good performance in the case where the components are well separated.
However, its performance exhibits sensitivity on the specification of the scale matrix V of the Wishart prior imposed on the precision matrix.
An incremental method for building the above mixture model has been proposed in [29].
At each step, learning is restricted in the data region occupied by a specific mixture component j, thus a local precision prior can be specified based on the precision matrix Tj.
In order to achieve this behavior, a modification to the generative model of Figure 7 was made that restricts the competition in a subset of the components only. -->

# 8. まとめ<!-- # 8. SUMMARY-->

EM アルゴリズムは，ML 推定に多くの利点をもたらす反復法である。
尤度関数の直接最適化が困難な問題に対して，局所収束が保証された単純な反復解を提供する。
多くの場合，共分散行列が正定値，確率ベクトルが正で和が 1 など，推定パラメータに対するいくつかの制約を満たす解を提供する。
さらに，EM の適用には尤度関数を明示的に評価する必要はない。
<!-- The EM algorithm is an iterative methodology that offers a number of advantages for ML estimation.
It provides simple iterative solutions, with guaranteed local convergence, for problems where direct optimization of the likelihood function is difficult.
In many cases it provides solutions that satisfy several constraints for the estimated parameters, for example covariance matrices are positive definite, probability vectors are positive, and sum to one, etc.
Furthermore, the application of EM does not require explicit evaluation of the likelihood function.-->

しかし EM アルゴリズムを適用するためには，観察が与えられたときの隠れ変数の事後知識が必要である。
これは，複雑なベイズモデルには EM を適用できないので重大な欠点である。
しかし，複雑なベイズモデルは，適切に構築されれば，データ生成機構の顕著な特性をモデル化する能力を持ち，困難な問題に対して非常に良い解を提供するため，非常に有用である。
<!-- However, to apply the EM algorithm we must have knowledge of the posterior of the hidden variables given the observations.
This is a serious drawback since the EM cannot be applied to complex Bayesian models.
However, complex Bayesian models can be very useful since, if properly constructed, they have the ability to model salient properties of the data generation mechanism and provide very good solutions to difficult problems. -->

変分法は EM アルゴリズムのこの欠点を改善するために，信号処理界で人気を集めている反復アプローチである。
この方法論によれば，観測値が与えられたときの隠れ変数の事後的近似値が使用される。
この近似に基づき，尤度関数の下界を最大化することでベイズ推定が可能となり，局所収束が保証される。
この方法論は，複雑なグラフモデルの推論を可能にし，ある場合には EM で解ける単純なモデルと比較して，著しい改善をもたらす。
<!-- The variational methodology is an iterative approach that is gaining popularity within the signal processing community to ameliorate this shortcoming of the EM algorithm.
According to this methodology an approximation to the posterior of the hidden variables given the observations is used.
Based on this approximation, Bayesian inference is possible by maximizing a lower bound of the likelihood function which also guarantees local convergence.
This methodology allows inference in the case of complex graphical models, that in certain cases provide significant improvements as compared to simpler ones that can be solved via the EM.-->

この問題は，信号処理アプリケーションの 2 つの基本的な問題である線形回帰とガウス混合モデリングの文脈で，本論文で実証された。
具体的には，線形回帰の文脈では，変分法によって解かれた複雑なベイズモデルが，局所的な信号特性をよりよく捉え，信号の不連続領域でのリンギング ringing を回避できることを実証した。
ガウス混合モデリングの文脈では，変分法によって解かれたモデルは，特異点を回避し，モデルの構成要素の数を推定することができた。
これらの結果は，変分法が信号処理アプリケーションを長い間悩ませてきた難問に解を提供する力を持つことを示すものである。
この方法の主な欠点は (少なくとも今のところ) 使用される境界の厳密さを評価できる結果がないことである。
<!-- This issue was demonstrated in this article within the context of linear regression and Gaussian mixture modeling, which are two fundamental problems for signal processing applications.
More specifically, we demonstrated that complex Bayesian models that were solved by the variational methodology, in the context of linear regression were able to better capture local signal properties and avoid ringing in areas of signal discontinuities.
In the context of Gaussian mixture modeling, the models solved by the variational methodology were able to avoid singularities and to estimate the number of the model components.
These results demonstrate the power of the variational methodology to provide solution to difficult problems that have plagued signal processing applications for a long time.
The main drawback of this methodology is the lack of results that allow (at least for the time being) assessing the tightness of the bound that is used.  -->


# B.1 ELBO: Evidence Lower BOund
from https://www.cs.princeton.edu/archive/fall11/cos597C/lectures/variational-inference-i.pdf

変分推論では潜在変数空間 $\mathcal{Z}$ 上での $z$ の確率密度を考える。
ここでの目標は，KL ダイバージェンスの意味で最良の以下の式を最適化することである:

$$
q^{\star}(z)=\operatorname{argmin}_{g(x)\in\mathcal{Z}} \text{KL}(q(z)\vert\vert p(z\vert x))\tag{eq:blei10}.
$$

この解が得られれば，$q^{\star}(\cdot)$ は，条件付近似となる。

変分推論では，潜在変数に対する密度の系列 $\mathcal{Z}$ を指定する。
各 $q(z\in\mathcal{Z})$ は，正確な条件に近似する候補である。
我々の目的は，最良の候補を見つけることであり，正確な条件に KL ダイバージェンスで最も近いものを見つけることである。
推論は，以下の最適化問題を解くことになる。

<!-- In variational inference, we specify a family $\mathcal{Z}$ of densities over the latent variables.
Each $q\of{z}\in\mathcal{Z}$ is a candidate approximation to the exact conditional.
Our goal is to find the best candidate, the one closest in KL-divergence to the exact conditional Inference  now amounts to solving the following optimization problem, -->

$$
q^{\star}(z)=\arg\min_ {g(x)\in\mathcal{Z}}\text{KL}(q(z)\vert\vert p(z\vert x))\tag{eq:blei11}.
$$

見つかった $q^{\star}(\cdot)$ は，$\mathcal{Z}$ 族の中で，条件式の最適な近似値となる。
分布族の複雑さは，この最適化の複雑さを決定する。
<!-- Once found, $q^{*}\of{\cdot}$ is the best approximation of the conditional, within the family $\mathcal{Z}$.
The complexity of the family determines the complexity of this optimization. -->

しかし，この目的は $p(x)=\int p(z,x)\,dz$ という式で $\log p(x)$ の証拠を計算する必要があるため，計算できない。
(証拠を計算するのが難しいからこそ，そもそも近似推論に訴えるのである)。
なぜかというと KL ダイバージェンスを思い出せば:
<!-- However, this objective is not computable because it requires computing the evidence $\log p\of{x}$ in Equat
ion $p\of{x}=\int p\of{z,x}dz$.
(That the evidence is hard to compute is why we appeal to approximate inference in the first place.)
To see why, recall that KL divergence is:  -->

$$
\text{KL}(q(z)\vert\vert p(z\vert x)) =\mathbb{E}(\log q(z)) - \mathbb{E}(\log p(z\vert x))
$$

ここで，すべての期待値は $q(z)$ を基準にしています。
ここで，条件式を展開する。
<!-- where all expectations are taken with respect to $q\of{z}$.
Expand the conditional, -->

$$
\text{KL}(q(z)\vert\vert p(z\vert x)) =\mathbb{E}(\log p(z)) - \mathbb{E}(\log p(z,x)) +\log p(x).\tag{eq:blei12}
$$

これにより $\log p(x)$ への依存性が明らかになった。
<!-- This reveals its dependence on $\log p\of{x}$. -->


KL を計算することができないので，定数を追加した上で KL と同等の代替目的を最適化する。
<!-- Because we cannot compute the $\operatorname{KL}$, we optimize an alternative objective that is equivalent to the $\operatorname{KL}$ up to an added constant, -->
$$
\text{ELBO}(q)=\mathbb{E}(\log p(z,x)) - \mathbb{E}(\log p(z)).\tag{eq:blei13}
$$

この関数は **evidence lower bound (ELBO)** と呼ばれる。
ELBO は 式(\ref{eq:blei12}) の **負の KL ダイバージェンス** に $q(z)$ に対する定数である **$\log p(x)$** を加えたものである。
ELBO を最大化することは KL ダイバージェンスを最小化することと同等である。
<!-- This function is called the \strong{evidence lower bound (ELBO)}.
The ELBO is the \strong{negative KL divergence} of Equation (\ref{eq:blei12}) plus \warn{$\log p\of{x}$}, which is a constant with respect to $q\of{z}$.
Maximizing the ELBO is equivalent to minimizing the $\operatorname{KL}$ divergence. -->

ELBO を調べることで， 最適な変分密度を直感的に理解することができる。
ELBO は，データの期待対数尤度と，事前 の $p(z)$ と $q(z)$ の間の KL ダイバージェンス の和と書き換えられる。
<!-- examining the ELBO gives intuitions about the optimal variational density.
We rewrite the ELBO as a sum of the expected log likelihood of the data and the KL divergence between the prior $p\of{z}$ and $q\of{z}$, -->
$$
\begin{aligned}
\text{ELBO}(q) &= \mathbb{E}(\log p(z)) + \mathbb{E}(\log p(x\vert z)) - \mathbb{E}(\log q(z))\\
               &= \mathbb{E}(\log p(x\vert z)) -\text{KL}(q(z)\vert\vert p(z)).\\
\end{aligned}
$$

この目的語は $q(z)$ がどのような値に質量を置くことを促すのだろうか？
第 1 項は，期待尤度で， 観測データを説明する潜在変数の構成に質量を置く密度を促す。
第 2 項は，変分密度と事前分布の間の負の発散で，事前分布に近い密度を奨励する。
このように，変分目標は，尤度と事前分布の通常のバランスを反映している。
<!-- Which values of $z$ will this objective encourage $q\of{z}$ to place its mass on?
The first term is an expected likelihood; it encourages densities that place their mass on configurations of the latent variables that explain the observed data.
% The second term is the negative divergence between the variational density and the prior; it encourages dens
ities close to the prior.
Thus the variational objective mirrors the usual balance between likelihood and prior. -->

ELBO のもう一つの特性は，任意の $q(z)$ に対して (対数) 証拠 である $\log p(x)\ge\text{ELBO}(q)$ を下界にすることである。
これが ELBO の名前の由来である。
これを見るには 式(eq:blei12) と(eq:blei13) が次のような証拠の表現を与えることに注意。
<!-- Another property of the $\operatorname{ELBO}$ is that it lower-bounds the (log) evidence, $\log p\of{x}\ge\text{ELBO}\of{q}$ for any $q\of{z}$.
This explains the name.
To see this notice that Equations (\ref{eq:blei12}) and (\ref{eq:blei13}) give the following expression of the evidence, -->
$$
\log p(x)=\text{KL}(q(z)\|p(z|x)) + \text{ELBO}(q).\tag{eq:blei14}
$$

そして，この境界は $\text{KL}(\cdot)\ge0$ [@1951KullbackLeibler] という事実から導かれる。
変分推論のオリジナル文献では，これは **Jensen's inequality** [@Jordan1999] によって導かれた。

<!--
The bound then follows from the fact that $\text{KL}(\cdot)\ge0$ [@1951KullbackLeibler].
In the original literature on variational inference, this was derived through **Jensen's inequality**[@Jordan1999].
-->

<!--
ここで $s$ はグラフのすべてのノードを表し $h$ と $e$ はそれぞれ隠れノードと証拠ノードを表す $s$ の分離した部分集合である。
条件付き確率 $p(h|e)$ を近似したい。
ここでは，条件付き確率分布の近似族である $q(h|e,\lambda)$ を導入する。
$q$ を表すグラフは， 一般に $p$ を表すグラフと同じではなく，一般には下位グラフである。
近似分布 $q$ の中から，変分パラメータに関する **カルバック・ライブラー (KL)ダイバージェンス** $\text{KL}(q||p)$ を最小にすることで，特定の分布を選択する。
-->
<!--
More formally, let $p\of{s}$ represent the joint distribution on the graphical model of interest, where as before $s$ represents all of the nodes of the graph and $h$ and $e$ are disjoint subsets of $s$   representing  the hidden nodes and the evidence nodes, respectively.
We wish to approximate the conditional probability $p\of{h\given{e}}$.
We introduce an approximating family of conditional probability distributions, $q\of{h\given{e,\lambda}}$, where $\lambda$ are variational parameters.
The graph representing $q$ is not generally the same as the graph representing $p$, generally it is a sub-graph.
From the family of approximating distributions $q$, we choose a particular distribution by minimizing the \strong{Kullback-Leibler} (KL) divergence, $\KL{q}{p}$, with respect to the variational parameters:
-->


$$
\lambda^{\star}=\operatorname{argmin}_{\lambda} D_{\text{KL}}{q(h|e,\lambda)\|p(h\vert e}),\tag{eq:jordan41}
$$

ここで、任意の確率分布 $q(s)$ と $p(s)$ に対して，カルバック・ライブラーダイバージェンスは以下のように定義される:
<!-- where for any probability distributions $q\of{s}$ and $p\of{s}$ the KL-divergence is defined as follows: -->

$$
D_ {\text{KL}} (q\vert\vert p) = \sum_ {s} q(s)\log\frac{q(s)}{p(s)}.\tag{eq:jordan42}
$$

変分パラメータ $\lambda^{\star}$ の最小値は，特定の分布 $q(h\vert e,\lambda^{\star})$ を定義し， これを $q(h\vert e,\lambda)$ 族における $p(h\vert e)$ の最良の近似値として扱う。
<!-- he minimizing values of the variational parameters, $\lambda^{*}$, define a particular distribution, $q\of{h\given{e,\lambda^{*}}}$, that we treat as the best approximation of $p\of{h\given{e}}$ in the family $q\of{h\given{e,\lambda}}$, -->

KL ダイバージェンスを近似精度の指標として用いる単純な理由の一つは，KL ダイバージェンスが近似 $q(h\vert e,\lambda)$ 族における証拠 $p(e)$ の確率 (すなわち尤度) の最良の **下界** をもたらすからである。
実際 $p(e)$ の対数を **イェンセンの不等式** を用いて以下のように束縛する。
<!-- One simple justification for using the KL divergence as a measure of approximation accuracy is that it yields the best \strong{lower bound} on the probability of the evidence $p\of{e}$ (i.e., the likelihood) in the family of approximations $q\of{h\given{e},\lambda}$.
Indeed, we bound the logarithm of $p\of{e}$ using \strong{Jensen's inequality} as follows: -->

$$
\begin{aligned}
\log p(e) &= \log\sum_ {H} p(h,e) \\
          &= \log\sum_ {h} q(h\vert e) \frac{p(h,e)}{q(h\vert e)}\\
          &\ge \sum_{h} q(h\vert e) \log\left[\frac{p(h,e)}{q(h\vert e)}\right].\\
\end{aligned}\tag{eq:jordan43}
$$


ELBO と $\log p(x)$ の関係から，モデル選択基準として変分境界を用いることになった。
これは，混合モデル (Ueda and Ghahramani, 2002; McGrory and Titterington, 2007) やより一般的なモデル (Beal and Ghahramani, 2003) で検討されている。
その前提は，境界が周辺尤度の良い近似であり，モデルを選択するための根拠となることである。
これは，実際にはうまくいくこともあるが，境界に基づいて選択することは，理論的には正当化されない。
他の研究では，対数予測密度の変分近似を用いて，交差妥当性に基づくモデル選択に VI を使用している (Nott et al.2012)。
<!-- The relationship between the ELBO and $\log p\of{x}$ has led to using the variational bound as a model selec tion criterion.
This has been explored for mixture models (Ueda and Ghahramani, 2002; McGrory and Titterington, 2007) and more generally (Beal and Ghahramani, 2003).
The premise is that the bound is a good approximation of the marginal likelihood, which provides a basis for selecting a model.
Though this sometimes works in practice, selecting based on a bound is not justified in theory.
Other research has used variational approximations in the log predictive density to use VI in cross-validation based model selection (Nott et al., 2012). -->

最後に，多くの読者は 式(eq:blei13) の ELBO の第 1 項が EM アルゴリズム [@Dempster1977] によって最適化された期待完全対数尤度であることに気づくだろう。
EM アルゴリズムは潜在変数を持つモデルの最尤推定値を求めるために設計されたものである。
このアルゴリズムは，ELBO が対数尤度 $\log p(x)$ (すなわち 対数尤度 $\log p(x)$) に等しいという事実を利用している。
($q(z)=p(z\vert x)$ のとき ELBO は対数尤度 $\log p(x)$ (すなわち対数証拠) に等しいという事実を利用している。
EM は $p(z\vert x0$ に従った期待される完全な対数尤度の計算 (**E ステップ**) と，モデルパラメータに関する最適化（**M ステップ**) を交互に行う。
変分推論とは異なり，EM は $p(z\vert x)$ の下での期待値が計算可能であることを前提とし，それ以外の難しいパラメータ推定問題に使用する。
EM とは異なり，変分推論は固定モデルのパラメータを推定するものではなく，古典的なパラメータが潜在変数として扱われるベイズの設定でよく使われる。
変分推論は，潜在変数の正確な条件を計算できないモデルに適用される。
<!--
Finally, many readers will notice that the first term of the ELBO in Equation (\ref{eq:blei13}) is the expected complete log-likelihood, which is optimized by the EM algorithm~\cite{Dempster1977}.
The EM algorithm was designed for finding maximum likelihood estimates in models with latent variables.
It uses the fact that the ELBO is equal to the log likelihood log $p\of{x}$ (i.e., the log evidence) when $q\of{z}=p\of{z\given{x}}$.
EM alternates between computing the expected complete log likelihood according to $p\of{z\given{x}}$ (\strong{the E step}) and optimizing it with respect to the model parameters (\strong{the M step}).
Unlike variational inference, EM assumes the expectation under $p\of{z\given{x}}$ is computable and uses it in otherwise difficult parameter estimation problems.
Unlike EM, variational inference does not estimate fixed model parameters--it is often used in a Bayesian setting where classical parameters are treated as latent variables.
Variational inference applies to models where we cannot compute the exact conditional of the latent variables.
-->


# C. Probabilistic Machine Learning
- from 2016Blei NIPS, slide 13

* 確率モデルは，潜在変数 $z$ と観測変数 $x$ の結合分布 $p(z,x)$ である。
* 未知数についての推論は，観測値が与えられたときの潜在変数の条件付き分布である **事後分布** を通して行われる。

<!-- * probabilistic model is a joint distribution of hidden variables $z$ and observed variables $x$, $p(z,x)$
* Inference about the unknowns is through the \strong{posterior}, the conditional distribution of the hidden variables given the observations
-->

$$
p(z\vert x)=\frac{p(z,x)}{p(x)}.
$$

* ほとんどの興味深いモデルにおいて，分母は扱いにくい。

**近似事後推論** に訴える。

<!-- * For most interesting models, the denominator is not tractable.
We appeal to **approximate posterior inference**
-->

<center>
<div class="fig">
<img src="../figures/2016Blei_NIPS_VI.svg" width="49%">
<div class="figcaption">
Schematic graph of VI
</div>
</div>
</center>

* VIは **推論を最適化に変える**。
* 潜在変数に対する分布の**変数族**を仮定する。

<!-- * VI turns **inference into optimization**
* Posit a **variational family** of distributions over the latent variables,
-->

$$
q(z;\nu)
$$

* **変動パラメータ $\nu$** を正確な事後処理に (KL で) 近づけるように適合させる。
(EP, BP などのアルゴリズムにつながる代替ダイバージェンスもある)。

<!-- * Fit the **variational parameters $\nu$**  to be close (in KL) to the exact posterior.
(There are alternative divergences, which connect to algorithms like EP, BP, and others.)
-->

## C.1 歴史
<!-- ## History -->

* 変分推論は，統計物理学の考え方を確率推論に応用したものである。
おそらく 80年代後半に Peterson と Anderson  (1987) が平均場近似法を用いてニューラルネットワークを適合させたのが始まりと思われる。
* このアイデアは 1990 年代初頭に Jordan 研究室 (Tommi Jaakkola, Lawrence Saul, Zoubin Gharamani) によって取り上げられ，多くの確率的モデルに一般化された (総説論文は Jordan et al.1999)。
* 並行して Hinton and Van Camp (1993) はニューラルネットワークのための平均ば近似を開発した。
Neal and Hinton (1993) はこのアイデアを EM アルゴリズムに結びつけ，さらに混合エキスパートモデルに対する変分法  (Waterhouse et al., 1996) や HMM (MacKay,1997) につなげた。

<!--* Variational inference adapts ideas from statistical physics to probabilistic inference.
Arguably, it began in the late eighties with Peterson and Anderson (1987), who used mean-field methods to fit a neural network.
* This idea was picked up by Jordan’s lab in the early 1990s—Tommi Jaakkola, Lawrence Saul, Zoubin Gharamani—who generalized it to many probabilistic models. (A review paper is Jordan et al., 1999.)
* In parallel, Hinton and Van Camp (1993) also developed mean-field for neural networks.
Neal and Hinton (1993) connected this idea to the EM algorithm, which lead to further variational methods for mixtures of experts (Waterhouse et al., 1996) and HMMs (MacKay, 1997).
-->

## C.2 今日
<!-- ## Today-->

* 現在，変分推論に関する新しい研究が盛んに行われており，スケーラブルにし，導出を容易にし，より速く，より正確にし，より複雑なモデルやアプリケーションに適用している。
* 現代の変分推論は，確率的プログラミング，強化学習，ニューラルネットワーク，凸最適化，ベイズ統計学，そして無数のアプリケーションなど，多くの重要な領域に関連している。
* 今日の我々の目標は，基本を学び，新しいアイデアを説明し，新しい研究のオープンな領域を提案することである。

<!-- * There is now a flurry of new work on variational inference, making it scalable, easier to derive, faster, more accurate, and applying it to more complicated models and applications.
* Modern VI touches many important areas: probabilistic programming, reinforcement learning, neural networks, convex optimization, Bayesian statistics, and myriad applications.
* Our goal today is to teach you the basics, explain some of the newer ideas, and to suggest open areas of new research.
-->
