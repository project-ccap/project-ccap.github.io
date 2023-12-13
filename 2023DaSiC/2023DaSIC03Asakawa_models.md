---
title: "DaSiC7 (2023) 発表資料"
author: 浅川伸一
layout: default
codemirror_mode: python
codemirror_mime_type: text/x-cython
---

<!-- <link href="asamarkdown.css" rel="stylesheet"></link> -->

[DaSiC 7 (2023)](https://sites.google.com/view/dasic7-2023) Linguistics and Data Science in Collaboration 発表資料

## 機械学習モデル

## 目次

1. 符号化器‐復号化器モデル
2. Transformer
3. 2021JNPS
4. 微調整と再び双対性について

## 生成 AI の性能向上

<center>
<img src="/figures/2021Brown_GPT3_fig3_13.jpg" width="77%">
<div style="width:88%;background-color:lavender;text-align:left;">

ニュース記事がモデルによって生成されたものであるかどうかを識別する人間の能力 (正しい割り当てと中立でない割り当ての比率で測定) は，モデルサイズが大きくなるほど低下する。
意図的に悪い対照モデル (出力のランダム性が高い無条件 GPT-3 小型モデル) の出力に対する精度を上部の破線で示し，ランダムな確率 (50 %) を下部の破線で示す。
ベストフィットの線は 95 %信頼区間を持つべき乗則である。[Brown+2021, arXiv:2005.14165](https://arxiv.com/abs/2005.14165/) Fig. 3
<!-- #### Figure 3.13: People’s ability to identify whether news articles are model-generated (measured by the
 ratio of correct assignments to non-neutral assignments) decreases as model size increases.
Accuracy on the outputs on the deliberately bad control model (an unconditioned GPT-3 Small model with higher
output randomness) is indicated with the dashed line at the top, and the random chance (50%) is indicated with
  the dashed line at the bottom. Line of best fit is a power law with 95% confidence intervals. -->
</div></center>


## Modeling

1. 記述モデル description model
箱と矢印モデルなど，質的予測
3. データ適合モデル data-fitting model
LDA と SVM との違いにあらわれている
5. アルゴリズムモデル algorithm model

<div class="figcenter">
<img src="/figures/1999Levelt_blueprint.jpg" width="49%">
<img src="/figures/1885LichtheimFig1.png" width="29%">
</div>


## 1.1 従来モデル
Dell モデルは，2 段階相互活性モデルであり，意味層，語彙層，音素層の 3 層からなるニューラルネットワークモデルである。
従来モデル (以下 Dell モデルと表記する) を図 1 (Foygel&Dell, 2000) に示した。
以下では，意味層を S 層，語彙層を L 層，音素層を P 層と表記する。
Dell モデルにおいては，ニューラルネットワークの特徴の一つである学習に基づくパラメータの調整は行われない。
このため Dell モデルを構成する各処理ユニット間の結合係数は一定である。
各層のユニット数は，S 層 10，L 層 5，P 層 9 個である。
Dell モデルの動作を略述すると以下のようになる:
シミュレーションは離散時間で行われ，時刻 $t=[1,\ldots,16]$ の 16 時刻である。
時刻 $t_1$ で S 層には 3 つのユニットが活性化され，他の 7 つのユニットは不活性である。
S, L, P 層の $i$ 番目のユニットの活性値を，それぞれ $x_i, x\in\left\{s,l,p\right\}$ とする。
任意の時刻 $t$ における各ユニットの活性値を表現する場合には，$x_{(i,t)}$ とする。
開始時刻 $t_1$ では S 層の 3 ユニットが必ず，活性化した状態，すなわち 1 にセットされ，残りの 7 つのユニットは 0 とされる。
活性化状態とされる 3 つのユニットは常に固定されて，シュミュレーションを通じて変化しない。
先述のように，離散化時刻が用いられるので，各ユニットは 16 回の更新を受けることとなる。
S 層と L 層，および L 層 と P 層とは相互に接続されているため，時刻 $t>1$ においては，結合している層間の影響を受け取る。
また，$t=8$ 時刻に，原著論文では jolt と命名された，最大値を強調する活性値の増強が行われる。
また，各時刻では，活性値の大きさに依存した乱数も付加される。
このため最終時刻 $t=16$ での L 層の活性値は，その都度変動することとなる。

Dell モデルでは，モデルを記述するための 4 つのパラメータ，s, p, w, d が用意されている。
s は S 層と L 層との結合係数に関わる係数，p は L 層と P 層とに関わる結合係数，w は結合係数全体に関わる結合係数であり，d 全ユニットの減衰率 (decay rate) である。
Dell モデルでは，s, p を調整可能なパラメータとする sp モデルと w, d を調整可能なパラメータとする wd モデルとが存在する。
sp モデルでは局在する結合係数の変動を記述する意味で，限局的パラメータであるということができる。
一方，wd モデルは，全処理ユニットに共通であるため，大域的パラメータとみなしうる。
すなわち sp モデルでは w と d は固定され，s と p との差異にのみ関心がある。
一方，wd モデルでは，限局したパラメータ変動を認めず ($w=s=p$)，大域的パラメータの変動によって課題成績を説明しようとする。


<center>
<img src="/figures/2000Foygel_Dell_fig1.png" width="43%"><br/>
<!-- <img src="figures/2013Dell_fig4.jpg" width="23%">
<img src="figures/2013Dell_fig5.jpg" width="23%"><br/> -->
<div style="text-align:left;width:77%;background-color:cornsilk">
[@2000Foygell_Dell_SP] Fig. 1 Dell らの 2 段階相互活性化モデルの概念図。
上から，意味層，語彙層，音素層という 3 層から構成される。
各層間を結ぶ線は結合係数を表しており，結合は対称的である。
例えば，S 層から L 層へ向かう前向き結合係数の値は，L 層から S 層への逆向き結合のそれと等しい。
<!-- 中央: [@2013Dell_VLPM] Fig. 4 Dell らのボクセルベース病変パラメータマッピング例,
右: [@2013Dell_VLPM] Fig. 5 -->
</div></center>

## 1.2 従来モデルの問題点
本稿では，上記 Dell モデルの記述をパラメータ推定問題として捉え，機械学習の手法を援用することでパラメータの推定を行うことを提案する。
<!-- また Dellのモデルでもシミュレーション方式が 2 通りあり，それぞれ SP モデルと WD モデルと呼ばれる。
SP モデルは～こんなモデル～で，d は固定である。
一方で,WD モデルは～こんなモデルで～s=p=w である～ｄどこに入る？～。
―上記からの流れで、問題提起が変わってくる？ー -->
しかし，以下の問題点が指摘できる。

1. 入力刺激が，常に1 種類と極めて限られている。
入力は，常にネコ (CAT) を表す意味ベクトルのみである。
この入力に対する出力が，cat なら正解，dog なら意味エラー (semantic erros), mat なら形式エラー (formal errors), rat なら混合エラー (mixed errors), log なら無関連エラー (unrelated errors), さらに lat なら非単語 (non-word errors) あるいは新造語エラーとみなされる。
このように，従来モデルにより産み出されるエラーパターンは失語症者の呼称エラーに対応づけて検討されているものの，実際の検査や実験においては複数の異なる入力刺激が用いられることを鑑みれば，モデルにおける入力刺激の少なさは妥当性を欠くと言える。
また，入力刺激に対して，各概念は 10 個のニューロンを活性化し，かつ，概念同士の間で共有されるニューロンの存在が仮定されている（たとえばターゲット語 CAT と意味関連語の DOG の間では 3 個のニューロンの活性を共有している）ものの，これらのニューロン数と内容の設定根拠について詳述されていない点も，関連する問題として挙げることができるだろう。
2. シミュレーションにおける刺激回数が少ない。
Foygel & Dell (2000) ではネコの画像を 175 回提示して，どの種のエラーが報告されたかを計数している。
175 という数字はフィラデルフィア絵画命名課題(Roach, Schwartz, Martin, Grewal,&Brecher, 1996) の試行数である 175 回と一致させるためである。
シミュレーションでは、ネコの呼称を 175 回繰り返して得られた 6 種類の反応カテゴリ(正答と 5 種類のエラー) の比率を最も良く表出するパラメータの値を探索するため，パラメータ値を定めて，175 回演算を繰り返し，得られた値からパラメータを探索している。
しかしながら,現代の技術から考えるとコンピュータは疲れを知らないので,精度を向上させるためには多数回繰り返すことを検討すべきだと言える。
3. Jolt の大きさとタイミングに関する根拠が明白でない。
Jolt とは，特定のタイミングでモデルに与えられるブーストのことを指し，2 段階相互活性化モデルにおいては意味処理と語彙処理に対して与えられるよう設定されている。
すなわち，時刻 t が 1 のときには意味ブースト 10，t=8 のときには語彙ブースト 100 が付与され，モデルの処理が促進されると考えられている。
しかしながら，これらの値やタイミングの設定は恣意的であり，明確な根拠は示されていない。
4. Dell モデルの出⼒は，被検査者の反応そのものをシミュレートするのでなく，反応を分類した結果を予測する。
このため従来モデルは，被検査者の課題遂⾏能⼒のみをシミュレートするものではなく，検査者の判断をも内包してしまっていると解釈できる。
そこで，今回提案する呼称モデルの改定案では、近年の機械学習分野における手法をモデル構築に取り入れることで，これらの問題点を解決することを試みた。


## Dell モデルの改定 温度パラメータの変化

図 3 は，ソフトマックス関数において，温度パラメータを変化させた場合の各反応カテゴリーの確率密度の変化を示している。

<center>
<img src="/figures/2021_0810Dell_beta_sim.svg" width="49%"><br/>
</center>

温度パラメータ $\beta$ が 0 に近い（すなわち，健常者の反応をシミュレートしている）ときにはモデルは正解を算出しやすくなる。
一方，温度パラメータ $\beta$ が大きくなる場合，すなわち，患者の反応をシミュレートする場合には，各エラー率の値が上昇する。
$\beta$ は統計力学からの類推から温度パラメータと呼ぶことにする。
ソフトマックス関数は，ボルツマン分布と同一であって $x_i$ のエネルギー順位を与える式である。
このとき $\lim_{\beta\approx0}$ の極限では系は統計力学におけるボルツマン分決定論的に振る舞い $\lim\beta\mapsto\infty$ ではあらゆるエネルギー準位をとることとなる。
温度パラメータの導入意図は,このように系が確率的変動を許容する程度を表すものと解釈できるようにすることで，健常者と患者の呼称成績を連続的に捉えることを可能にするためであった。
すなわち，絵画命名課題において，健常者は温度パラメータが小さい，従って温度が低く安定した反応を生じることに対応する。
一方,反応が検査の都度変動するような，ある種の患者の錯語反応は，温度パラメータが大きい，従って温度が高いとみなしうる。
ソフトマックス関数については，ニューラルネットワークでの画像識別やカテゴリー判定などの分類課題に用いられている意味で汎用性が高い。
～ソフトマックス関数を呼称課題の反応生成に使用した利点～
また，相互活性化モデルの確率論的改訂である多項相互活性化モデル MIA (Multinomial Interactive Activation)モデル (McClelland, 2013; McClelland, Mirman, Bolger, & Khaitan, 2014) でも類似の概念が用いられている。
すなわち，本稿で提案するモデルは，オリジナルの IA モデルを拡張した MIA モデルの概念を援用して，絵画命名課題の妥当な解釈を行っているものとみなしうる。
ただし MIA モデルでは，温度概念を推定すべきパラメータとして扱っておらず，温度パラメータ $\beta$ を反応の安定性，あるいは多様性をとみなす考え方は，本提案手法に独自のものである。
また，ソフトマックス関数に温度パラメータを導入するアイデアは，ボルツマンマシン (Ackley, Hinton, & Sejnowski, 1985; Hinton & Sejnowski,1986) からの伝統である。
加えて，近年精度向上が著しい深層学習分野での自己半教師あり学習 (Self semi-suupervised learning でも採用されている概念でもある。
深層学習においては，ソフトマックス関数に温度パラメータ $\beta$ を導入することで，系の確率的振る舞いの変動を制御する意味合いがある (Chen, Konblith, Nouzi & Hinton, 2020; Jaiswal, Babuadeh, Makedon, 2020; Oord, Li, & Vinyals,2018)。
本手法では，確率的な振る舞いの度合いが，患者と健常統制群の課題成績を記述するパラメータであると考えことになる。
従って，ある種の患者の示す個々の反応は，都度都度変動し，決定論的な応答を得ることが少ない症例を模倣していることになると見なしうる。
一方，健常統制群の課題成績は，任意の刺激図版に対して安定的な応答が得られやすいと考えられる。
このことは，健常統制群の系では，温度パラメータ $\beta$ が低く，すなわち，系の応答が安定しているとみなすことなる。

## A.1 コード

* [](https://colab.research.google.com/github/project-ccap/project-ccap.github.io/blob/master/notebooks/2021cnps_ccap1_simpleDell.ipynb)

## A.2 患者ごとのシミュレーション結果

<center>
<img src="/figures/2021_0811W_B.svg" width="23%">
<img src="/figures/2021_0811T_T_.svg" width="23%">
<img src="/figures/2021_0811J_Fr_.svg" width="23%">
<img src="/figures/2021_0811V_C_.svg" width="23%">
<img src="/figures/2021_0811L_B_.svg" width="23%">
<img src="/figures/2021_0811J_B_.svg" width="23%">
<img src="/figures/2021_0811J_L_.svg" width="23%">
<img src="/figures/2021_0811G_S_.svg" width="23%">
<img src="/figures/2021_0811L_H_.svg" width="23%">
<img src="/figures/2021_0811J_G_.svg" width="23%">
<img src="/figures/2021_0811E_G_.svg" width="23%">
<img src="/figures/2021_0811B_Me_.svg" width="23%">
<img src="/figures/2021_0811B_Mi_.svg" width="23%">
<img src="/figures/2021_0811J_A_.svg" width="23%">
<img src="/figures/2021_0811A_F_.svg" width="23%">
<img src="/figures/2021_0811N_C_.svg" width="23%">
<img src="/figures/2021_0811I_G_.svg" width="23%">
<img src="/figures/2021_0811H_B_.svg" width="23%">
<img src="/figures/2021_0811J_F_.svg" width="23%">
<img src="/figures/2021_0811G_L_.svg" width="23%">
<img src="/figures/2021_0811W_R_.svg" width="23%">
</center>

## A.3 学習

教師信号 $\mathbf{t}=[0.97, 0.01, 0.00, 0.01, 0.00, 0.00]$ とする。
このとき最小化すべき目的関数(損失関数，誤差関数) $l$
を次のように定義する:

$$
l\left(p,x;\theta\right)\equiv\sum_i\left( t_i\log(p_i) + (1-t_i)\log(1-p_i)\right)\tag{A.1}
$$

$$
\frac{\partial l}{\partial p}=\sum_i\left(
\frac{t_i}{p_i}-\frac{1-t_i}{1-p_i}
\right)
= \sum_i\frac{t_i(1-p_i)-p_i(1-t_i)}{p_i(1-p_i)}
= \sum_i\frac{t_i-p_i}{p_i(1-p_i)}\tag{A.2}
$$

この $l$ を最小化する学習をニューラルネットワークの学習則に従い以下のような勾配降下法を用いて訓練する:
$$
\Delta\theta = \eta\frac{\partial l}{\partial\theta}
= \eta\frac{\partial l}{\partial p}\frac{\partial p}{\partial x_t}\frac{\partial x_t}{\partial\theta}\tag{A.3}
$$

合成関数の微分則に従って
$\displaystyle\frac{\partial l}{\partial\theta}=\frac{\partial l}{\partial p}\frac{\partial p}{\partial x}\frac{\partial x}{\partial\theta}\tag{A.4}$
である。

更に $p_{i}$ を $x_{j,t}$ で微分，すなわちソフトマックスの微分:
$$
\begin{aligned}
\frac{\partial p_i}{\partial\beta x_i} &=\frac{e^{\beta x_i}\left(\sum e^{\beta x_j}\right)-e^{\beta x_i}e^{\beta x_i}}{\left(\sum e^{\beta x_j}\right)^2}\\
 &=\left(\frac{e^{\beta x_i}}{\sum e^{\beta x_j}}\right)
\left(\frac{{\sum e^{\beta x_j}}-e^{\beta x_j}}{{\sum e^{\beta x_j}}}\right)\\
&=p_i \left(\frac{\sum e^{\beta x_j}}{\sum e^{\beta x_i}}
-\frac{e^{\beta x_i}}{\sum e^{\beta x_j}}\right)\\
&=p_i (1-p_j)
\end{aligned}\tag{A.5}$$

$$
\frac{\partial p_i}{\partial x_j} =  p_i(\delta_{ij}- p_j)\tag{A.6}
$$

$\theta$ を $\beta$ とそれ以外 ($w,d,s,p$) とに分けて考える。
更に，各個のパラメータについて微分することを考える。
Dell らのモデルでは次式のような漸化式が用いられた:
$$
x_{t+1} = (1-d)x_{t} + \sum w x_{t} + z,\tag{A.7}
$$

ここで $z\sim\mathcal{N}\left(0,a_1^2 + a_2^2x_{t}\right)$ である。

$$
\Delta \theta=\eta\frac{\partial l}{\partial\theta}\tag{A.8}
$$

ここで $l$ は損失関数 (誤差関数，目的関数) であり，\(\eta\) は学習係数 (learning ratio) である。

$$
\begin{aligned}
\frac{\partial l}{\partial\theta} &=\frac{\partial l}{\partial p}\frac{\partial p}{\partial x_t}\frac{\partial x_t}{\partial\theta}\\
&=\sum_i\frac{t_i-p_i}{p_i(1-p_i)}\sum_j p_i\left(\delta_{ij}-p_j\right)\frac{\partial x_{j,t}}{\partial\theta}\\
\end{aligned}\tag{A.9}
$$

今一度，損失関数を \(l\), 最終層出力の出力を確率密度関数を $y$, 各ニューロンの出力値を $x$ とする。
推定すべき Dell モデルのパラメータ $\theta=\{w,d,s,p\}$ とする。

それぞれ以下の通りである:
* $w$: 重みパラメータ
* $d$: 崩壊パラメータ
* $s$: 視覚入力層と語彙層との間の結合パラメータ
* p$: 語彙層と音韻層との結合パラメータ

$x_{i,\tau}^{(\text{Layer})}$ を $\text{Layer}=\left[s:\text{視覚的意味層}, l:\text{語彙層}, p:\text{音韻層}\right]$ を時刻 $\tau$ での層 (Layer) における $i$  番目のニューロンであるとすれば，次式を得る:

$$
\begin{array}{ll}
x_{i,t+1}^{(s)} &= (1-d)x_{i,t}^{(s)} + sw \sum_j u_{ij}^{(l)}x_j,\\
x_{i,t+1}^{(l)} &= (1-d)x_{i,t}^{(l)} + sw \sum_{j\in(s)} u_{ij}^{(s)}x_j^{(s)} + pw\sum_{j\in(p)}u_{ij}^{(p)}
x_j^{(p)},\\
x_{i,t+1}^{(p)} &= (1-d)x_{i,t}^{(p)} + pw \sum_{j\in(l)} u_{ij}^{(l)}x_j^{(l)},\\
\end{array}\tag{A.10}
$$

$$
\mathbf{\Theta}=
\left(
    \begin{array}{ccc}
    1-d & wp & 0\\
    wp & 1-d & ws\\
    0  & ws & 1-d\\
    \end{array}
\right)\tag{A.11}
$$
とすれば，

$$
\mathbf{x}_t=\mathbf{\Theta x}_{t-1}+ z\left(\mathbf{x}_{t-1};a_1^,a_2^2\right)\tag{A.12}
$$


## 機械学習と脳画像研究および心理モデル

### 言語と機能的脳画像研究を結びつけるために，単語の分散表現を機械学習的手法で表現

- [名詞の意味に関連した人間の脳活動の予測, Mitchell, 2018, Predicting Human Brain Activity Associated with the  Meanings of Nouns](https://shinasakawa.github.io/2008Mitchell_Predicting_Human_Brain_Activity_Associated_with
_the_Meanings_of_Nounsscience.pdf){:target="_blank"}

<center>
<img src="/figures/2019mitchell-54_20.png" style="width:49%"><br/>
</center>

### 下図 左のように，「セロリ」から右の脳画像を予測するために，中間表現として，兆 単位の言語コーパス (言語研究では訓練や検証に用いる言語
データをコーパスと呼ぶ) から得られた **意味特徴** を用いる

<center>
<img src="/figures/2008Mitchell_fig1.svg" style="width:49%"><br/>
<p style="text-align: left;width: 66%; background-color: cornsilk;">
Mitchell (2008) 図 1. 任意の名詞刺激に対するfMRI活性化を予測するモデルの形式。
fMRI の活性化は、2段階 プロセスで予測される。
第 1 段階では，入力刺激語の意味を，典型的な単語使用を示す大規模なテキストコーパスから値を抽出した中間的な意味的特徴の観点から符号化する。
第 2 段階では，これらの中間的な意味的特徴のそれぞれに関連する fMRIシグネチャ の線形結合として，fMRI 画像を予測する。
<!-- Form of the model for predicting fMRI activation for arbitrary noun stimuli.
fMRI activation is predicted in a two-step process.
The first step encodes the meaning of the input stimulus word in terms of intermediate semantic features whose values are extracted from a large corpus of text exhibiting typical word use.
The second step predicts the fMRI image as a linear combination of the fMRI signatures associated with each of these intermediate semantic features. -->
</p>
</center>

### 他の単語 (下図左) eat, taset, fill などの単語から セロリ を予測する回帰モデルを使って予測する
<center>
<img src="/figures/2008Mitchell_fig2.svg" style="width:66%"><br/>
<p style="text-align: left;width: 66%;background-color: cornsilk;">
Mitchell (2008) 図 2. 与えられた刺激語に対する fMRI 画像の予測。
(A) 参加者 P1 が 「セロリ」刺激語に対して、他の 58 の単語で学習した後に予測を行う。
25 個の意味的特徴のうち 3 つの特徴量のベクトルを単位長にスケーリングすることである。
(食べる, 味わう, 満たす) について学習した $c_{vi}$ 係数は， パネル上部の 3 つの画像のボクセルの色で示されている。
刺激語「セロリ」に対する各特徴量の共起値は， それぞれの画像の左側に表示されている (例えば 「食べる（セロリ）」の 共起値は 0.84)。
刺激語の活性化予測値 ((A）の下部に表示) は 25個 の意味的 fMRI シグネチャを線形結合し， その共起値で重み付けしたものである。
この図は 予測された三次元画像の1つの水平方向のスライス [z=-12 mm in Montreal Neurological Institute (MNI) space] を示している。
(B) 「セロリ」と「飛行機」について， 他の 58 個の単語を使った訓練後に予測された fMRI 画像と観察された fMRI 画像。
予測画像と観測画像の上部（後方領域）付近にある赤と青の 2本 の長い縦筋は、左右の楔状回である。
<!-- Predicting fMRI images for given stimulus words.
(A) Forming a prediction for participant P1 for the stimulus word “celery” after training on 58 other words.
Learned $c_{vi}$ coefficients for 3 of the 25 semantic features (“eat,” “taste,” and “fill”) are depicted by the voxel colors in the three images at the top of the panel.
The cooccurrence value for each of these features for the stimulus word “celery” is shown to the left of their respective images [e.g., the value for “eat (celery)” is 0.84].
The predicted activation for the stimulus word [shown at the bottom of (A)] is a linear combination of the 25 semantic fMRI signatures, weighted by their co-occurrence values.
This figure shows just one horizontal slice [z = –12 mm in Montreal Neurological Institute (MNI) space] of the predicted three-dimensional image.
(B) Predicted and observed fMRI images for “celery” and “airplane” after training that uses 58 other words.
The two long red and blue vertical streaks near the top (posterior region) of the predicted and observed images are the left and right fusiform gyri. -->}
</p>
</center>


<center>
<img src="/figures/2008Mitchell_fig3.svg" style="width:49%"><br/>
<p style="text-align: left;width:66%;background-color:cornsilk;">
Mitchell (2008) 図 3. 最も正確に予測されたボクセルの位置。
参加者 P5 の訓練セット以外の単語について、予測されたボクセルの活性化と実際のボクセルの活性化の相関を表面（A）とグラスブレイン（B）で表したもの。
これらのパネルは、少なくとも 10個 の連続したボクセルを含むクラスタを示しており、それぞれのボクセルの予測-実際の相関は少なくとも 0.28 である。
これらのボクセル・クラスターは、大脳皮質全体に分布しており、左右の後頭葉と頭頂葉、左右の豆状部、中央後葉、中央前葉に位置している。
左右の後頭葉、頭頂葉、中前頭葉、左下前頭回、内側前頭回、前帯状回に分布している。
(C) 9人の参加者全員で平均化した予測-実測相関の表面表現。
このパネルは、平均相関が 0.14 以上の連続した10 個以上のボクセルを含むクラスターを示している。
<!-- Locations of most accurately predicted voxels.
Surface (A) and glass brain (B) rendering of the correlation between predicted and actual voxel activations for words outside the training set for participant P5.
These panels show clusters containing at least 10 contiguous voxels, each of whose predicted-actual correlation is at least 0.28.
These voxel clusters are distributed throughout the cortex and located in the left and right occipital and parietal lobes; left and right fusiform,
postcentral, and middle frontal gyri; left inferior frontal gyrus; medial frontal gyrus; and anterior cingulate.
(C) Surface rendering of the predicted-actual correlation averaged over all nine participants.
This panel represents clusters containing at least 10 contiguous voxels, each with average correlation of at least 0.14. -->
</p>
</center>

### chatGPT

<div class="figure figcenter">
<img src="/figures/2022Quyang_instructGPT_fig2ja.svg" width="99%">
<div class="figcaption">

### instructGPT の概要 [2022Quyang+](https://arxiv.org/abs/2203.02155) Fig.2 を改変

</div></div>

chatGPT の GPT とは **Genrative Pre-trained Transformer** の頭文字。
**生成モデル (generative modeling)** と **事前学習 (pre-trained models)** と **トランスフォーマー (transformer)** についての理解が必要

Transformer は **言語モデル (Lanugage models)** です。
言語モデルによって，文章が処理され，適切な応答をするようになったモデルの代表が chatGPT となる。

言語モデルを理解するために，その構成要素である Transformer を取り上げる。
Transformer 2017 年の論文 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) で提案された，**ニューラルネットワーク neural network** モデル。
トランスフォーマーはゲームチェンジャーとなった。
最近の **大規模言語モデル (LLM: Large Language Model)** は，トランスフォーマーを基本構成要素とするモデルがほとんど。
上記の論文のタイトルにあるとおり，Transformer は，**注意機構 attention mechanism** に基づいて，自然言語処理の諸課題を解くモデル。



## BERT: 埋め込みモデルによる構文解析

BERT の構文解析能力を下図示した。
各単語の共通空間に射影し，単語間の距離を計算することにより構文解析木と同等の表現を得ることができることが報告さ
れている [@2019HewittManning_structural]。

<div class="figure figcenter">
<img src="/figures/2019hewitt-header.jpg" width="44%">
<img src="/figures/2019HewittManning_blogFig1.jpg" width="22%">
<img src="/figures/2019HewittManning_blogFig2.jpg" width="22%">
<div class="figcaption">
BERT による構文解析木を再現する射影空間
From `https://github.com/john-hewitt/structural-probes``
</div></div>

word2vec において単語間の距離は内積で定義されていた。
このことから，文章を構成する単語で張られる線形内積空間内の距離が構文解析木を与えると見なすことは不自然ではない
。

そこで構文解析木を再現するような射影変換を見つけることができれば BERT を用いて構文解析が可能となる。
例えば上図における chef と store と was の距離を解析木を反映するような空間を見つけ出すことに相当する。
2 つの単語 $w_i$, $w_j$ とし単語間の距離を $d\left(w_i,w_j\right)$ とする。 適当な変換を施した後の座標を $h_i$, $h_j$ とすれば，求める変換 $B$ は次式のような変換を行なうことに相当する:

$$
\min_{B}\sum_l\frac{1}{\left|s_\ell\right|^2}\sum_{i,j}\left(d\left(w_i,w_j\right)-\left\|B\left(h_i-h_j\right)\right\|^2\right)
$$

ここで $\ell$ は文 s の訓練文のインデックスであり，各文の長さで規格化することを意味している。


## Seq2seq model

<div class="figure figcenter">
<img src="/figures/2014Sutskever_S22_Fig1.svg" width="77%">
<div class="figcaption">

Sutskever+2014 Fig. 1, 翻訳モデル `seq2seq` の概念図
</div>
</div>

`eos` は文末を表す。
中央の `eos` の前がソース言語であり，中央の `eos` の後はターゲット言語の言語モデルである SRN の中間層への入力
として用いる。

注意すべきは，ソース言語の文終了時の中間層状態のみをターゲット言語の最初の中間層の入力に用いることであり，それ
以外の時刻ではソース言語とターゲット言語は関係がない。
逆に言えば最終時刻の中間層状態がソース文の情報全てを含んでいるとみなしうる。
この点を改善することを目指すことが 2014 年以降盛んに行われてきた。
顕著な例が後述する **双方向 RNN**，**LSTM** 採用したり，**注意** 機構を導入することであった。

<div class="figure figcenter">
<img src="/figures/2015Bahdanau_attention.jpg" width="44%">
<img src="/figures/2015Luong_Fig2.svg" width="44%">
<div class="figcaption">
左: Bahdanau+2014,
中: Luong+2015, Fig. 2,
右: Luong+2015, Fig. 3
</div></div>

<div class="figure figcenter">
<img src="/figures/2014Sutskever_Fig2left.svg" width="44%">
<img src="/figures/2014Sutskever_Fig2right.svg" width="44%">
<div class="figcaption">
左: Bahdanau+2014,
中: Luong+2015, Fig. 2,
右: Luong+2015, Fig. 3
</div></div>


## 埋め込みモデル，ベクトル空間

* ピラミッド・パームツリー・テスト: 認知症検査
* ターゲットと最も関連のあると考えられる選択肢を一つ選べ。

1. ターゲット: オートバイ，選択肢: 麦わら帽子，帽子，ヘルメット，兜
2. ターゲット: かもめ，選択肢: 水田，池，滝，海
3. ターゲット: 柿，選択肢: 五重塔，教会，病院，駅

<div class="figure figcenter">
<img src="/figures/2023_0712projection_concept.svg" width="24%">
<img src="/figures/2021_0831jcss_PPT1.svg" width="29%">
<img src="/figures/2021_0831jcss_PPT2.svg" width="29%">
</div>

<!-- ベクトル空間の例として，word2vec による PPT  `~/study/2021ccap/notebooks/2021_0831jcss_PPT_projection.[ip
ynb,pdf]` まで

* いまだに，このような記事が出ることの方が問題だろうと思う。
[<img src="https://www.mag2.com/p/news/wp-content/uploads/2017/09/logo_mag2news_290x50-1.png" style="width:9%"

> 大ウソつきChatGPT。訴訟文書「過去の判例」が“ほぼ出鱈目”だった理由](https://www.mag2.com/p/news/577615)

生成 AI と呼ばれる，生成 generative とは，サンプリングを行うことを意味している。
このため，サンプリングに伴う変動は常に存在する。

$$
p(x_i,\beta) = \frac{e^{x_i/\beta}}{\sum_{j\in X} e^{x_j/\beta}}
$$ -->

<!-- <img src="figures/2017Vaswani_Fig2_1.svg">
<img src="figures/2017Vaswani_Fig2_2.svg"> -->

## GPT-4

加えて，chatGPT の後続モデルである GPT-4 では，マルチモーダル，すなわち，視覚と言語の統合が進みました。

<div class="figcenter">
<img src="/figures/2023kosmos_coverpage.png" width="77%"><br/>
<div class="figcaption">

[Kosmos-1 の概念図](https://arXiv.org/abs/2302.14045)
</div></div>


まず第一に，大規模ではない，言語モデルについて考えます。
言語モデルは，機械翻訳などでも使われる技術です。
ですから，DeepL や Google 翻訳で，使っている方もいることでしょう。

chatGPT を使える方は，上記太字のキーワードについて，chatGPT に質問してみることをお勧めします。
とりわけ 注意 については，認知，視覚，心理学との関連も深く，注意の障害は，臨床，教育，発達などの分野と関係する
でしょう。

<div class="figcenter">
<img src="/figures/2017Vaswani_Fig2_1ja.svg" width="19%">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="/figures/2017Vaswani_Fig2_2ja.svg" width="29%">&nbsp;&nbsp;&nbsp;
<img src="/figures/2017Vaswani_Fig1.svg" width="39%">
<div class="figcaption">

Transformer [2017Vaswani++](https://arxiv.org/abs/1706.03762) Fig.2 を改変
</div></div>

上図で，`matmul` は行列の積，`scale` は，平均 0 分散 1 への標準化，`mask` は 0 と 1 とで，データを制限すること，`softmax` はソフトマックス関数である。

トランスフォーマーの注意とは，このソフトマックス関数である。

<!-- 本日は，機械学習と統計学との関係を取り上げ，ニューラルネットワークの導入と実習を行います。 -->

<!--
### 心理学における注意の専門用語の確認

* 外的注意 external と内的注意 internal
* 現外的注意 overt と内省的注意 covert : 眼球運動を伴う注意と，伴わない注意。
* 空間的注意
* 視覚探索: 特徴 features と対象 objects

先週は，心理学と人工知能との関係 -->




## 従来モデルの問題点

BERT の意味，文法表現を知るために，從來モデルである word2vec の単語表現概説しておく。
各単語はワンホット onehot 表現からベクトル表現に変換するモデルを単語埋め込みモデル word embedding models あるいはベクトル表現モデル vector representation models と呼ぶ。
下図  のように各単語を多次元ベクトルとして表現する。

<img src="/figures/2019Devlin_BERT01upper.svg">
単語のベクトル表現


単語埋め込み (word2vec など) 単語は周辺単語の共起情報 [点相互情報量 PMI](https://en.wikipedia.org/wiki/Pointwise_mutual_information) に基づく [@2014LevyGoldberg:nips,@2014Levy:3cosadd]。
すなわち周辺単語との共起情報を用いて単語の意味を定義している。

<img src="/figures/2019Devlin_BERT01lower.svg">

形式的には，skip-gram であれ CBOW であれ同じである。

#### 単語埋め込みモデルの問題点

単語の意味が一意に定まらない場合，ベクトル表現モデルでは対処が難しい。
とりわけ多義語の意味を定めることは困難である。
%文脈自由表現 下図の単語「アップル」は果物であるか，IT 企業であるかは，その単語を単独で取り出した場合一意に定める事ができない。

<img src="/figures/2019Devlin_BERT02upper.svg">
単語の意味を一意に定めることができない場合

<img src="/figures/2019Devlin_BERT02lower.svg">


単語の多義性解消のために，あるいは単語のベクトル表現を超えて，より大きな意味単位である，
句，節，文のベクトル表現を得る努力がなされてきた。
適切な普遍文表現ベクトルを得ることができれば，翻訳を含む多くの下流課題にとって有効だと考えられる。
seq2seq モデルは RNN の中間層に文情報が表現されることを利用した翻訳モデルであった
(図 \ref{fig:bert03})~\citep{2014Sutskever_Sequence_to_Sequence}。

\includegraphics[width=0.74\textwidth]{2019Devlin_BERT03.pdf}
seq2seq モデル

BERT は上述の從來モデルを凌駕する性能を示した。


