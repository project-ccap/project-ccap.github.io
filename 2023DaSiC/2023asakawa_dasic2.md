---
title: "DaSiC7 (2023) 発表資料"
author: 浅川伸一
layout: default
codemirror_mode: python
codemirror_mime_type: text/x-cython
---

[DaSiC 7 (2023)](https://sites.google.com/view/dasic7-2023) Linguistics and Data Science in Collaboration 発表資料

## 機械学習モデル

<!-- <link href="/asamarkdown.css" rel="stylesheet"></link> -->

## WEAVER++, Dell モデルの再現シミュレーション colab files

<!-- - [2021年02月22日実施 Dell モデル (Dell, 1997; Foygell and Dell,2000) 再現実験 <img src="/assets/colab_icon.svg">](https://colab.research.google.com/github/project-ccap/project-ccap.github.io/blob/master/notebooks/2021Foygel_Dell_model.ipynb)
- [2021ccap word2vec による単語連想課題のデモ, Rotaru(2018) に関連 <img src="/assets/colab_icon.svg">](https://colab.research.google.com/github/project-ccap/project-ccap.github.io/blob/master/notebooks/2021ccap_word_association_demo.ipynb)
  -  [word2vec による単語連想 + 頻度 デモ <img src="/assets/colab_icon.svg">](https://colab.research.google.com/github/project-ccap/project-ccap.github.io/blob/master/notebooks/2021ccap_word_assoc_with_freq.ipynb) -->

- [他言語プライミング課題での事象関連電位 （ERP) のシミュレーション Roelofs, Cortex (2016) <img src="/assets/colab_icon.svg">](https://colab.research.google.com/github/project-ccap/project-ccap.github.io/blob/master/notebooks/2021Roelofs_ERP_bilingual_lemret.ipynb)
- [概念バイアス `Conceptual Bias` (Reolofs, 2016) 絵画命名，単語音読，ブロック化，マルチモーダル統合 <img src="/assets/colab_icon.svg">](https://colab.research.google.com/github/project-ccap/project-ccap.github.io/blob/master/notebooks/2021Roelofs_Conceptual_bias.ipynb)
<!-- - [2 ステップ相互活性化モデルデモ (Foygell and Dell, 2000) <img src="/assets/colab_icon.svg">](https://colab.research.google.com/github/project-ccap/project-ccap.github.io/blob/master/notebooks/2020ccap_Foygel_Dell2000_2step_interactive_activaition_model_demo.ipynb) -->
- [WEVER++ デモ 2020-1205 更新 Reolofs(2019) Anomia cueing <img src="/assets/colab_icon.svg">](https://colab.research.google.com/github/project-ccap/project-ccap.github.io/blob/master/notebooks/2020ccap_Roelofs2019_Anomia_cueing_demo.ipynb)
	- [上の簡単なまとめ](2020-1214about_Roelofs_anomia_cueing)


## ソフトマックス関数における温度パラメータ

<img src="/figures/2023_1026kWTA_fig1.svg">
<div class="figcaption">

データ [4,3,2,1,0] に対して，異なるベータでソフトマックス関数を実施した結果
</div>


## 3. 転移学習と微調整

<div class="figcenter">
<img src="/figures/2017Ruder_fig1.jpg" width="22%">
<div class="figcaption">

微調整 (fine tuning) による課題ごとの訓練。
多層ニューラルネットワークモデルの最終層を，課題ごとに入れ替えることで，複数の課題に適応できるようにする。
今回は，BIT の 4 課題 (線分二等分，線分検出，文字検出，星印検出) を考える。
ただし，文字検出課題と星印検出課題は，文字と記号検出を微調整した。
そのため，課題ごとの微調整は，この両課題については同一である。
従って，3 種類の微調整を行った。

図は Ruder (2017) [An Overview of Multi-Task Learning in Deep Neural Network](https://arXiv.org/abs/1706.05098),
Fig. 1. より
</div></div>

<img src="/figures/2023_0318Daimon_CNPS_p31_32.svg" width="44%">

<img src="/figures/2023_0721bit_line_bisection_demo0.svg" width="24%">
<img src="/figures/2023_0721bit_line_bisection_demo1.svg" width="24%">

<img src="/figures/2023_0723tlpa_sala_214.png" width="33%">
<img src="/figures/2023_0723tlpa_sala_83.png" width="33%">
<img src="/figures/2023_0723tlpa_sala_243.png" width="33%">

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


## 4. JNPS2021

### 4.1 従来モデル

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

教師信号 $\mathbf{t}=\left[0.97, 0.01, 0.00, 0.01, 0.00, 0.00\right]$ とする。
このとき最小化すべき目的関数(損失関数，誤差関数) $l$
を次のように定義する:

$$\tag{A.1}
l\left(p,x;\theta\right)\equiv\sum_i\left( t_i\log(p_i) + (1-t_i)\log(1-p_i)\right)
$$

$$\tag{A.2}
\frac{\partial l}{\partial p}=\sum_i\left(
\frac{t_i}{p_i}-\frac{1-t_i}{1-p_i}
\right)
= \sum_i\frac{t_i(1-p_i)-p_i(1-t_i)}{p_i(1-p_i)}
= \sum_i\frac{t_i-p_i}{p_i(1-p_i)}
$$

この $l$ を最小化する学習をニューラルネットワークの学習則に従い以下のような勾配降下法を用いて訓練する:
$$\tag{A.3}
\Delta\theta = \eta\frac{\partial l}{\partial\theta}
= \eta\frac{\partial l}{\partial p}\frac{\partial p}{\partial x_t}\frac{\partial x_t}{\partial\theta}
$$

合成関数の微分則に従って
$$\tag{A.4}
\frac{\partial l}{\partial\theta}
$$
である。

<!-- \frac{\partial l}{\partial\theta}=\frac{\partial l}{\partial p}\frac{\partial p}{\partial x}\frac{\partial x}{\partial\theta} -->

更に $p_{i}$ を $x_{j,t}$ で微分，すなわちソフトマックスの微分:

$$\tag{A.5}
\begin{aligned}
\frac{\partial p_i}{\partial\beta x_i} =\frac{e^{\beta x_i}\left(\sum e^{\beta x_j}\right)-e^{\beta x_i}e^{\beta x_i}}{\left(\sum e^{\beta x_j}\right)^{2}}
&=\left(\frac{e^{\beta x_i}}{\sum e^{\beta x_j}}\right)
\left(\frac{{\sum e^{\beta x_j}}{-e^{\beta x_{j}}}{\sum e^{\beta x_{j}}}\right)\\
                                       &=p_i \left(\frac{\sum e^{\beta x_j}}{\sum e^{\beta x_i}}-\frac{e^{\beta x_i}}{\sum e^{\beta x_j}}\right)\\
                                      &=p_i (1-p_j)\frac{\partial p_i}{\partial x_j} \\
                                      &=  p_i(\delta_{ij}- p_j)\\
\frac{\partial p_i}{\partial x_j} &=  p_i(\delta_{ij}- p_j)
\end{aligned}
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

