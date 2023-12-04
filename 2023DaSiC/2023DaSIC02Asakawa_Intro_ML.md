---
title: "DaSiC7 (2023) 発表資料"
author: 浅川伸一
layout: default
codemirror_mode: python
codemirror_mime_type: text/x-cython
---

<!-- <link href="asamarkdown.css" rel="stylesheet"></link> -->

## [DaSiC 7 (2023)](https://sites.google.com/view/dasic7-2023) Linguistics and Data Science in Collaboration 発表資料

<div class="figcentner">
<img src="https://ds.cc.yamaguchi-u.ac.jp/~math/toybox/polyhedron_soral/explanation/poly_in_poly_long.gif">
<div class='figcaption'>

怪物と戦うものは，自分もその怪物とならないように用心するがよい。そして，君が長く深淵を覗き込むならば，深淵もまた君を覗き込む 146 (ニーチェ，木場深定訳，善悪の彼岸，120ページ，岩波書店)<br/>
画像出典: [双対性](https://ds.cc.yamaguchi-u.ac.jp/~math/toybox/polyhedron_soral/explanation/037_4.html) 左から，正四面体，正六面体，正八面体，正十二面体，正二十面体
</div></div>


昨今の LLM，生成 AI は，ハリー・ポッターの世界で「心の奥底にある，最も切実な願望以上のものは何も示してくれない」 [みぞの鏡](https://komazawa-deep-learning.github.io/2023assets/HarryPotter_erised_mirror_chapt12_p207_.svg) (Mirror of Erised: Desire を逆から綴った) かもしれない。


## 実習

* [ニューラルネットワークで遊んでみよう](https://komazawa-deep-learning.github.io/tensorflow-playground/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.98055&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false){:target="_blank"}
* [Transformer による百人一首 エンコーダ・デコーダモデル 実習 <img src="/figures/colab_icon.svg">](https://colab.research.google.com/github/ShinAsakawa/ShinAsakawa.github.io/blob/master/2023notebooks/2023_1113chihaya_Transformer.ipynb){:target="_blank"}
* [効率よく t-SNE を使う方法](https://project-ccap.github.io/misread-tsne/)

<!-- # PyTorch -->

* [Pytorch によるニューラルネットワークの構築 <img src="/figures/colab_icon.svg">](https://colab.research.google.com/github/komazawa-deep-learning/komazawa-deep-learning.github.io/blob/master/2021notebooks/2021_1115PyTorch_buildmodel_tutorial_ja.ipynb)
* [Dataset とカスタマイズと，モデルのチェックポイント，微調整 <img src="/figures/colab_icon.svg">](https://colab.research.google.com/github/ShinAsakawa/ShinAsakawa.github.io/blob/master/2023notebooks/2023_0824pytorch_simple_fine_tune_tutorial.ipynb)
* [PyTorch Dataset, DataLoader, Sampler, Transforms の使い方 <img src="/figures/colab_icon.svg">](https://colab.research.google.com/github/ShinAsakawa/ShinAsakawa.github.io/blob/master/2023notebooks/2023_0824pytorch_dataset_data_loader_sampler.ipynb)

<!-- # tSNE -->

* [kmnist による PCA と tSNE の比較 <img src="/figures/colab_icon.svg">](https://colab.research.google.com/github/ShinAsakawa/2019komazawa/blob/master/notebooks/2019komazaawa_kmnist_pca_tsne.ipynb)

# Google colabratory でのファイルの [アップ|ダウン]ロード

<div style="width:77%;align:center;text-align:left;margin-left:10%;margin-right:10%">

```python
from google.colab import files
uploaded = files.upload()
```

```python
from google.colab import files
files.download('ファイル名')
```
</div>



## 生成 AI の性能向上

<div class="figcenter">
<img src="/figures/2021Brown_GPT3_fig3_13.jpg" width="55%">
<div class="figcaption" style="width:94%">

ニュース記事がモデルによって生成されたものであるかどうかを識別する人間の能力 (正しい割り当てと中立でない割り当ての比率で測定) は，モデルサイズが大きくなるほど低下する。
意図的に悪い対照モデル (出力のランダム性が高い無条件 GPT-3 小型モデル) の出力に対する精度を上部の破線で示し，ランダムな確率 (50 %) を下部の破線で示す。
ベストフィットの線は 95 %信頼区間を持つべき乗則である。[Brown+2021, arXiv:2005.14165](https://arxiv.com/abs/2005.14165/) Fig. 3
<!-- #### Figure 3.13: People’s ability to identify whether news articles are model-generated (measured by the
 ratio of correct assignments to non-neutral assignments) decreases as model size increases.
Accuracy on the outputs on the deliberately bad control model (an unconditioned GPT-3 Small model with higher
output randomness) is indicated with the dashed line at the top, and the random chance (50%) is indicated with
  the dashed line at the bottom. Line of best fit is a power law with 95% confidence intervals. -->
</div></div>


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


### 機械学習と心理統計学の違い

仮説検定とパラメータチューニングの差異は，母集団の相違に期すのか，それとも選択しているモデルによるものなのか。
心理統計では，データを説明する努力よりも，母集団の相違，すなわち，帰無仮説が棄却できるか採択されるかに興味がある。
ところが，帰無仮説が正しいかどうかは，選択する統計モデルに依存する。
このとき統計モデルの精度が正しいのかどうかを問題にすることは少ない。
だが，用いるモデルに依存して推論結果が変化するかも知れない。
そうするとモデルの優劣が問題になるであろう。

一方，機械学習では，心理統計の母集団に相当する概念が，汎化性能である。
所与のデータにだけ当てはまるモデルではなく，未知のデータにたいして性能の高いモデルが選択される。
未知のデータ，未学習のデータに対する性能と母集団の差異を，一概に比較することは難しいが，予測精度を高くすることが，現実には用いられる実用性が高い。
応用が可能で，実学として世の中の役に立つ成果を生み出すことができる。

### ASA アメリカ統計学会の声明再録

<!-- 1. **P 値は，データが指定された統計モデルとどの程度相性が悪いかを示すことができる** P-values can indicate how incompatible the data are with a specified statistical model. -->
<!-- 2. **P 値は，研究された仮説が真である確率を測定するものではない。そうではなく，データがランダムな偶然だけから，生成された確率を測定するものである** P-values do not measure the probability that the studied hypothesis is true, or the probability that the data were produced by random chance alone. -->
<!-- 3. **科学的な結論やビジネスや政策の決定は，p 値が特定の閾値を超えたかどうかだけに基づくべきではない** Scientific conclusions and business or policy decisions should not be based only on whether a p-value passes a specific threshold. -->
<!-- 4. **適切な推論を行うには，完全な報告と透明性が必要である** Proper inference requires full reporting and transparency. -->
<!-- 5. **P 値や統計的有意性は，効果の大きさや結果の重要性を測定するものではない** A p-value, or statistical significance, does not measure the size of an effect or the importance of a result. -->
<!-- 6. **それ自体では，p 値はモデルや仮説に関する証拠の良い尺度を提供しない。** By itself, a p-value does not provide a good measure of evidence regarding a model or hypothesis. -->

* [基礎と応用社会心理学 (BASP)  編集方針 (2014,2015)](2015Basic_and_Applied_Social_Psychology_ban_p_values_ja.md)
* [アメリカ統計学会の声明 2014, 2015](2015Basic_and_Applied_Social_Psychology_ban_p_values_ja.md)
* [統計学の誤り : 統計的妥当性の「ゴールドスタンダード」である P 値は多くの科学者が想定しているほど信頼できるものではない (Nuzzo+2014)](2014Nuzzo_Statistical_errors_ja.md)
* [統計的有意性を引退させろ (サイエンティフィックアメリカン, 2019)](2019Amrhein_Retire_statistical_significance_ja.md)

### Breiman によるデータサイエンスにおける 2 つの文化 <!-- あるいは，統計学と機械学習とニューラルネットワークの関係-->

<div class="figcenter">
<img src="/figures/2001Breiman_Two_Cultures_fig2.svg" width="39%"><br/>
<img src="/figures/2001Breiman_Two_Cultures_fig3_.svg" width="39%"><br/>
<!-- <img src="/2023assets/2001Breiman_cultures.svg" width="23%"><br/> -->
<div class="figcaption">
<!-- ![Breiman(2001)](/2023assets/2001Breiman_cultures.svg){#fig:2001breiman style="width:34%"} -->

From Leo Breiman, Statistical Modeling: The Two Cultures, _Statistical Science_, 2001, Vol. 16, No. 3, 199–231, doi:10.1214/ss/1009213725.
[pdf](https://projecteuclid.org/journals/statistical-science/volume-16/issue-3/Statistical-Modeling--The-Two-Cultures-with-comments-and-a/10.1214/ss/1009213726.full)
</div></div>

Breiman は，アンサンブル学習 (バギング，ブートストラップ法) など，影響力のあるいくつかの機械学習手法を提案した機械学習界隈のレジェンド。
<!-- Breiman によれば，2 つの文化 -->

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


