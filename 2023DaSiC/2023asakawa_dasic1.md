---
title: "機械学習からみた言語モデルの鏡 DaSiC7 (2023) 発表資料 (1)"
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


# 機械学習からみた言語モデルの鏡

<div class="memo" style="width:77%">

昨今の LLM，生成 AI は，ハリー・ポッターの世界で「心の奥底にある，最も切実な願望以上のものは何も示してくれない」 [みぞの鏡](Mirror of Erised: Desire を逆から綴った) かもしれない。<br/>
LLMs that reflect your needs as well as your intelligence could be a Mirror of Erised (“Desired” spelt backward), which in the world of Harry Potter “shows us nothing more or less than the deepest, most desperate desire of our hearts.
[Sejnowski2022](https://doi.org/10.1162/neco_a_01563)
</div>

[みぞの鏡](https://komazawa-deep-learning.github.io/2023assets/HarryPotter_erised_mirror_chapt12_p207_.svg){:target="_blank"}


#### Table of contents: part 1 第一部目次 (14:10-14:50)

1. 計算論的モデルへの招待 Invitation to computational modelings (10 min.)
   1. p 値廃止 Ban of p-values
   2. データサイエンスの 2 つの文化 Two cultures in data science
   3. 神経科学に対する機械学習の 4 つの役割 Four roles of machine learning to neuroscience
2. 記号表象と分散表象との橋渡し (あるいは，規則に基づく操作と幾何学的操作) Building a bridge between symbolic and distributional representations (or between rule-based and vector-based geometrical operations) (10 min.)
   1. ワンホット符号化と埋め込みベクトル one-hot encodings and embedding vectors
   2. ソフトマックス関数と結合係数行列 Softmax function and (tying) weight matrices
3. 符号化器・復号化器モデル Encoder-decoder models (10 min.)
   1. 言語モデル Language models
   2. 翻訳モデル Translation models
   3. 注意機構 Attention mechanism
   4. Transformer
4. 微調整と転移学習 Fine tuning and transfer learning (10 min.)
   1. 最終直下層に含まれている情報 Information containing in penultimate layers
   2. 大規模言語モデルから特定の課題へ，言い誤りの型からパラメータ推定 What models do from LLM to specific tasks is analogous to what speech errors be modified to those who produced them.
   3. マルチモーダル，マルチタスク統合へ向けて Towards Multi-modal and multi-task integrations

#### Table of contents: part 2 第二部目次 (15:15-16:15)

1. Dell+ モデルによるいい誤りのシミュレーション Dell+'s model for speech errors
   1. ソフトマックス関数の温度パラメータ thermal parameter in softmax function
   2. 患者ごとの微調整
2. A encoder-decoder model for word repetition tasks

#### Table of contents: part 3 第三部目次 (16:25-17:40)

1. A model of 百人一首 implemented on Transformer
2. Horizontal and vertical errors in speech errors

---

#  主張 Takeaways

1. 大規模言語モデル (LLM)，一般画像錦 (ImageNet) で事前訓練されたモデルに対して，転移学習 transfer learning を行うことで，関心領域の課題を解くモデルを作成
2. 関心課題に特化したモデルに対して，任意の条件とデータとを用いて，微調整 fine-tuning を行うことで，条件間の差異や生成機序を解明。
3. モデル，データ，パラメータ の三項は，言語学的規範，行動・臨床データ，機械学習モデルの三項と連結。微調整に用いる条件は，制約条件付き最適化 constrained optimization とみなしうる。このことは，データサイエンスにおける，モデルとパラメータの関する双対性原理 duality principle として定式化可能

#### キーワード keywords

**符号化・復号化モデル**，**転移学習**，**微調整**，**トランスフォーマー**，**注意**，**ソフトマックス**，**ワンホットベクトル**, **埋め込み表現**，**ラグランジェ双対性**<br/>
**Encoder-decoder models**, **Transfer learning**, **Fine-tuning**, **Transformer**, **Attention**, **Softmax**, **onehot vector**, **Embeddings**, **Lagrange duality**,

---

# 0. 背景

## 0.1 生成 AI の性能向上

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

## 0.2 Modeling

1. 記述モデル description model
箱と矢印モデルなど，質的予測
3. データ適合モデル data-fitting model
LDA と SVM との違いにあらわれている
5. アルゴリズムモデル algorithm model

<!-- <div class="figcenter">
<img src="/figures/1999Levelt_blueprint.jpg" width="49%">
<img src="/figures/1885LichtheimFig1.png" width="29%">
</div> -->

## 0.3 機械学習の定義，古典的定義と現代的定義

* アーサー・サミュエル Arthur Samuel (1959): 機械学習とは，明示的にプログラムで指示せずにコンピュータに学習させる能力を研究する分野である。<br/>
  * “field of study that gives computers the ability to learn without being explicitly programmed”
* トム・ミッチェル Tom Mitchell (1999): ある課題 T とその成績 P の評価からなる経験 E をとおして学習するコンピュータプログラムを機械学習という。<br/>
  * "A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.”

<div class="figcenter">
<img src="/figures/SAM.JPG" width="13%">
<img src="/figures/tom_mitchell.jpg" width="15%"><br/>
<div class="figcaption" style="width:66%">

左: Arthr L. Samual(1901-1990) from <http://www.i-programmer.info/history/people/669-a-l-samuel-ai-and-games-pioneer.html> <!-- (/2023assets/SAM.JPG) {#fig:SAM style="width39%"}--><br/>
右: Tom Mitchell from <http://wamc.org/post/dr-tom-mitchell-carnegie-mellon-university-language-learning-computer> <!--/2023assets/tom_mitchell.jpg){#fig:tom style="width:39%"}-->
</div></div>

## 0.3 機械学習と心理統計学の違い

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

# 1. 計算論モデルへの招待 Invitation of computational modelings

## 1.1 P 値廃止 ban of p-values

### ASA アメリカ統計学会の声明

1. **P 値は，データが指定された統計モデルとどの程度相性が悪いかを示すことができる** <!--P-values can indicate how incompatible the data are with a specified statistical model.-->
2. **P 値は，研究された仮説が真である確率を測定するものではない。そうではなく，データがランダムな偶然だけから，生成された確率を測定するものである** <!--P-values do not measure the probability that the studied hypothesis is true, or the probability that the data were produced by random chance alone.-->
3. **科学的な結論やビジネスや政策の決定は，p 値が特定の閾値を超えたかどうかだけに基づくべきではない** <!--Scientific conclusions and business or policy decisions should not be based only on whether a p-value passes a specific threshold.-->
4. **適切な推論を行うには，完全な報告と透明性が必要である** <!--Proper inference requires full reporting and transparency.-->
5. **P 値や統計的有意性は，効果の大きさや結果の重要性を測定するものではない** <!--A p-value, or statistical significance, does not measure the size of an effect or the importance of a result.-->
6. **それ自体では，p 値はモデルや仮説に関する証拠の良い尺度を提供しない。** <!--By itself, a p-value does not provide a good measure of evidence regarding a model or hypothesis.-->

   * [基礎と応用社会心理学 (BASP)  編集方針 (2014,2015)](https://komazawa-deep-learning.github.io/2023/2015Basic_and_Applied_Social_Psychology_ban_p_values_ja){:target="_blank"}
   * [アメリカ統計学会の声明 2014, 2015](https://komazawa-deep-learning.github.io/2023/2015Basic_and_Applied_Social_Psychology_ban_p_values_ja){:target="_blank"}
   * [統計学の誤り : 統計的妥当性の「ゴールドスタンダード」である P 値は多くの科学者が想定しているほど信頼できるものではない (Nuzzo+2014)](https://komazawa-deep-learning.github.io/2023/2014Nuzzo_Statistical_errors_ja){:target="_blank"}
   * [統計的有意性を引退させろ (サイエンティフィックアメリカン, 2019)](https://komazawa-deep-learning.github.io/2023/2019Amrhein_Retire_statistical_significance_ja){:target="_blank"}

## 1.2 Breiman2001 によるデータサイエンスにおける 2 つの文化 <!-- あるいは，統計学と機械学習とニューラルネットワークの関係-->

* データモデル文化とアリゴリズムモデル文化
* データモデル文化とアルゴリズムモデル文化とは，心理統計で用いられるような統計学モデルを使用する文化を指す。
* 一方，アルゴリズムモデル文化とは，機械学習における統計学的技法を重視する文化である。


<center>
<img src="/figures/2001Breiman_Two_Cultures_fig1.svg" width="33%"><br/>
<img src="/figures/2001Breiman_Two_Cultures_fig2.svg" width="24%">
<img src="/figures/2001Breiman_Two_Cultures_fig3.svg" width="24%">
<div style="text-align:left;width:77%;background-color:lavender">

* 上: 入力 x に対して，自然がどのように応答 $y$ するかという枠組み
* 下左: データモデル，
* 下右: アルゴリズムモデル
</div></center>

From Leo Breiman, Statistical Modeling: The Two Cultures, _Statistical Science_, 2001, Vol. 16, No. 3, 199–231, doi:10.1214/ss/1009213725.
[pdf](https://projecteuclid.org/journals/statistical-science/volume-16/issue-3/Statistical-Modeling--The-Two-Cultures-with-comments-and-a/10.1214/ss/1009213726.full)

## 1.3 Glaser+2019 神経科学における機械学習モデルの 4 つの役割

1. 工学的な問題の解決 機械学習は， 医療診断， ブレインコンピュータインターフェース， 研究ツールなど， 神経科学者が使用する手法の予測性能を向上させることができる。
2. 予測可能な変数の特定 機械学習により， 脳や外界に関連する変数がお互いを予測しているかどうかをより正確に判断することができる。
3. 単純なモデルのベンチマーク。 解釈可能な簡易モデルと精度の高い ML モデルの性能を比較することで， 簡易モデルの良し悪しを判断するのに役立つ。
4. 脳のモデルとしての役割。 脳が機械学習システム， 例えばディープニューラルネットワークと同様の方法で問題を解決しているかどうかを論じることができる。

<div class="figcenter">
<img src="/figures/2019Glaser_fig2.jpg" width="55%">
</div>
<div class="figcaption">

出典: [Glaser+2019](https://linkinghub.elsevier.com/retrieve/pii/S0301008218300856) Fig. 2
</div>

<!-- 訓練済ニューラルネットワークを脳と比較する傾向は，画像認識などの行動課題におけるニューラルネットワークの大きな成果により，最近になって再燃した (He+2015)。
興味深いことに，これらのネットワークは視覚における腹側流路と類似点が多い。 -->

# 2. 記号表現と分散表象との間隙を埋める

1. ワンホットベクトル を 分散表現へ
2. 分散表現をワンホット表現へ

記号とベクトルとを相互に変換できれば，便利。

* [‘戦’,’争’] の書記表現 [ 1, 1565, 1604, 2]
* ['s', 'e', 'N', 's', 'o', 'u'] の音韻表現 [ 1, 37, 7, 14, 37, 8, 6, 2]

同様に，単語 (もしくはトークン) についてすべて付番し，そのトークン番号で表現 (tokenizer)

~~~python
>>> from transformers import EncoderDecoderModel, BertTokenizer, BertConfig
>>> tknz = BertTokenizer.from_pretrained('sonoisa/sentence-bert-base-ja-mean-tokens-v2')
>>> print(tknz('誰が一番に着くか私には分かりません。').input_ids)
[2, 3654, 14, 52, 246, 7, 816, 609, 28470, 1325, 7, 28450, 155, 20431, 28491, 4263, 8, 3]

>>> print(tknz.convert_ids_to_tokens([2, 3654, 14, 52, 246, 7, 816, 609, 28470, 1325, 7, 28450, 155, 20431, 28491, 4263, 8, 3]))
['[CLS]', '誰', 'が', '一', '番', 'に', '着', 'く', '##か', '私', 'に', '##は', '分', 'かり', '##ま', '##せん', '。', '[SEP]']

>>> tknz('言語学').input_ids
[2, 1882, 112, 3]

>>> tknz.convert_ids_to_tokens([2, 1882, 112, 3])
['[CLS]', '言語', '学', '[SEP]']

>>> tknz('データサイエンス').input_ids
[2, 1676, 14916, 3]

>>> tknz.convert_ids_to_tokens([1676,14916])
['データ', 'サイエンス']

>>> tknz('DaSiC7').input_ids
[2, 28367, 28583, 28535, 28598, 127, 3]

>>> tknz.convert_ids_to_tokens([28367, 28583, 28535, 28598, 127])
['Da', '##S', '##i', '##C', '7']
~~~

単語 `データサイエンス` は データ が 1676 番という ID 番号を持ち，かつ，サイエンスが 14916 番であることが分かる。
したがって，データサイエンスを表現するには，32000 次元のワンホットベクトルを 2 つ用いて，

~~~python
[0,0,..,1,,...,0,0,0]
~~~

のようなベクトルを作成する。このとき，ベクトルの 1 つの要素だけが 1 で，残りすべての要素が 0 であるベクトルをワンホットベクトルと呼ぶ。

<div class="figcenter">
<img src='/figures/2023dasic_onehot2embeddings.svg' width="49%">
</div>

* ワンホットベクトルから，埋め込み表現ベクトルを作成するには，変換行列を掛ければ良い
* 埋め込み表現から，ワンホットベクトルに戻すことによって，出力トークンを得る。このためにはソフトマックス関数を用いる

$$\tag{ソフトマックス関数}
p(x_i) = \frac{\exp{x_i}}{\sum_{j\neq\mathcal{X}}\exp(x_j)}
$$

* 指数関数 $e^{x}$ は，$x$ が実数の範囲であれば，正負に関わらず，正の値に変換し，かつ変換後も大小関係が保たれる。すなわち $x > y:\rightarrow\exp(x)>\exp(y)$
* すべて要素が正であれば，各要素の総和を分母とし，分子に，その値をとれば，確率とみなしうる。
* すなわちソフトマックス関数とは，実数値をとるニューラルネットワークの出力を確率に変換し，その最大値を強調するように作用することが分かる。
* ソフトマックス関数をニューラルネットワークモデルの最終層の出力とした場合，学習に用いられる教師信号もワンホット表現されていれば，ワンホット表現を近似する目的でソフトマックス関数は用いられていると解釈することもできる。
* ソフトマックス関数は，数ある候補項目の中から，一つを選ぶという意思決定に用いられる。
* ソフトマックス関数は，数ある候補項目の中から，唯一つの項目に着目するという意味で，注意機構に用いられる。
* ソフトマックス関数において，考慮すべき項目が x と y と2 つしかない場合，<br/>
$$
p(x) = \frac{\exp(x)}{\exp(x)+\exp(y)}=\frac{1}{1+\exp(-(x-y))}=\frac{1}{1+\exp(-Z)}
$$
<br/>
ここで，$z=(x-y)$，と書くことができる。
上式は，ロジスティック回帰などでも用いられる，ロジスティックシグモイド関数である。
2 値分類問題における真偽値，コーホート分析における生存確率などとも考えることができる。
* ソフトマックス関数の分母に現れる $\sum_{j}\exp(x_{j})$ は，物理学 (統計力学，熱力学) における分配関数 partition function と形式的には同じ形をしている。
* このことは，ニューラルネットワークの出力層の素子全体を，熱力学的系と同一視して考えることができることを意味する。熱力学的エネルギーであり，熱力学的エントロピーと情報論敵エントロピーが同じ式をしていることが，ソフトマックス関数の定義から導出できる。

まとめると，ワンホットベクトルを埋め込みベクトルに変換することは，記号表現を高次元実空間へ射影することであり，反対に，埋め込み表現を，ソフトマックス関数を用いて記号表現へ変換することで，記号 (あるいは規則，言語) と分散表現，脳内の神経活動へと変換する橋渡しができることとなる。

<!-- ## 機械学習と脳画像研究および心理モデル -->

### 言語と機能的脳画像研究を結びつけるために，単語の分散表現を機械学習的手法で表現

- [名詞の意味に関連した人間の脳活動の予測, Mitchell, 2018, Predicting Human Brain Activity Associated with the  Meanings of Nouns](https://shinasakawa.github.io/2008Mitchell_Predicting_Human_Brain_Activity_Associated_with_the_Meanings_of_Nounsscience.pdf){:target="_blank"}

<div class="figcenter">
<img src="/figures/2019mitchell-54_20.png" style="width:55%">
</div>
<div class="figcaption" style="width:33%">

Mitchell+2019 より
</div>

下図 左のように，「セロリ」から右の脳画像を予測するために，中間表現として，兆 単位の言語コーパス (言語研究では訓練や検証に用いる言語データをコーパスと呼ぶ) から得られた **意味特徴** を用いる

<div class="figcenter">
<img src="/figures/2008Mitchell_fig1.svg" style="width:66%"><br/>
</div>
<div class="figcaption">

Mitchell+2008 図 1. 任意の名詞刺激に対するfMRI活性化を予測するモデルの形式。
fMRI の活性化は、2段階 プロセスで予測される。
第 1 段階では，入力刺激語の意味を，典型的な単語使用を示す大規模なテキストコーパスから値を抽出した中間的な意味的特徴の観点から符号化する。
第 2 段階では，これらの中間的な意味的特徴のそれぞれに関連する fMRIシグネチャ の線形結合として，fMRI 画像を予測する。
<!-- Form of the model for predicting fMRI activation for arbitrary noun stimuli.
fMRI activation is predicted in a two-step process.
The first step encodes the meaning of the input stimulus word in terms of intermediate semantic features whose values are extracted from a large corpus of text exhibiting typical word use.
The second step predicts the fMRI image as a linear combination of the fMRI signatures associated with each of these intermediate semantic features. -->
</div>

### 他の単語 (下図左) eat, taset, fill などの単語から セロリ を予測する回帰モデルを使って予測

<div class="figcenter">
<img src="/figures/2008Mitchell_fig2.svg" style="width:66%">
</div>
<div class="figcaption">

Mitchell+2008 図 2. 与えられた刺激語に対する fMRI 画像の予測。
(A) 参加者 P1 が 「セロリ」刺激語に対して、他の 58 の単語で学習した後に予測を行う。
25 個の意味的特徴のうち 3 つの特徴量のベクトルを単位長にスケーリングすることである。
(食べる, 味わう, 満たす) について学習した $c_{vi}$ 係数は， パネル上部の 3 つの画像のボクセルの色で示されている。
刺激語「セロリ」に対する各特徴量の共起値は， それぞれの画像の左側に表示されている (例えば 「食べる（セロリ）」の 共起値は 0.84)。
刺激語の活性化予測値 (A）の下部に表示) は 25個 の意味的 fMRI シグネチャを線形結合し， その共起値で重み付けしたものである。
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
</div>

<!-- <center>
<img src="/figures/2008Mitchell_fig3.svg" style="width:49%"><br/>
<p style="text-align: left;width:66%;background-color:cornsilk;"> -->
<!-- Mitchell (2008) 図 3. 最も正確に予測されたボクセルの位置。
参加者 P5 の訓練セット以外の単語について、予測されたボクセルの活性化と実際のボクセルの活性化の相関を表面（A）とグラスブレイン（B）で表したもの。
これらのパネルは、少なくとも 10個 の連続したボクセルを含むクラスタを示しており、それぞれのボクセルの予測-実際の相関は少なくとも 0.28 である。
これらのボクセル・クラスターは、大脳皮質全体に分布しており、左右の後頭葉と頭頂葉、左右の豆状部、中央後葉、中央前葉に位置している。
左右の後頭葉、頭頂葉、中前頭葉、左下前頭回、内側前頭回、前帯状回に分布している。
(C) 9人の参加者全員で平均化した予測-実測相関の表面表現。
このパネルは、平均相関が 0.14 以上の連続した10 個以上のボクセルを含むクラスターを示している。 -->
<!-- Locations of most accurately predicted voxels.
Surface (A) and glass brain (B) rendering of the correlation between predicted and actual voxel activations for words outside the training set for participant P5.
These panels show clusters containing at least 10 contiguous voxels, each of whose predicted-actual correlation is at least 0.28.
These voxel clusters are distributed throughout the cortex and located in the left and right occipital and parietal lobes; left and right fusiform,
postcentral, and middle frontal gyri; left inferior frontal gyrus; medial frontal gyrus; and anterior cingulate.
(C) Surface rendering of the predicted-actual correlation averaged over all nine participants.
This panel represents clusters containing at least 10 contiguous voxels, each with average correlation of at least 0.14. -->
<!-- </p>
</center> -->

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
このことから，文章を構成する単語で張られる線形内積空間内の距離が構文解析木を与えると見なすことは不自然ではない。

そこで構文解析木を再現するような射影変換を見つけることができれば BERT を用いて構文解析が可能となる。
例えば上図における chef と store と was の距離を解析木を反映するような空間を見つけ出すことに相当する。
2 つの単語 $w_i$, $w_j$ とし単語間の距離を $d\left(w_i,w_j\right)$ とする。 適当な変換を施した後の座標を $h_i$, $h_j$ とすれば，求める変換 $B$ は次式のような変換を行なうことに相当する:

$$
\min_{B}\sum_l\frac{1}{\left|s_\ell\right|^2}\sum_{i,j}\left(d\left(w_i,w_j\right)-\left\|B\left(h_i-h_j\right)\right\|^2\right)
$$

ここで $\ell$ は文 s の訓練文のインデックスであり，各文の長さで規格化することを意味している。

## 単語埋め込みモデル word2vec

<center>
<img src="/figures/2013Mikolov_KingQueenFig.svg" width="44%">
<img src="/figures/2013Mikolov_FigCountries.svg" width="44%"><br/>
<!-- <img src="figures/2013Mikolov_skip-gram_cbow.svg" width="49%"><br/> -->
<div class="figcaption">
図 Mikolov+2013 より
</div></center>

### skip-gram と CBOW

* Mikolov は **word2vec** によりニューラルネットワークによる意味実装を示した (Mikolov+2013)。
* Word2vec は実装に 2 種類ある。それぞれ **CBOW** と **skip-gram** と命名されている。
* "king" - "man" + "woman" = "queen" のアナロジーを解くことができると喧伝された。
* 下図左は意味的なアナロジーがベクトルの向きとして表現されていることに注目。

ベクトルは方向と大きさを持っている矢印で表現される。
このとき，矢印の原点を移動することを考える。
たとえば "MAN" から "WOMAN" へ向かう矢印を平行移動して "KING" まで持ってくると，その矢印は "QUEEN" を重なることが予想できる。
これがアナロジー問題の解放の直感的説明である。

* word2vec の実装には 2 種類ある。どちらを使っても同じような結果を得ることが可能である。
    1. CBOW: Continous Bog of Words 連続単語袋
    2. skip-gram: スキップグラム

両者は反対の関係になrる。下図を参照。

<div class="figcenter">
<img src="/figures/2013Mikolov_Fig1.svg" style="width:49%"><br>
From Mikolov+2013, Fig. 1
</div>

CBOW も skip-gram も 3 層にニューラルネットワークです。その中間層に現れた表現を **ベクトル埋め込みモデル**
あるいは **単語埋め込みモデル** と言う。

* CBOW モデルは周辺の単語の単語袋詰め表現から中央の単語を予測
* skip-gram は中心の単語から周辺の単語袋詰表現を予測

たとえば，次の文章を考える:

~~~python
["彼", "は", "意味論", "を", "論じ", "た"]
~~~

表記を簡潔にするため各単語に ID を割り振ることとする。
すると上文は，単語 ID を用いて以下のように表現される:

```python
{"彼":0, "は":1, "意味論":2, "を":3, "論じ":4, "た":5}
```

~~~python
[0, 1, 2, 3, 4, 5]
~~~

ウィンドウ幅がプラスマイナス 2 である CBOW モデルでは 3 層の多層パーセプトロンの入出力関係は，入力が 4 次元
ベクトル，出力も 4 次元ベクトルとなる。
文の境界を無視すれば，以下のような入出力関係とみなせます。


### 負事例サンプリング

Word2vec を使って大規模コーパスを学習させる際に，学習させるデータ以外に全く関係のない組み合わせをペナルティーとして与えることで精度が向上する ([Mikolov+2013](https://arXiv.org/abs/1310.4546v))。

罰則項に，乗する重みをどの程度にするかは，経験による。


## ベクトル空間

* ピラミッド・パームツリー・テスト: 認知症検査 ([意味連合検査，佐藤2022](https://www.amazon.co.jp/dp/4909375112){:target="_blank"})<br/>
* ターゲットと最も関連のあると考えられる選択肢を一つ選べ。

1. ターゲット: オートバイ，選択肢: 麦わら帽子，帽子，ヘルメット，兜
2. ターゲット: かもめ，選択肢: 水田，池，滝，海
3. ターゲット: 柿，選択肢: 五重塔，教会，病院，駅

<div class="figure figcenter">
<img src="/figures/2023_0712projection_concept.svg" width="33%">
<img src="/figures/2021_0831jcss_PPT2.svg" width="55%">
<img src="/figures/2021_0831jcss_PPT1.svg" width="77%">
</div>
<div class="figcaption" width="33%">

近藤・浅川2020 より
</div>

### 人工ニューラルネットワークと神経細胞の機能的類似

人工ニューラルネットワークと脳の反応特性の類似性は，これらのモデルが脳の計算の重要な側面を捉えている可能性を示す

* 両者 (ニューラルネットワークと神経細胞) 共に階層的，多層的
* 画像の画素からの情報は通常，十数層の「ニューロン」(ノード) を通して処理される
* 類似した組織に加えて，その活性化も類似
  * 初期ノードは Gabor のような受容野を持つ (Güçlü&van_Gerven2015)，V1 に見られるエッジ検出器と類似
  * これらのネットワークの初期層/中間層/後期層における活性化は，それぞれ V1/V4/IT 反応 (個々のニューロンと fMRI 反応の両方) を予測 (Yamins&DiCarlo2016, Yamins+2014, Khaligh-Razavi&Kriegeskorte2014, Güçlü&van_Gerven2015)。
  * 深層ニューラルネットワークは，物体認識において視点に対して同様に不変 (Kheradpisheh+2016a,b)
  * 画像間で同様に反応し (Khaligh-Razavi&Kriegeskorte2014)，同様のタイプのエラーを犯す (Kheradpisheh+2016a,b)

<!-- これら類似は，競合するどのクラスのモデルよりも長く，視覚野のより広い範囲に及んでいる。 -->
<!-- 訓練されたニューラルネットワークと脳の類似性は，視覚系以外にも広がっている。
これらの研究の形式は，ほぼ共通して，脳領域の内部反応特性と，その脳領域に関連する行動課題で訓練された神経ネットワークの特性を比較するものである。 -->

* <!--30 年前に発表された先駆的な研究では，-->後頭頂ニューロンと，視覚的場面で物体の位置を特定するよう訓練された神経回路網との類似性が示された (Zipser&Andersen 1988)。
* <!--さらに最近では，-->情景認識について訓練されたネットワークは，後頭葉の場所領域における反応を正確に予測 (Bonner&2018)。
* 音声認識と音楽ジャンル予測について訓練されたネットワークは，聴覚皮質と同様の活動を示す (Kell+2018)。
* サルの動きを再現するように訓練されたリカレントニューラルネットワークには，一次運動皮質のニューロンと選択性が非常によく似た活動をするユニットが含まれる (Sussillo+2015)。
* ナビゲーション課題を訓練したリカレントネットワークの素子は，嗅内皮質や海馬のグリッド細胞や場所細胞と似た活性を持つ (Kanitscheider&Fiete2017, Cueva &Wei2018, Banino+2018)。


<!--
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
 -->

# 3. 符号化器・復号化器モデル

## 3.1 言語モデル


言語モデル (LM: Language Model) とは，単語系列 $w_t, t\in[0,t]$ が与えられたとき，次単語 $w_{t+1}$
の確率を予測することを指す。

$$\begin{aligned}
L &= \max -\log p(x) & \text{ 負の対数尤度 negative log likelihood}\\
  &= \max -\log p(x_{t+1}|x_{t}) & \text{ uni-gram}\\
  &= \max -\log p(x_{t+1}|x_{t},x_{t-1}) & \text{ bi-gram}\\
  &= \max -\log p(x_{t+1}|x_{t},x_{t-1},x_{t-2}) & \text{ tri-gram}\\
  &= \max -\log p(x_{t+1}|x_{t},x_{t-1},x_{t-2}, x_{t-3}) & \text{ quad-gram}\\
  &= \max -\log p(x_{t+1}|x_{t},x_{t-1},x_{t-2}, x_{t-3}) & \text{ quint-gram}\\
\end{aligned}$$

<!-- $P(w_{t+1}|w_{[0:t]})$ -->

t 番目の位置に現れる単語を $w_t$ と表記し，その確率を $p(w_t)$ とする。
このとき，次式を言語モデル LM と呼ぶ。

<!-- $$\tag{LM}
p(w_t) = p(w_t|w_{t\in [0,\ldots,t-1]})
$$

上式右辺は，条件付き確率を表す。すなわち，事前に出現した単語すべてによって，次の単語が定まるとするモデルである。 -->
最も単純なモデルでは，直前に現れた単語のみによって定まるモデル $p(w_t|w_{t-1})$ である 。
この場合には，総単語数を $|V|$ とすれば，言語モデルが保持しなければならない記憶容量は $|V|^2$ で与えられる (バイグラム bi-gram 言語モデル)。
直前の 2 単語から定まるモデルでは $p(w_{t}|w_{t-1},w_{t-2})$ であるが，この場合の記憶容量は $|V|^3$ となる (トリグラム tri-gram 言語モデル)。
Transformer で採用された語彙数は $|V|=32,000$ である。
従って，$|V|^n$ の際に必要となる記憶容量は，膨大である。

このことは，統計的言語モデルにとって大きな制約となっていた。
言語モデルの記憶容量問題を解決する手法として，リカレントニューラルネットワークに基づく言語モデル (neural network language models: NLM) が用いられてきた ([Mikolov+2010](https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf))。
リカレントニューラルネットワークを言語モデルとして用いれば，上記記憶容量の問題は発生しない。

だが，NLM では，長距離依存 long term dependency を解消する問題が残されていた。
このため，エルマン型の単純再帰型リカレントニューラルネットワークではなく，LSTM や GRU といった，長期記憶を調整する機構を持ったリカレントニューラルネットワークを用いることが試みられてきた。

以下に示す，翻訳モデルでは，加えて注意機構を実装することにより，長距離依存解消に対応することが試みられてきた。



## 3.2 翻訳モデル (Seq2seq model)

<div class="figcenter">
<img src="/figures/2014Sutskever_S22_Fig1.svg" width="77%">
<div class="figcaption">

Sutskever+2014 Fig. 1, 翻訳モデル `seq2seq` の概念図
</div></div>

`eos` は文末を表す。
中央の `eos` の前がソース言語であり，中央の `eos` の後はターゲット言語の言語モデルである SRN の中間層への入力
として用いる。


<div class="figcenter">
<img src="/figures/2014Sutskever_Fig2left.svg" width="44%">
<img src="/figures/2014Sutskever_Fig2right.svg" width="44%">
<div class="figcaption">
左: Bahdanau+2014,
中: Luong+2015, Fig. 2,
右: Luong+2015, Fig. 3
</div></div>

## 3.3 注意機構

注意すべきは，ソース言語の文終了時の中間層状態のみをターゲット言語の最初の中間層の入力に用いることであり，それ以外の時刻ではソース言語とターゲット言語は関係がない。
逆に言えば最終時刻の中間層状態がソース文の情報全てを含んでいるとみなしうる。
この点を改善することを目指すことが 2014 年以降盛んに行われてきた。
顕著な例が後述する **双方向 RNN**，**LSTM** 採用したり，**注意** 機構を導入することであった。

<div class="figcenter">
<img src="/figures/2015Bahdanau_attention.jpg" width="44%">
<img src="/figures/2015Luong_Fig2.svg" width="44%">
<div class="figcaption">
左: Bahdanau+2014,
中: Luong+2015, Fig. 2,
右: Luong+2015, Fig. 3
</div></div>

## 3.4 注意機構を用いて系列データを処理する Transformer

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

<!--
## 従来モデルの問題点

BERT の意味，文法表現を知るために，從來モデルである word2vec の単語表現概説しておく。
各単語はワンホット onehot 表現からベクトル表現に変換するモデルを単語埋め込みモデル word embedding models あるいはベクトル表現モデル vector representation models と呼ぶ。
下図  のように各単語を多次元ベクトルとして表現する。

<img src="/figures/2019Devlin_BERT01upper.svg">
単語のベクトル表現

単語埋め込み (word2vec など) 単語は周辺単語の共起情報 [点相互情報量 PMI](https://en.wikipedia.org/wiki/Pointwise_mutual_information) に基づく [@2014LevyGoldberg:nips,@2014Levy:3cosadd]。
すなわち周辺単語との共起情報を用いて単語の意味を定義している。

<img src="/figures/2019Devlin_BERT01lower.svg">

形式的には，skip-gram であれ CBOW であれ同じである。 -->


<!--
#### 単語埋め込みモデルの問題点

単語の意味が一意に定まらない場合，ベクトル表現モデルでは対処が難しい。
とりわけ多義語の意味を定めることは困難である。
%文脈自由表現 下図の単語「アップル」は果物であるか，IT 企業であるかは，その単語を単独で取り出した場合一意に定める事ができない。

<img src="/figures/2019Devlin_BERT02upper.svg">
単語の意味を一意に定めることができない場合

<img src="/figures/2019Devlin_BERT02lower.svg"> -->



<!--
まず第一に，大規模ではない，言語モデルについて考えます。
言語モデルは，機械翻訳などでも使われる技術です。
ですから，DeepL や Google 翻訳で，使っている方もいることでしょう。

chatGPT を使える方は，上記太字のキーワードについて，chatGPT に質問してみることをお勧めします。
とりわけ 注意 については，認知，視覚，心理学との関連も深く，注意の障害は，臨床，教育，発達などの分野と関係する
でしょう。 -->


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

<!-- ## GPT-4

加えて，chatGPT の後続モデルである GPT-4 では，マルチモーダル，すなわち，視覚と言語の統合が進みました。

<div class="figcenter">
<img src="/figures/2023kosmos_coverpage.png" width="77%"><br/>
<div class="figcaption">

[Kosmos-1 の概念図](https://arXiv.org/abs/2302.14045)
</div></div>

 -->
