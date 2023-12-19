---
title: "機械学習からみた言語モデルの鏡 DaSiC7 (2023) 発表資料 (3)"
author: 浅川伸一
layout: default
codemirror_mode: python
codemirror_mime_type: text/x-cython
---

<link href="/assets/css/asamarkdown.css" rel="stylesheet">

[DaSiC 7 (2023)](https://sites.google.com/view/dasic7-2023) Linguistics and Data Science in Collaboration 発表資料

# 実演 鏡を覗いてみると

<div class="memo" style="width:77%">

怪物と戦うものは，自分もその怪物とならないように用心するがよい。
そして，君が長く深淵を覗き込むならば，深淵もまた君を覗き込む<br/>
146 (ニーチェ，木場深定訳，善悪の彼岸，120ページ，岩波書店)<br/>
</div>

<div class="figcenter">
<img src="/figures/poly_in_poly_long.gif" width="94%">

<!-- <img src="https://ds.cc.yamaguchi-u.ac.jp/~math/toybox/polyhedron_soral/explanation/poly_in_poly_long.gif"><br/> -->
<div class="figcaption">

画像出典: [双対性](https://ds.cc.yamaguchi-u.ac.jp/~math/toybox/polyhedron_soral/explanation/037_4.html)<br/>
左から，正四面体，正六面体，正八面体，正十二面体，正二十面体
</div></div>

* [機械学習における双対性 Duality principle for machine learning, ICML2023 workshop](https://dp4ml.github.io/cfp/){:target="_blank"}

#### 実習ファイル

* [百人一首の上の句とエンコーダによって符号化し，下の句をデコーダで生成する自作 Transformer モデル <img src="/assets/colab_icon.svg">](https://colab.research.google.com/github/ShinAsakawa/ShinAsakawa.github.io/blob/master/2023notebooks/2023_1113chihaya_Transformer.ipynb){:target="_blank"}

# Transformer, [Attention is all you need](https://arxiv.org/abs/1706.03762){:target="_blank"}

単語の多義性解消のために，あるいは単語のベクトル表現を超えて，より大きな意味単位である，
句，節，文のベクトル表現を得る努力がなされてきた。
適切な普遍文表現ベクトルを得ることができれば，翻訳を含む多くの下流課題にとって有効だと考えられる。

そこで，注意機構を積極的に取り込んだゲームチェンジャーが Transformer である。

<div class="figcenter">
<img src="/figures/2017Vaswani_Fig2_1ja.svg" width="19%">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="/figures/2017Vaswani_Fig2_2ja.svg" width="29%">&nbsp;&nbsp;&nbsp;
<img src="/figures/2017Vaswani_Fig1.svg" width="39%">
<div class="figcaption">

Transformer [2017Vaswani++](https://arxiv.org/abs/1706.03762) Fig.2 を改変
</div></div>

上図で，`matmul` は行列の積，`scale` は，平均 0 分散 1 への標準化，`mask` は 0 と 1 とで，データを制限すること，`softmax` はソフトマックス関数である。

トランスフォーマーの注意とは，このソフトマックス関数である。

### Transformer における位置符号化器 (PE: position encoders)

$$
\text{PE}_{(\text{pos},2i)} = \sin\left(\frac{\text{pos}}{10000^{\frac{2i}{d_{\mathop{model}}}}}\right)
$$

$$
\mathop{PE}_{(\mathop{pos},2i+1)} = \cos\left(\frac{\mathop{pos}}{10000^{\frac{2i}{d_{\mathop{model}}}}}\right)
$$

<div class="figcenter">
<img src="/figures/2023_0723PE_Transformer_curves.png" width="77%">
<div class="figcaption" style="width:55%">

Transformer の位置符号化器の出力。
Transformer は位置情報を持たないので，位置情報を周波数変換して用いる。
</div></div>

### 性能評価

<div class="figcenter">

<img src="/figures/2021Brown_GPT3_fig3_13.jpg" width="77%">
<div class="figcaption">

図 3.13: ニュース記事がモデルによって生成されたものであるかどうかを識別する人間の能力 (正しい割り当てと中立でない割り当ての比率で測定) は，モデルサイズが大きくなるほど低下する。
意図的に悪い対照モデル (出力のランダム性が高い無条件 GPT-3 小型モデル) の出力に対する精度を上部の破線で示し，ランダムな確率 (50 %) を下部の破線で示す。ベストフィットの線は 95 %信頼区間を持つべき乗則である。
<!-- Figure 3.13: People’s ability to identify whether news articles are model-generated (measured by the ratio of correct assignments to non-neutral assignments) decreases as model size increases.
Accuracy on the outputs on the deliberately bad control model (an unconditioned GPT-3 Small model with higher output randomness) is indicated with the dashed line at the top, and the random chance (50%) is indicated with the dashed line at the bottom. Line of best fit is a power law with 95% confidence intervals. -->
</div></div>


# ありえない有能さ Unreasonable effectiveness

* 1960: 自然科学における数学のありえない有能さ, [Wigner1960](https://www.maths.ed.ac.uk/~v1ranick/papers/wigner.pdf){:target="_blank"}
* 1980: 数学のあり得ない有能さ, [Hamming](https://math.dartmouth.edu/~matc/MathDrama/reading/Hamming.html){:target="_blank"}
* 2009: データのありえない有能さ, [Halevy+2009](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/35179.pdf){:target="_blank"}
* 2015: リカレントニューラルネットワークのあり得ない有能さ, [Karpathy2015](https://karpathy.github.io/2015/05/21/rnn-effectiveness/){:target="_blank"}
* 2016: 工学のあり得ない有能さ (これだけは論文のタイトルではない), [Yamis+2016](https://www.nature.com/articles/nn.4244){:target="_blank"}
* 2018: 忘却ゲートのあり得ない有能さ, [Westhuizen&Lasenby2018](https://arXiv.org/abs/1804.04849){:target="_blank"}
* 2020: 人工知能のあり得ない有能さ, [Sejnowski2020](https://www.pnas.org/doi/full/10.1073/pnas.1907373117){:target="_blank"}
* 2021: ニューラルネットワーク埋め込みのあり得ない有能さ, [Gao2021](https://medium.com/aquarium-learning/the-unreasonable-effectiveness-of-neural-network-embeddings-93891acad097){:target="_blank"}

<!--
1960. The Unreasonable Effectiveness of Mathematics in the Natural Science, [Wigner1960](https://www.maths.ed.ac.uk/~v1ranick/papers/wigner.pdf)
1980. The Unreasonable Effectiveness of Mathematics, [Hamming](https://math.dartmouth.edu/~matc/MathDrama/reading/Hamming.html)
2009. The Unreasonable Effectiveness of Data, [Halevy+2009](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/35179.pdf)
2015. The Unreasonable Effectiveness of Recurrent Neural Networks, [Karpathy2015](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
2016. The Unreasonable Effectiveness of Engineering, [Yamis+2016](https://www.nature.com/articles/nn.4244)
2018. The Unreasonable Effectiveness of The Forget Gate, [Westhuizen&Lasenby2018](https://arXiv.org/abs/1804.04849)
2020. The unreasonable effectiveness of deep learning in artificial intelligence [Sejnowski2020](https://www.pnas.org/doi/full/10.1073/pnas.1907373117)
2021. The Unreasonable Effectiveness of Neural Network Embeddings, [Gao2021](https://medium.com/aquarium-learning/the-unreasonable-effectiveness-of-neural-network-embeddings-93891acad097) -->
