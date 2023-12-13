---
title: "実演 鏡を覗いてみると"
layout: default
author: CCAP プロジェクト
---
<link href="asamarkdown.css" rel="stylesheet">

[DaSiC 7 (2023)](https://sites.google.com/view/dasic7-2023) Linguistics and Data Science in Collaboration 発表資料

## 実演 鏡を覗いてみると

実習には，Google アカウントが必要です。

* [百人一首の上の句とエンコーダによって符号化し，下の句をデコーダで生成する自作 Transformer モデル <img src="/figures/colab_icon.svg">](https://colab.research.google.com/github/ShinAsakawa/ShinAsakawa.github.io/blob/master/2023notebooks/2023_1113chihaya_Transformer.ipynb)
<!-- ## WEAVER++, Dell モデルの再現シミュレーション colab files -->
* [2021年02月22日実施 Dell モデル (Dell, 1997; Foygell and Dell,2000) 再現実験 <img src="/figures/colab_icon.svg">](https://colab.research.google.com/github/project-ccap/project-ccap.github.io/blob/master/notebooks/2021Foygel_Dell_model.ipynb)
* [2021ccap word2vec による単語連想課題のデモ, Rotaru(2018) に関連 <img src="/figures/colab_icon.svg">](https://colab.research.google.com/github/project-ccap/project-ccap.github.io/blob/master/notebooks/2021ccap_word_association_demo.ipynb)
  *  [word2vec による単語連想 + 頻度 デモ <img src="/figures/colab_icon.svg">](https://colab.research.google.com/github/project-ccap/project-ccap.github.io/blob/master/notebooks/2021ccap_word_assoc_with_freq.ipynb)

* [他言語プライミング課題での事象関連電位 （ERP) のシミュレーション Roelofs, Cortex (2016) <img src="/figures/colab_icon.svg">](https://colab.research.google.com/github/project-ccap/project-ccap.github.io/blob/master/notebooks/2021Roelofs_ERP_bilingual_lemret.ipynb)
* [概念バイアス `Conceptual Bias` (Reolofs, 2016) 絵画命名，単語音読，ブロック化，マルチモーダル統合 <img src="/figures/colab_icon.svg">](https://colab.research.google.com/github/project-ccap/project-ccap.github.io/blob/master/notebooks/2021Roelofs_Conceptual_bias.ipynb)
* [2 ステップ相互活性化モデルデモ (Foygell and Dell, 2000) <img src="/figures/colab_icon.svg">](https://colab.research.google.com/github/project-ccap/project-ccap.github.io/blob/master/notebooks/2020ccap_Foygel_Dell2000_2step_interactive_activaition_model_demo.ipynb)
* [WEVER++ デモ 2020-1205 更新 Reolofs(2019) Anomia cueing <img src="/figures/colab_icon.svg">](https://colab.research.google.com/github/project-ccap/project-ccap.github.io/blob/master/notebooks/2020ccap_Roelofs2019_Anomia_cueing_demo.ipynb)
	<!-- * [上の簡単なまとめ](2020-1214about_Roelofs_anomia_cueing)
* [日本語wikipedia による word2vec Colab 版 2021年5月 neologd 追加<img src="/figures/colab_icon.svg">](https://colab.research.google.com/github/project-ccap/project-ccap.github.io/blob/master/notebooks/2021_0531ccap_word2vec.ipynb){:target="_blank"} -->

### プレイグラウンド

* [TensorFlow Playgournd ニューラルネットワークの基本](https://project-ccap.github.io/tensorflow-playground)
* [リカレントニューラルネットワークによる文字ベース言語モデル Javascript](https://komazawa-deep-learning.github.io/character_demo.html)
* [効果的な t-SNE 使用方法](https://project-ccap.github.io/misread-tsne/index.html)



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

