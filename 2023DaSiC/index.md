---
title: "言語データとその「鏡」: 機械学習モデルを用いた言い誤りと失語症例の分析 DaSiC203"
layout: default
author: CCAP プロジェクト
---
<link href="asamarkdown.css" rel="stylesheet">

<img src="2023DaSiC_QRcode.png" style="width:12%" align="left">

<!-- # [DaSiC2023 ワークショップ](https://sites.google.com/view/dasic7-2023){:target="_blank"} -->

# 言語データとその「鏡」

## 機械学習モデルを用いた言い誤りと失語症例の分析

## [DaSiC203 ワークショップホームページ](https://sites.google.com/view/dasic7-2023/workshop?authuser=0){:target="_blank"}

* 日時: 2023年12月23日(土)
* 会場: [筑波大学天王台キャンパス 第一エリア1D201講義室 Google map](https://www.google.co.jp/maps/place/1D201%E6%95%99%E5%AE%A4/@36.108528,140.1019327,16.79z/data=!4m6!3m5!1s0x60220c0745ebad25:0x83c473710859d960!8m2!3d36.1084607!4d140.1018482!16s%2Fg%2F11g6yv8vk7?hl=ja&entry=ttu){:target="_blank"}
* 概要：
<div class="abstract">

健常者は日常の発話でついうっかり、また失語症患者は主に脳の疾患により言い誤り(錯語)を表出することが知られています。
今回のイベントでは、こうした言語データを機械学習モデルと神経科学といういわば２枚の「鏡」の前に置いた時、そこに映し出されるのはどのような景色、振る舞いかを実演を交えて示します。
はたしてそれは機械学習モデルの貢献か研究者の願望か。言語学者、機械学習の専門家、言語聴覚士という登壇者それぞれの３つの視座から、実際の健常者の言い誤りや失語症患者の錯語の実際のデータを供覧しつつ、それらのデータが機械学習モデルではどのように説明されるのか、から議論していきます。
</div>

# スケジュール

1. 13:30-14:10 趣旨説明：寺尾康 (静岡県立大学)
   * データの紹介1：[健常者の言い誤り](2023DaSIC01Terao_健常者言い誤り紹介.pptx){:target="_blank"}: 寺尾康
   * データの紹介1：[健常者の言い誤り pdf](2023DaSIC01Terao_健常者言い誤り紹介.pdf){:target="_blank"}: 寺尾康
   * [発話についての認知モデルの変遷](DaSic_Sec3発話認知モデルの変遷2.pdf){:target="_blank"}: 寺尾康
   * データの紹介2：失語症の錯語: 上間清司(武蔵野大学), 橋本幸成(目白大学)，立場文音 (JCHO 熊本総合病院)
2. 14:10-14:50 [機械学習からみた言語モデルの鏡](2023DaSIC02Asakawa_Intro_ML){:target="_blank"}: 浅川伸一（東京女子大学）

   * 休憩

3. 15:00-16:15 認知モデルからみた言語モデルの鏡と機械学習の鏡との接点
   * 認知モデルの説明: 健常者：寺尾康、失語例：上間清司、橋本幸成、大門正太郎（クラーク病院）、高倉祐樹（北海道大学）
   * [機械学習モデルの説明](2023DaSIC02Asakawa_Intro_ML): 浅川伸一
4. 16:25-17:25 [実演 鏡を覗いてみると](2023DaSIC04_demo): モデルのデモンストレーション
	浅川伸一、吉原将大（東北大学）

   * 休憩

5. 17:25-17:40 議論
	登壇者全員

* [用語集](glossary)

<center>
<img src="/figures/2004Roelofs_PsychRev_comment_fig2_.png" style="width:49%">
<img src="/figures/1885LichtheimFig1.png" style="width:32%"><br/>
左図: Roelofs (2004) Fig. 2, 右図: Lichtheim (1885) Fig.1 <br/>
<img src="/figures/2019Roelofs_Aphasiology_fig1.png" style="width:77%"><br/>
Roelofs (2019) Fig. 1
</center>

<!--
### 文献資料
- [ディープラーニング概説, 2015, LeCun, Bengio, Hinton, Nature](https://komazawa-deep-learning.github.io/2021/2015LeCun_Bengio_Hinton_NatureDeepReview.pdf){:target="_blank"}
- [ディープラーニング回顧録 Senjowski, 2020, Unreasonable effectiveness of deep learning in artificial intelligence](https://komazawa-deep-learning.github.io/2021/2020Sejnowski_Unreasonable_effectiveness_of_deep_learning_in_artificial_intelligence.pdf){:target="_blank"}
- [ディープラーニングレビュー Storrs ら, 2019, Neural Network Models and Deep Learning, 2019](https://komazawa-deep-learning.github.io/2021/2019Storrs_Golan_Kriegeskorte_Neural_network_models_and_deep_learning.pdf){:target="_blank"}
- [深層学習と脳の情報処理レビュー Kriegestorte, 2015, Deep Neural Networks: A New Framework for Modeling Biological Vision and Brain Information Processing](2015Kriegeskorte_Deep_Neural_Networks-A_New_Framework_for_Modeling_Biological_Vision_and_Brain_Information_Processing.pdf){:target="_blank"}
- [計算論的認知神経科学 Kriegeskorte and Douglas, 2018, Cognitive computational neuroscience](2018Kriegeskorte_Douglas_Cognitive_Computational_Neuroscience.pdf){:target="_blank"}
- [視覚系の畳み込みニューラルネットワークモデル，過去現在未来 Lindsay, 2020, Convolutional Neural Networks as a Model of the Visual System: Past, Present, and Future](2020Lindsay_Convolutional_Neural_Networks_as_a_Model_of_the_Visual_System_Past_Present_and_Future.pdf){:target="_blank"}
- [注意レビュー論文 Lindsay, 2020, Attention in Psychology, Neuroscience, and Machine Learning](2020Lindsay_Attention_in_Psychology_Neuroscience_and_Machine_Learning.pdf){:target="_blank"}
- [運動制御のカルマンフィルター仮説 Wolpert, Ghahramani, and Jordan, 1995, An Internal Model for Sensorimotor Integration](1995WolpertGhahramaniJordan_Internal_Model_for_Sensorimotor_Integration.pdf){:target="_blank"}
- [ハブ＆スポーク仮説 Lambon Ralph, M., Jefferies, E., Patterson, K, and Rogers, T.T., 2017 The neural and computational bases of semantic cognition](2017LambonRalphJefferiesPattersonRogers_The_neural_and_computational_bases_of_semantic_cognition.pdf){:target="_blank"}
- [2021_0705 リカレントニューラルネットワーク 概説 (再)](2016RNNcamp2handout.pdf){:target="_blank"} -->

<!--
```python
from google.colab import files<br/>
uploaded = files.upload()<br/>
```


```python
from google.colab import files<br/>
files.download('ファイル名')
``` -->


<!--
## その他情報

- [2020-0819 st2vec の tSNE](../2020-0819st2vec_tsne/2020-0819st2vec_tsne.html)
- [2020-0720PNT+Snodgrass の結果](../2020-0720pnt_snodgrass_resnet18.pdf)
- [2020-0604word2vec から見た TLPA 名詞200語のプロットtSNE_バージョン](../figures/tlpa_tSNE.pdf)
- [2020-0323ccap_handouts.pdf](../2020-0323ccap_handouts.pdf)
- [2020-0201bibliography.pdf](../2020-0201bibliography.pdf)
- [2020-0128cnps_handouts.pdf](../2020-0128cnps_handouts.pdf)
- [2020computational_neuropsychology.pdf](../2020computational_neuropsychology.pdf)


## 2019CNPS 資料より

- [Colab について](https://jpa-bert.github.io/supp01_colab)
- [Colab による外部ファイルとのインタフェース](https://jpa-bert.github.io/supp02_colab_file_management)
- [Python と numpy の初歩](https://jpa-bert.github.io/python_numpy_intro_ja)
- [CNN についての蘊蓄](https://jpa-bert.github.io/supp05_cnn)
- [RNN についての蘊蓄](https://jpa-bert.github.io/supp06_rnn)
- [NLP についての蘊蓄](https://jpa-bert.github.io/supp07_nlp) -->
