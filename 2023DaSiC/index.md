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
   * データの紹介1：[健常者の言い誤り 寺尾康](2023terao_dasic2.pdf){:target="_blank"}
   * [発話についての認知モデルの変遷 寺尾康](2023terao_dasic3.pdf){:target="_blank"}:
   * データの紹介2：[失語症の錯語: 高倉祐樹 (北海道大学)](2023takakura_dasic.pdf)，[立場文音 (JCHO 熊本総合病院)](2023tateba_DaSiC.pdf){:target="_blank"}，[大門正太郎（クラーク病院）](23DaSic_大門準備分_初版.pdf){:target="_blank"}，
2. 14:10-14:50 [機械学習からみた言語モデルの鏡 浅川伸一 (東京女子大学)](2023asakawa_dasic1){:target="_blank"}:

   * 休憩

3. 15:00-16:15 認知モデルからみた言語モデルの鏡と機械学習の鏡との接点
   * 認知モデルの説明: 健常者：寺尾康、[失語例：上間清司(武蔵野大学)](2023uema_dasic.pdf){:target="_blank"}, [橋本幸成(目白大学)](2023hashimoto_dasic.pdf){:target="_blank"}，
   * [機械学習モデルの説明](2023asakawa_dasic2){:target="_blank"}: 浅川伸一
4. 16:25-17:25 [実演 鏡を覗いてみると](2023asakawa_daisc3){:target="_blank"}: モデルのデモンストレーション
	浅川伸一、吉原将大（東北大学）
   * [Colab 操作方法](2023yoshihara_colab.pdf){:taget="_blank"}
   * [Google colabratory について](supp_eco){:target="_blank"}

   * 休憩

5. 17:25-17:40 議論
	登壇者全員

* [用語集](glossary){:target="_blank"}
* [演者 自己または他己紹介](self_introduction){:target="_blank"}

<center>
<img src="/figures/2004Roelofs_PsychRev_comment_fig2_.png" style="width:49%">
<img src="/figures/1885LichtheimFig1.png" style="width:32%"><br/>
<div style="width:88%;background-color:lavender;text-align:left">
左: WEAVER++ モデルにおける物品命名時の情報の流れ。
レンマ検索後，音声単語計画は厳密にフィードフォワード方式で行われ，フィードバックは音声理解系を介してのみ行われる。
内部モニタリングには，右方向に増分的に構成された音韻単語を音声理解系にフィードバックすることが含まれ，外部モニタリングには単語の発音を聞くことが含まれる。
Roelofs (2004) Fig. 2.<br/>
<!-- The flow of information in the WEAVER++ model during object naming.
After lemma retrieval, spoken word planning happens in a strictly feedforward fashion, with feedback occurring only via the speech comprehension system.
Internal monitoring includes feeding the rightward incrementally constructed phonological word back into the speech comprehension system, whereas external monitoring involves listening to the pronunciation of the word. -->
右: 図は，子どもに見られる模倣による言語習得の現象と，この処理過程が前提とする反射弧に基づく。
子どもはこの手段によって，言葉の聴覚的記憶 (聴覚的単語表象) と，協調運動の運動的記憶(運動的単語表象) を持つようになる。
これらの記憶が固定されている脳の部位を，それぞれ「聴覚心像の座」と「運動心像の座」と呼ぶことにする。
本スキーマでは，これらの部位を文字 A と M で表す。
反射円弧は，音響心像を A に伝える求心性枝 a A と，M からのインパルスを発声器官に伝える遠心性枝 M m からなり，A と M を結合する交連によって完成する。
Lichtheim (1885) Fig. 1.
<!-- The schema is founded upon the phenomena of the acquisition of language by imitation, as observed in the child, and upon the reflex arc which this process presupposes.
The child becomes possessed, by this means, of auditory memories of words (auditory word-representations) as well as of motor memories of co-ordinated movements (motor word-representations).
We may call 'centre of auditory images' and 'centre of motor images', respectively the parts of the brain where these memories are fixed.
They are designated in the schema by the letters A and M.
The reflex arc consists in an afferent branch a A, which transmits the acoustic impressions to A; and an efferent branch M m, which conducts the impulses from M to the organs of speech; and is completed by the commissure binding together A and M. -->
<!-- 左図: Roelofs (2004) Fig. 2, 右図: Lichtheim (1885) Fig.1 <br/> -->
</div>
<img src="/figures/2019Roelofs_fig1.svg" style="width:77%"><br/>
<div style="width:66%;background-color:lavender;text-align:left" >
WEAVER++/ARC モデルの概念図。
脳の領域にマッピングされた連合ネットワークと条件-行動規則。
語彙検出の語形符号化成分を赤で強調。N は名詞。Roelofs (2019) Fig. 1
<!-- Illustration of the WEAVER++/ARC model: -->
<!-- An associative network and condition-action rules mapped onto areas of the brain.
The word-form encoding component of word finding is highlighted in red. -->
<!-- N = noun.
Roelofs (2019) Fig. 1 -->
</div>
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
