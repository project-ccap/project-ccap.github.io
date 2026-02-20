---
title: Probabilistic machine learning and artificial intelligence
author: Zoubin Ghahramani
date: 2015
journal: 452 NATURE, VOL 521, 28 MAY 2015
doi: 10.1038/nature14541
---
<link href="/asamarkdown.css" rel="stylesheet"></link>
<div class="title">

確率的機械学習と人工知能<br/>
Probabilistic machine learning and artificial intelligence<br/>
Zoubin Ghahramani (2015) Nature<br/>
</div>

<div class="abstract">

機械が経験から学ぶには？
確率論的モデリングは，学習とは何かを理解するための枠組みを提供する。
したがって，経験によって得られたデータから学習する機械を設計するための主要な理論的および実践的アプローチの 1 つとして浮上してきた。
確率論的枠組みは，モデルや予測に関する不確実性をどのように表現し，操作するかを記述するものであり，科学的データ解析，機械学習，ロボット工学，認知科学，人工知能において中心的な役割を担っている。
本総説では，このフレームワークについて紹介し，この分野における最先端の進歩，すなわち，確率的プログラミング，ベイズ最適化，データ圧縮，自動モデル発見について論じる。
<!-- How can a machine learn from experience? Probabilistic modelling provides a framework for understanding what learning is, and has therefore emerged as one of the principal theoretical and practical approaches for designing machines that learn from data acquired through experience.
The probabilistic framework, which describes how to represent and manipulate uncertainty about models and predictions, has a central role in scientific data analysis, machine learning, robotics, cognitive science and artificial intelligence.
This Review provides an introduction to this framework, and discusses some of the state-of-the-art advances in the field, namely, probabilistic programming, Bayesian optimization, data compression and automatic model discovery.-->
</div>

# 論文要約

## 背景と目的
この論文は、機械学習と人工知能における確率論的アプローチの重要性を論じたレビュー論文です。著者は、真の機械知能には不確実性の適切な表現と処理が不可欠であり、確率的モデリングがその理論的・実践的基盤を提供すると主張しています。

## 主要な論点

### 1. ベイズ機械学習の基礎
- **学習の本質**: 学習とは観測データを説明するためにもっともらしいモデルを推論すること
- **不確実性の重要性**: データは多くのモデルと一致する可能性があり、モデル選択と予測には本質的に不確実性が伴う
- **ベイズアプローチ**: 事前分布を事後分布に変換することで、データからの学習を確率論的に実現

### 2. 5つの最先端研究分野

**① 確率的プログラミング**
- コンピュータプログラムとして確率モデルを表現
- シミュレータとしてのプログラムと、万能推論エンジンによる自動的な条件付け推論
- グラフィカルモデルの一般化により、より柔軟なモデル表現が可能

**② ベイズ最適化**
- 評価コストの高い未知関数の大域最適化
- ガウス過程により関数の不確実性を表現し、次の評価点を効率的に決定
- 深層学習のハイパーパラメータ最適化などに応用

**③ 確率的データ圧縮**
- Shannon理論に基づく圧縮と確率モデリングの対応関係
- 優れた確率モデルほど高い圧縮率を実現
- ベイズ型ノンパラメトリックモデルが世界最高水準の圧縮アルゴリズムと等価

**④ 自動統計家**
- データから解釈可能なモデルを自動発見し、平易な英語で説明
- 確率的ビルディングブロックを文法的に組み合わせてモデル空間を探索
- ベイズ型ノンパラメトリックと周辺尤度による原理的なモデル選択

**⑤ ノンパラメトリック手法**
- データ量に応じて複雑さが適応的に変化するモデル
- ガウス過程（関数のモデリング）、ディリクレ過程（クラスタリング）、Indian buffet process（特徴選択）
- 無限パラメータモデルでもベイズアプローチにより過学習を回避

## 将来展望
- ビッグデータ時代においても、個別化や階層的モデリングにより不確実性表現は重要
- 不確実性が中心的役割を果たす問題では確率的アプローチが不可欠
- 機械知能の発展には、不完全に観測された不確実な世界での理解と行動が必要

## 結論
この論文は、パターン認識を超えた幅広い機械学習問題において、確率論的フレームワークが原理的で強力なアプローチを提供することを示しています。特に、不確実性の表現と操作が重要な問題領域では、確率的機械学習が今後も中心的な役割を担っていくと予測されています。

# ベイズ機械学習
<!-- # Bayesian machine learning -->

機械学習における確率論の重要な考え方は，学習とは，観測されたデータを説明するために，もっともらしいモデルを推論することと考えることができるということである。
機械はそのようなモデルを用いて将来のデータを予測し，その予測に基づいた合理的な判断を下すことができる。
このとき，基本的な役割を果たすのは不確実性である。
観測されたデータは多くのモデルと一致する可能性があり，したがって，データが与えられたときにどのモデルが適切であるかは不確実である。
同様に，将来のデータや行動の結果に関する予測も不確実である。
確率論は不確実性をモデル化するための枠組みを提供する。
<!-- The key idea behind the probabilistic framework to machine learning is that learning can be thought of as inferring plausible models to explain observed data.
A machine can use such models to make predictions about future data, and take decisions that are rational given these predictions.
Uncertainty plays a fundamental part in all of this.
Observed data can be consistent with many models, and therefore which model is appropriate, given the data, is uncertain.
Similarly, predictions about future data and the future consequences of actions are uncertain.
Probability theory provides a framework for modelling uncertainty.-->

このレビュー論文では，機械学習とベイズ推定における確率論的アプローチの紹介から始まり，この分野における最先端の進歩について論じる。
学習と知能の多くの側面は，不確実性を慎重に確率論的に表現することに決定的に依存している。
確率的アプローチが人工知能 (1), ロボット工学 (2), 機械学習 (3,4) などの分野で主流となったのはごく最近のことである。
現在でも，これらの分野では，不確実性を完全に表現することがいかに重要であるかについて論争が続いている。
例えば，音声認識(5)，画像分類(6,7)，テキスト中の単語予測 (8) などの難しいパターン認識問題を解くために深層ニューラルネットワークを利用した進歩は，そのような問題を解決していない。
これらのニューラルネットワークの構造やパラメータの不確実性をあからさまに表現している。
しかし，我々の焦点は，大量のデータが利用可能であることを特徴とするこれらのタイプのパターン認識問題ではなく，不確実性が本当に重要な要素である問題，例えば，不確実性の量に決定が依存するような問題に当てたい。
<!-- This Review starts with an introduction to the probabilistic approach to machine learning and Bayesian inference, and then discusses some of the state-of-the-art advances in the field.
Many aspects of learning and intelligence crucially depend on the careful probabilistic representation of uncertainty.
Probabilistic approaches have only recently become a mainstream approach to artificial intelligence1, robotics(2) and machine learning(3,4).
Even now, there is controversy in these fields about how important it is to fully represent uncertainty.
For example, advances using deep neural networks to solve challenging pattern-recognition problems such as speech recognition(5), image classification(6,7), and prediction of words in text(8), do not
overtly represent the uncertainty in the structure or parameters of those neural networks.
However, my focus will not be on these types of pattern recognition problems, characterized by the availability of large amounts of data, but on problems for which uncertainty is really a key ingredient, for example where a decision may depend on the amount of uncertainty.-->

確率的機械学習のフロンティアで現在研究されている 5 つの分野を取り上げ，多くの分野の科学者に広く関連する分野を強調する。
確率的プログラミングは，確率的モデルをコンピュータプログラムとして表現するための一般的な枠組みで，科学的モデリングに大きな影響を与える可能性がある。
ベイズ最適化は，未知の関数をグローバルに最適化するアプローチで，確率的データ圧縮，データからもっともらしく解釈可能なモデルの自動発見，個別化医療など多くの関連モデルを学習する階層的モデリング または推薦を行う。
しかし，今後 10 年で，確率論的な枠組みに基づく人工知能や機械学習が大きく発展することが期待されている。
<!-- I highlight five areas of current research at the frontier of probabilistic machine learning, emphasizing areas that are of broad relevance to scientists across many fields: probabilistic programming, which is a general framework for expressing probabilistic models as computer programs and which could have a major impact on scientific modelling; Bayesian optimization, which is an approach to globally optimizing unknown functions; probabilistic data compression; automating the discovery of plausible and interpretable models from data; and hierarchical modelling for learning many related models, for example for personalized medicine
or recommendation.
Although considerable challenges remain, the coming decade promises substantial advances in artificial intelligence and machine learning based on the probabilistic framework.-->

## 1. 確率的モデリングと不確実性の表現<!-- # 1. Probabilistic modelling and representing uncertainty-->

機械学習は，最も基本的なレベルでは，観測されたデータを基に，コンピュータがある課題の成績を向上させるための方法を開発しようとするものである。
例えば，自律走行車の画像から歩行者を検出する，白血病患者の遺伝子発現パターンを臨床結果ごとに分類する，英語の文章をフランス語に翻訳する，といった課題が典型的な例である。
しかし，本稿で述べるように，機械学習の対象は，こうしたパターン分類やマッピングよりもさらに広く，最適化や意思決定，データの圧縮，データから解釈可能なモデルを自動的に抽出することなどが含まれる。
<!-- At the most basic level, machine learning seeks to develop methods for computers to improve their performance at certain tasks on the basis of observed data.
Typical examples of such tasks might include detecting pedestrians in images taken from an autonomous vehicle, classifying gene-expression patterns from leukaemia patients into subtypes by clinical outcome, or translating English sentences into French.
However, as I discuss, the scope of machine-learning tasks is even broader than these pattern classification or mapping tasks, and can include optimization and decision making, compressing data and automatically extracting interpretable models from data. -->

機械学習の系において，データは重要な要素である。
しかし，データ，いわゆるビッグデータであっても，そこから知識や推論を引き出さない限り，それ自体では意味がない。
ほとんど全ての機械学習の課題は，観測されたデータから欠損データや潜在データを推論することである。
例えば，白血病の患者を，測定された遺伝子発現パターンに基づいて，4 つの主要な下位タイプのいずれかに分類することを考えてみよう。
ここで，観測されたデータは遺伝子発現パターンとラベル付けされた下位タイプの組であり，推論すべき未観測または欠損データは新しい患者の下位タイプである。
観測されたデータから未観測のデータを推論するために，学習系はいくつかの仮定をする必要があり，これらの仮定を合わせてモデルを構成する。
モデルは，古典的な統計的線形回帰モデルのような非常に単純で硬いものから，大規模で深いニューラルネットワークのような複雑で柔軟なもの，さらには無限に多くのパラメータを持つモデルもある。
この点については，次節で再び触れる。
観測されたデータで訓練されたモデルが，観測されていないデータについて予測や予想を行うことができれば，モデルはよく定義されていると考えられる (そうでなければ，哲学者カール・ポパーの仮説評価の提案の意味で，モデルが予想を行うことができなければ，反証することができないし，理論物理学者ウォルフガング・パウリがモデルは「間違ってさえいない」と言ったように，である)。
例えば，分類の設定において，明確に定義されたモデルは，新しい患者のクラスラベルの予測を提供することができるはずである。
どんな賢明なモデルでも，観測されないデータを予測するときは不確実なので，不確実性はモデリングにおいて基本的な役割を果たす。
<!-- Data are the key ingredients of all machine-learning systems.
But data, even so-called big data, are useless on their own until one extracts knowledge or inferences from them.
Almost all machine-learning tasks can be formulated as making inferences about missing or latent data from the observed data — I will variously use the terms inference, prediction or forecasting to refer to this general task.
Elaborating the example mentioned, consider classifying people with leukaemia into one of the four main subtypes of this disease on the basis of each person’s measured gene-expression patterns.
Here, the observed data are pairs of gene-expression patterns and labelled subtypes, and the unobserved or missing data to be inferred are the subtypes for new patients.
To make inferences about unobserved data from the observed data, the learning system needs to make some assumptions; taken together these assumptions constitute a model.
A model can be very simple and rigid, such as a classic statistical linear regression model, or complex and flexible, such as a large and deep neural network, or even a model with infinitely many parameters.
I return to this point in the next section.
A model is considered to be well defined if it can make forecasts or predictions about unobserved data having been trained on observed data (otherwise, if the model cannot make predictions it cannot be falsified, in the sense of the philosopher Karl Popper’s proposal for evaluating hypotheses, or as the theoretical physicist Wolfgang Pauli said the model is “not even wrong”).
For example, in the classification setting, a well-defined model should be able to provide predictions of class labels for new patients.
Since any sensible model will be uncertain when predicting unobserved data, uncertainty plays a fundamental part in modelling.-->

モデリングにおける不確実性には様々な形態がある。
最も低レベルでは，モデルの不確実性は測定ノイズ，例えば，画素のノイズや画像のぼやけからもたらされる。
より高次のレベルでは，モデルは線形回帰の係数のような多くのパラメータを持つことがあり，これらのパラメータのどの値が新しいデータを予測するのに適しているかについての不確実性が存在する。
最後に，最も高いレベルでは，線形回帰が適切か，ニューラルネットワークが適切か，後者なら何層が必要か，などモデルの一般的な構造さえ不確かなことがよくある。
<!-- There are many forms of uncertainty in modelling.
At the lowest level, model uncertainty is introduced from measurement noise, for example, pixel noise or blur in images.
At higher levels, a model may have many parameters, such as the coefficients of a linear regression, and there is uncertainty about which values of these parameters will be good at predicting new data.
Finally, at the highest levels, there is often uncertainty about even the general structure of the model: is linear regression or a neural network appropriate, if the latter, how many layers should it have, and so on.-->

モデリングに対する確率論的アプローチは，あらゆる形の不確実性を表現するために確率論を用いる(9)。
確率論は，微積分が変化率を表現し，操作するための言語であるのと同じように，不確実性を表現し，操作するための数学的言語である(10)。
幸いなことに，モデリングに対する確率論的アプローチは概念的に非常に単純で，モデル中の不確かな未観測量 (構造的，パラメトリック，雑音関連を含む) とそれらがどのようにデータに関連するかを表すために確率分布が使用される。
そして，観測されたデータから観測されない量を推測するために，確率論の基本的な規則が使用される。
データからの学習は，事前確率分布 (データを観測する前に定義) を事後分布 (データを観測した後) に変換することによって行われる。
このように，データからの学習に確率論を応用したものをベイズ学習と呼ぶ (囲み記事 1)。
<!-- The probabilistic approach to modelling uses probability theory to express all forms of uncertainty(9).
Probability theory is the mathematical language for representing and manipulating uncertainty10, in much the same way as calculus is the language for representing and manipulating rates of change.
Fortunately, the probabilistic approach to modelling is conceptually very simple: probability distributions are used to represent all the uncertain unobserved quantities in a model (including structural, parametric and noise-related) and how they relate to the data.
Then the basic rules of probability theory are used to infer the unobserved quantities given the observed data.
Learning from data occurs through the transformation of the prior probability distributions (defined before observing the data), into posterior distributions (after observing data).
The application of probability theory to learning from data is called Bayesian learning (Box 1). -->

機械知能のための確率的枠組みには，その概念の単純さ以外に，いくつかの魅力的な特性がある。
単一または少数の変数に対する単純な確率分布は，より大きく，より複雑なモデルのビルディングブロックを形成するために構成することができる。
このような構成的確率モデルの表現方法として，過去 20 年の機械学習の主流はグラフィカルモデル (1)であり，有向グラフ (ベイジアンネットワーク，信念ネットワーク)，無向グラフ (マルコフネットワーク，確率場)，有向辺と無向辺の混合グラフなどの種類がある (図1)。
後述するように，確率的プログラミングはグラフィカルモデルを一般化するエレガントな方法であり，より豊かなモデル表現が可能である。
確率モデルの構成性は，非線形力学系 (例えばリカレントニューラルネットワーク) を別のものに結合した場合にどうなるかということよりも，より大きなモデルの文脈におけるこれらの構成要素の振る舞いが，しばしばはるかに理解しやすいことを意味している。
特に，よく定義された確率モデルでは，モデルからデータを生成することが常に可能である。
このような「想像上の」データは，確率モデルの「心」を覗く窓となり，初期の事前仮定と後の段階でモデルが学習したことの両方を理解するのに役立つ。
<!-- Apart from its conceptual simplicity, there are several appealing properties of the probabilistic framework for machine intelligence.
Simple probability distributions over single or a few variables can be composed to form the building blocks of larger, more complex models.
The dominant paradigm in machine learning over the past two decades for representing such compositional probabilistic models has been graphical models1(1), with variants including directed graphs (also known as Bayesian networks and belief networks), undirected graphs (also known as Markov networks and random fields), and mixed graphs with both directed and undirected edges (Fig. 1).
As discussed later, probabilistic programming offers an elegant way of generalizing graphical models, allowing a much richer representation of models.
The compositionality of probabilistic models means that the behaviour of these building blocks in the context of the larger model is often much easier to understand than, say, what will happen if one couples a non-linear dynamical system (for example, a recurrent neural network) to another.
In particular, for a well-defined probabilistic model, it is always possible to generate data from the model; such ‘imaginary’ data provide a window into the ‘mind’ of the probabilistic model, helping us to understand both the initial prior assumptions and what the model has learned at any later stage. -->

また，確率論的モデリングは，人工知能システムにおける学習の規範となる理論であるため，他の方法と比較して概念的に優れている点もある。
人工知能システムは，データに照らして世界に関する信念をどのように表現し，更新すべきなのだろうか。
コックス公理は信念を表現するためのいくつかの望ましい条件を定義している。
これらの公理の結果として「ありえない」から「絶対確実」までの「信念の度合い」は確率論のすべての規則に従わなければならない (10,12,13)。
このことは，人工知能において主観的なベイズ型確率表現を用いることを正当化する。
決定論を動機とする人工知能におけるベイズ表現の論拠は，「オランダ本定理」によって与えられる。
この主張は，エージェントの信念の強さは，様々なオッズ (ペイオフの比率) での賭けを受け入れるかどうかをエージェントに尋ねることで評価できるという考えに基づいている。
「オランダ人の賭けの定理」は，人工知能システムの (あるいは人間の) 信念の度合いが確率の規則と一致しない限り，損をすることが確実な賭けも喜んで引き受けるというものである (14)。
知能にとって不確実性を原理的に取り扱うことが重要であるというこれらや他の多くの議論の力により，ベイズ型確率モデルは人工知能システムの合理性の理論的基礎としてだけでなく，人間や動物の規範的行動のモデルとしても登場し (15-18) (ただし議論は文献 19, 20 参照)，神経回路がベイズ推定をいかに実装するかの理解に多くの研究がなされている(21, 22)。
<!-- Probabilistic modelling also has some conceptual advantages over alternatives because it is a normative theory for learning in artificially intelligent systems.
How should an artificially intelligent system represent and update its beliefs about the world in light of data?
The Cox axioms define some desiderata for representing beliefs; a consequence of these axioms is that ‘degrees of belief’, ranging from ‘impossible’ to ‘absolutely certain’, must follow all the rules of probability theory(10,12,13).
This justifies the use of subjective Bayesian probabilistic representations in artificial intelligence.
An argument for Bayesian representations in artificial intelligence that is motivated by decision theory is given by the Dutch book theorem.
The argument rests on the idea that the strength of beliefs of an agent can be assessed by asking the agent whether it would be willing to accept bets at various odds (ratios of payoffs).
The Dutch book theorem states that unless an artificial intelligence system’s (or human’s, for that matter) degrees of beliefs are consistent with the rules of probability it will be willing to accept bets that are guaranteed to lose money(14).
Because  of the force of these and many other arguments on the importance of a principled handling of uncertainty for intelligence, Bayesian probabilistic modelling has emerged not only as the theoretical foundation for rationality in artificial intelligence systems, but also as a model for normative behaviour in humans and animals(15–18) (but see refs 19, 20 for a discussion), and much research is devoted to understanding how neural circuitry may be implementing Bayesian inference(21,22). -->

機械学習における完全な確率論的アプローチは，概念的には単純であるが，計算上およびモデル化上の多くの課題をもたらす。
計算上の主な課題は，学習がモデル中の注目する変数以外のすべての変数を周辺化 (総和) することである (囲み記事 1)。
このような高次元の和や積分は一般に計算が難しく，多くのモデルでそれらを正確に実行する多項式時間アルゴリズムが知られていない，という意味である。
幸い，<font style="color: blue;font-weight: 900">マルコフ連鎖モンテカルロ法 (MCMC)，変分近似，期待値伝播，逐次モンテカルロ法</font> など，多くの近似的な積分アルゴリズムが開発されている (23-26)。
ベイズ流機械学習が他の機械学習と異なる点として，計算技術が挙げられる。
ベイズ流確率の研究者にとって，主な計算問題は統合であるが，他の多くの研究者にとっては，モデルパラメータの最適化が焦点となる。
しかし，この 2 分法は見た目ほど厳密なものではない。
多くの勾配に基づく最適化手法は，Langevin やハミルトンモンテカルロ法を用いることで統合手法に変えることができる (27,28)。
統合問題は変分近似を用いることで最適化問題に変えることができる (24)。
最適化については，後節で再確認する。
<!-- Although conceptually simple, a fully probabilistic approach to machine learning poses a number of computational and modelling challenges.
Computationally, the main challenge is that learning involves marginalizing (summing out) all the variables in the model except for the variables of interest (Box 1).
Such high-dimensional sums and integrals are generally computationally hard, in the sense that for many models there is no known polynomial time algorithm for performing them exactly.
Fortunately, a number of approximate integration algorithms have been developed, including Markov chain Monte Carlo (MCMC) methods, variational approximations, expectation propagation and sequential Monte Carlo(23–26).
It is worth noting that computational techniques are one area in which Bayesian machine learning differs from much of the rest of machine learning: for Bayesian researchers the main computational problem is integration, whereas for much of the rest of the community the focus is on optimization of model parameters.
However, this dichotomy is not as stark as it appears: many gradient-based optimization methods can be turned into integration methods through the use of Langevin and Hamiltonian Monte Carlo methods(27,28), while integration problems can be turned into optimization problems through the use of variational approximations(24).
I revisit optimization in a later section. -->

確率的機械学習の主なモデリング上の課題は，対象となる予測課題を達成するために必要なデータの特性をすべて把握できるような柔軟なモデルでなければならないということである。
この課題に対処するための一つのアプローチは，データに対して複雑さを適応できるモデルのオープンエンドな宇宙を包含する事前分布を開発することである。
データに応じて複雑さを増す柔軟なモデルの基礎となる重要な統計的概念は，ノンパラメトリックである。
<!-- The main modelling challenge for probabilistic machine learning is that the model should be flexible enough to capture all the properties of the data required to achieve the prediction task of interest.
One approach to addressing this challenge is to develop a prior distribution that encompasses an open-ended universe of models that can adapt in complexity to the data.
The key statistical concept underlying flexible models that grow in complexity with the data is non-parametrics. -->


## 2. ノンパラメトリックによる柔軟性
<!-- # 2. Flexibility through non-parametrics-->

現代の機械学習の教訓の 1 つは，特に大規模なデータセットから学習する場合，柔軟性の高い学習系から最高の予測性能が得られることが多いということである。
柔軟性の高いモデルは，より優れた予測を行うことができる。
なぜなら，より高度にデータを「自説」に従わせることができるからである。
(しかし，すべての予測は仮定を含んでおり，したがって，データは決して排他的に「自分自身のために語る」わけではないことに注意されたい)。
柔軟性を実現するには、基本的に 2 つの方法がある。
モデルはデータセットに比べて多数のパラメータを持つことができる (例えば，最近，英語とフランス語の文の翻訳に使われたニューラルネットワークは，3 億 8400 万個のパラメータを持つ確率的モデルである (29)。
あるいは，ノンパラメトリックな構成要素を用いてモデルを定義することも可能である。
<!-- One of the lessons of modern machine learning is that the best predictive performance is often obtained from highly flexible learning systems, especially when learning from large data sets.
Flexible models can make better predictions because to a greater extent they allow data to ‘speak for themselves’.
(But note that all predictions involve assumptions and therefore the data are never exclusively ‘speaking for themselves’.)
There are essentially two ways of achieving flexibility.
The model could have a large number of parameters compared with the data set (for example, the neural network recently used to achieve translations of English and French sentences that approached the accuracy of state-of-the-art methods is a probabilistic model with 384 million parameters(29).
Alternatively, the model can be defined using non-parametric components. -->

ノンパラメトリックモデルを理解する最も良い方法は、パラメトリックモデルと比較することである。
パラメトリックモデルでは，固定された有限の数のパラメータがあり，いくら学習データを観測しても，データができることは，将来の予測を制御するこれらの有限の数のパラメータを設定することだけである。
これに対し，ノンパラメトリックアプローチでは，パラメトリックモデルを入れ子にしてパラメータを増やすか，無限に多くのパラメータを持つモデルから始めることで，学習データの量に応じて予測が複雑化する。
例えば，分類問題において，線形 (パラメトリック) 分類器は常にクラス間の線形境界を用いて予測を行うのに対し，ノンパラメトリック分類器は，データが増えるにつれて形状が複雑になる非線形の境界を学習することが可能である。
多くのノンパラメトリックモデルは，パラメトリックモデルから出発し，モデルが無限大のパラメータを持つ限界まで成長したときに何が起こるかを考えることで導き出すことができる (30)。
明らかに，有限の学習データに無限に多くのパラメータを持つモデルを当てはめると「過学習」になる。つまり，モデルの予測は，テストデータに一般化できる規則性ではなく，学習データの癖を反映しているかもしれない。
幸いなことに，ベイズアプローチはパラメータを適合させるのではなく，平均化するため，このようなオーバーフィットに陥ることはない (囲み記事 1)。
さらに，多くの応用例では，膨大なデータセットがあるため，過学習よりも，単純化しすぎたパラメトリックモデルの選択による未学習が主な懸念となる。
<!-- The best way to understand non-parametric models is through comparison to parametric ones.
In a parametric model, there are a fixed, finite number of parameters, and no matter how much training data are observed, all the data can do is set these finitely many parameters that control future predictions.
By contrast, non-parametric approaches have predictions that grow in complexity with the amount of training data, either by considering a nested sequence of parametric models with increasing numbers of parameters or by starting out with a model with infinitely many parameters.
For example, in a classification problem, whereas a linear (parametric) classifier will always make predictions using a linear boundary between classes, a non-parametric classifier can learn a non-linear boundary whose shape becomes more complex with more data.
Many non-parametric models can be derived starting from a parametric model and considering what happens as the model grows to the limit of infinitely many parameters(30).
Clearly, fitting a model with infinitely many parameters to finite training data would result in ‘overfitting’, in the sense that the model’s predictions might reflect quirks of the training data rather than regularities that can be generalized to test data.
Fortunately, Bayesian approaches are not prone to this kind of overfitting since they average over, rather than fit, the parameters (Box 1).
Moreover, for many applications we have such huge data sets that the main concern is underfitting from the choice of an overly simplistic parametric model, rather than overfitting. -->

<center>
<div class="footnote">

#### 囲み記事 1: ベイズ機械学習 Bayesian machine learning

確率論の根底には、2 つの単純な規則がある: 「和の法則」とは:
<!-- There are two simple rules that underlie probability theory: the sum rule:-->

$$ P(x)=\sum_{y\in Y}P(x,y), $$

そして，「積の法則」とは:
<!-- and the product rule: -->

$$ P(x,y) = P(x) P(y\vert x). $$

ここで $x$ と $y$ は観測された量または不確実な量に対応し，それぞれある集合 $X$ と $Y$ に値をとる。
例えば，$x$ と $y$ はそれぞれケンブリッジとロンドンの天気に関するもので，どちらも集合 $X=Y=${雨，曇，晴} の値をとる。
$P(x)$ は $x$ の確率に相当し，特定の値を観測する頻度についての記述でも，それについての主観的な信念でもよい。
$P(x,y)$ は $x$ と $y$ を観測する合同確率，$P(y\vert x)$ は $x$ の値を観測することを条件とした $y$ の確率である。
和の法則は，$x$ のマージンは $y$ 上の結合を和 (連続変数の場合は積分) することで得られるとするものである。
積の法則は，結合が周辺と条件付の積として分解されることを述べる。
ベイズ則はこの 2 つの法則の帰結である。
<!-- Here $x$ and $y$ correspond to observed or uncertain quantities, taking values in some sets $X$ and $Y$, respectively.
For example, $x$ and $y$  might relate to the weather in Cambridge and London, respectively, both taking values in the set $X = Y =$ {rainy,cloudy,sunny}.
$P(x)$ corresponds to the probability of $x$, which can be either a statement about the frequency of observing a particular value, or a subjective belief about it.
$P(x,y)$ is the joint probability of observing $x$ and $y$, and $P(y|x)$ is the probability of $y$ conditioned on observing the value of $x$.
The sum rule states that the marginal of $x$ is obtained by summing (or integrating for continuous variables) the joint over $y$.
The product rule states that the joint can be decomposed as the product of the marginal and the conditional.
Bayes rule is a corollary of these two rules: -->

$$ P(y\vert x)=\frac{P(x\vert y)P(y)}{P(x)}=\frac{P(x\vert y)P(y)}{\sum_{y\in Y}P(x,y)} $$

$x$ を $\mathcal{D}$ に置き換えて観測データを，$y$ を $\theta$ に置き換えてモデルの未知パラメータを，全ての項を $m$ (我々が考える確率モデルのクラス) で条件付けることで，確率論を機械学習に適用することができる。
学習の場合，次のようになる。
<!-- We can apply probability theory to machine learning by replacing the symbols above: we replace $x$ by $D$ to denote the observed data, we replace $y$ by $\theta$ to denote the unknown parameters of a model, and we condition all terms on $m$, the class of probabilistic models we are considering.
For learning, we thus get: -->

$$ P(\theta\vert \mathcal{D},m)=\frac{P(\mathcal{D}\vert \theta,m) P(\theta\vert m)}{P(\mathcal{D}\vert m)} $$

ここで、$P(D\vert\theta,m)$ はモデル $m$ におけるパラメータ $\theta$ の尤度，$P( \theta\vert m)$ は$\theta$ の事前確率，$P( \theta\vert \mathcal{D}, m)$ はデータ $\mathcal{D}$ を与えたときの $\theta$の事後確率である。
<!-- where $P(D\vert\theta,m)$ is the likelihood of parameters $\theta$ in model $m$, $P(\theta\vert m)$ is the prior probability of $\theta$ and $P(\theta\vert D, m)$ is the posterior of $\theta$ given data $D$.-->

例えば，データ $\mathcal{D}$ はケンブリッジとロンドンの天気を 1 時間毎に観測した時系列であり、モデルは両地点の連続する時間での合同天気パターンを捉えようとし，パラメータ $\theta$ は時間と空間の相関をモデリングする。
学習とは，データ $\mathcal{D}$ を通して，パラメータ $P( \theta\vert m)$ に関する事前知識または仮定を，パラメータに関する事後知識 $P( \theta\vert\mathcal{D},m)$ に変換することである。
この事後知識は，将来のデータに対して使用される事前知識となる。
学習されたモデルは、新しい未知のテストデータ $\mathcal{D}_\text{test}$ に対して，和と積の法則を適用して予測を得るだけで，予測や予想に用いることができます。
<!-- For example, the data $D$ might be a time series of hourly observations of the weather in Cambridge and London, and the model might attempt to capture the joint weather patterns at both locations over successive hours, with parameters $\theta$ modelling correlations over time and space.
Learning is the transformation of prior knowledge or assumptions about the parameters $P(\theta\vert m)$, through the data $D$, into posterior knowledge about the parameters, $P(\theta\vert D,m)$.
This posterior is now the prior to be used for future data.
A learned model can be used to predict or forecast new unseen test data, $D_\text{test}$, by simply applying the sum and product rule to get the prediction: -->

$$ P(\mathcal{D}_{\text{test}}\vert \mathcal{D},m) = \int P(\mathcal{D}_{\text{test}}\vert \theta, \mathcal{D}, m) P(\theta\vert \mathcal{D},m) d\theta $$

最後に $m$ のレベルでベイズ則を適用することで，異なるモデルを比較することができる。
<!-- Finally, different models can be compared by applying Bayes rule at the level of m: -->

$$\begin{aligned}
P(m|\mathcal{D}) &= \frac{P(\mathcal{D}|m)P(m)}{P(\mathcal{D})}\\
P(\mathcal{D}|m) &= \int P(\mathcal{D}|\theta,m) P(\theta|m) d\theta
\end{aligned}$$

$P(\mathcal{D}|m)$ は周辺尤度またはモデル証拠であり，ベイズのオッカムの剃刀として知られるより単純なモデルの優先順位を実装している。
<!-- The term $P(\mathcal{D}\vert m)$ is the marginal likelihood or model evidence, and implements a preference for simpler models known as Bayesian Ockham’s razor. -->
</div>
</center>

<!-- <center> -->
<div class="figcenter">
<img src="../figures/2015Ghahramani_fig1.svg" width="66%"><br/>
<div class="figcaption">
<!-- <div style="text-align: left;width: 94%;background-color: cornsilk"> -->

図 1. ベイズ推論。
ベイズ推論を医療診断の問題に適用した簡単な例。
ここでは，患者の症状と，潜在的にはこの病気に対する素因 (gen pred) を示す患者の遺伝子マーカー測定値からの情報を用いて，希少な病気を診断することが問題である。
この例では，すべての変数が 2 値であると仮定する。
T は真，F は偽である。
変数間の関係は有向矢印で示され，各変数が直接依存する他の変数が与えられた場合の確率も示されている。
黄色のノードは測定可能な変数を表し，緑のノードは隠れ変数を表す。
<!-- Figure 1. Bayesian inference.
A simple example of Bayesian inference applied to a medical diagnosis problem.
Here the problem is diagnosing a rare disease using information from the patient’s symptoms and, potentially, the patient’s genetic marker measurements, which indicate predisposition (gen pred) to this disease.
In this example, all variables are assumed to be binary.
T, true; F, false.
The relationships between variables are indicated by directed arrows and the probability of each variable given other variables they directly depend on is also shown.
Yellow nodes denote measurable variables, whereas green nodes denote hidden variables.-->
囲み記事 1 の「加算則」を用いて，その患者が希少疾患の罹患確率は，以下のように計算される:
<!-- Using the sum rule (Box 1), the prior probability of the patient having the rare disease is:  -->
$P(\text{希少疾患}=\text{真})=P(\text{希少疾患}=\text{真}\vert\text{遺伝子検査予測}=\text{真}) p(\text{遺伝子検査予測}=\text{真}) + p(\text{希少疾患}=\text{真}\vert\text{遺伝子検査予測}=\text{偽}) p(\text{陰電子検査予測}=\text{偽})=1.1\times10^{-5}$
<!-- P(rare disease = T) = P(rare disease = T|gen pred = T) p(gen pred = T) + p(rare disease = T|gen pred = F) p(gen pred = F) = 1.1 × 10−5.  -->
ベイズ則を適用すると、症状が観察された患者について，その希少疾患の確率は:
<!-- Applying Bayes rule we find that for a patient observed to have the symptom, the probability of the rare disease is:  -->
$p(\text{希少疾患}=\text{真}\vert\text{症状}=\text{真})=8.8\times10^{-4}$
<!-- p(rare disease = T|symptom = T) = 8.8 × 10−4, whereas for a patient observed to have the genetic marker (gen marker) it is p(rare disease = T|gen marker = T) = 7.9 × 10−4.  -->
患者が症状と遺伝子マーカーを両方持っていると仮定すると，希少疾患の確率は `p(rare disease=T|symptom=T, gen marker=T)=0.06` に増加する。
ここでは，固定された既知のモデルパラメータ，すなわち，$\theta=(10^{-4}, 0.1, 10^{-6}, 0.8, 0.01, 0.8, 0.01)$ という数字を示した。
しかし，これらのパラメータとモデルの構造 (矢印や追加の隠れ変数の有無) の両方は，囲み記事  1 の方法を用いて患者記録のデータセットから学習することが可能である。
<!-- Assuming that the patient has both the symptom and the genetic marker the probability of the rare disease increases to p(rare disease = T|symptom = T, gen marker = T) = 0.06.
Here, we have shown fixed, known model parameters, that is, the numbers θ=(10−4, 0.1, 10−6, 0.8, 0.01, 0.8, 0.01).
However, both these parameters and the structure of the model (the presence or absence of arrows and additional hidden variables) could be learned from a data set of patient records using the methods in Box 1.  -->
</div>
</div>
<!-- </center> -->

<!-- <center> -->
<div class="footnote">

### Dutch Book Arguments
- source: https://plato.stanford.edu/entries/dutch-book/#ProbAxioDutcBookTheo

確率論（エージェントの信念の程度は確率の公理を満たすべきであるという考え方）に対するオランダ人の賭け論法 (ダッチブック論法 DBA:Dutch Book Argument) は，Ramsey の「真実と確率」での研究に端を発するものである。Ramsey は，確率公理に違反するエージェントは，自分に対して賭けられる可能性があることをほんの少し述べているが，このことは Ramsey が何を示したかったのか，また，この議論の説得力のあるバージョンが与えられるのか，どのように与えられるのか，かなりの議論と混乱につながった。その中には，エージェントの現在の信念にさらなる制約を加えるものや，信念の程度が時間とともにどのように変化すべきかを規定すると主張する条件化などの原則がある。
<!-- The Dutch Book argument (DBA) for probabilism (namely the view that an agent’s degrees of belief should satisfy the axioms of probability) traces to Ramsey’s work in “Truth and Probability”. He mentioned only in passing that an agent who violates the probability axioms would be vulnerable to having a book made against him and this has led to considerable debate and confusion both about exactly what Ramsey intended to show and about if, and how, a cogent version of the argument can be given.
The basic idea behind the argument has also been applied in defense of a variety of principles, some of which place additional constraints on an agent’s current beliefs, with others, such as Conditionalization, purporting to govern how degrees of belief should evolve over time.-->

### オランダ人の賭けの定理
DBA 自体は，いわゆるオランダ人の賭けの定理から始まり，ある賭け(ベット)の集合が一方に純損益を保証する条件，つまりオランダ人の賭けに関するものである。ここでは，de Finetti とともに，命題 H に対する賭けは，次のような正準形式を持つ取り決めであると仮定する。
<!-- The DBA itself begins with the so-called Dutch Book theorem, which concerns the conditions under which a set of bets guarantees a net loss to one side, or a Dutch Book.
With de Finetti, it is here assumed that a bet on a proposition H is an arrangement that has the following canonical form:-->

<div style="text-align:center;width:22%">

|H   |Payoff|
|:---:|:---:|
|True  |  S−qS|
|False |  −qS|
</div>
この表は，H が真であれば S が得られる賭け金 S を qS で買ったエージェントの正味のペイオフを表している。S は賭け金と呼ばれ，H が真の場合のペイオフと，H が偽の場合の没収額を合わせた，賭けに関わる総金額である。量 q は賭け金商と呼ばれ，H が偽の場合の損失額を賭け金で割ったものである。これで，ダッチブック定理を述べることができる。
<!-- The table gives the net payoff to an agent who buys a bet with stake S for the price qS, where S is won if H is true.S is called the stake, as it is the total amount involved in the wager, that is the payoff in the case that H is true together with the amount forfeited if H is false. The quantity q is called the betting quotient, which is the amount lost if H is false divided by the stake. The Dutch Book theorem can now be stated:-->

確率公理を満たさないベット商のセットがあるとき，そのベット商で片方の正味の負けを保証するベットのセットが存在する。
<!-- Given a set of betting quotients that fails to satisfy the probability axioms, there is a set of bets with those quotients that guarantees a net loss to one side. -->
</div>
<!-- </center> -->

ベイズ型ノンパラメトリックの完全な議論はこの総説論文の範囲外であるが (これについては文献 9, 31,32 参照)，いくつかの重要なモデルについて言及することは価値があると思われる。ガウス過程は未知の関数に対する非常に柔軟なノンパラメトリックモデルであり，回帰，分類，その他関数に対する推論を必要とする多くの応用に広く使用されている (33)。例えば，ある化学物質の投与量とその化学物質に対する生物の反応とを関連付ける関数を学習することを考えてみよう。この関係を，例えば線形パラメトリック関数でモデル化する代わりに，ガウス過程を用いて，データと一致する非線形関数のノンパラメトリック分布を直接学習することができる。ガウス過程の最近の応用例として，人間やディープラーニング手法を凌駕する顔認識への最先端アプローチである GaussianFace が注目されている(34)。ディリクレ過程は統計学で長い歴史を持つノンパラメトリックモデルで (35)，密度推定，クラスタリング，時系列分析，文書のトピックのモデル化などに利用されている(36)。ディリクレ過程を説明するために，各人が多くのコミュニティの一つに属することができる社会的ネットワークにおける友人関係のモデル化への応用を考えてみよう。ディリクレ過程を用いると，推定されるコミュニティ (つまりクラスタ) の数が人数に比例して増加するモデルが可能になる (37)。ディリクレ過程は遺伝子発現パターンのクラスタリングにも利用されている (38,39)。Indian buffet process (IBP)(40) はノンパラメトリックモデルであり，潜在特徴モデリング，重複クラスタの学習，疎行列因子分解，またはディープネットワークの構造をノンパラメトリックに学習するために用いることができる(41)。社会ネットワークのモデリングの例を詳しく述べると，IBP に基づくモデルでは，各人が単一のコミュニティではなく，（例えば，異なる家族，職場，学校，趣味などによって定義される) 多数の潜在的コミュニティのある部分集合に属することができ 2 人の間の友情の確率は，彼らが持つ重複するコミュニティの数に依存する(42)。この場合，各人の潜在的特徴はコミュニティと対応し，直接観測されることは想定されていない。IBP は，ベイズ型ノンパラメトリックモデルに，ニューラルネットワークの文献で広まった「分散表現」を付与する方法と考えることができる (43)。ベイズ型ノンパラメトリックとニューラルネットワークの間の興味深い結びつきは，かなり一般的な条件下で，無限個の隠れユニットを持つニューラルネットワークがガウス過程と等価であることである(44)。上記のノンパラメトリックの構成要素は，前述したように，より複雑なモデルに構成することができるビルディングブロックとして再度考える必要があることに注意されたい。次節では，さらに強力なモデル構成法である確率的プログラ ミングについて述べる。
<!-- A full discussion of Bayesian non-parametrics is outside the scope of this Review (see refs 9, 31, 32 for this), but it is worth mentioning a few of the key models.
Gaussian processes are a very flexible non-parametric model for unknown functions, and are widely used for regression, classification, and many other applications that require inference on functions (33).
Consider learning a function that relates the dose of some chemical to the response of an organism to that chemical.
Instead of modelling this relationship with, say, a linear parametric function, a Gaussian process could be used to learn directly a non-parametric distribution of non-linear functions consistent with the data.
A notable example of a recent application of Gaussian processes is GaussianFace, a state-of-the-art approach to face recognition that outperforms humans and deep-learning methods(34).
Dirichlet processes are a non-parametric model with a long history in statistics(35) and are used for density estimation, clustering, time-series analysis and modelling the topics of documents(36).
To illustrate Dirichlet processes, consider an application to modelling friendships in a social network, where each person can belong to one of many communities.
A Dirichlet process makes it possible to have a model whereby the number of inferred communities (that is, clusters) grows with the number of people(37).
Dirichlet processes have also been used for clustering gene-expression patterns(38,39).
The Indian buffet process (IBP)(40) is a non-parametric model that can be used for latent feature modelling, learning overlapping clusters, sparse matrix factorization, or to non-parametrically learn the structure of a deep network(41).
Elaborating the social network modelling example, an IBP-based model allows each person to belong to some subset of a large number of potential communities (for example, as defined by different families, workplaces, schools, hobbies, and so on) rather than a single community, and the probability of friendship between two people depends on the number of overlapping communities they have(42).
In this case, the latent features of each person correspond to the communities, which are not assumed to be observed directly.
The IBP can be thought of as a way of endowing Bayesian non-parametric models with ‘distributed representations’, as popularized in the neural network literature(43).
An interesting link between Bayesian non-parametrics and neural networks is that, under fairly general conditions, a neural network with infinitely many hidden units is equivalent to a Gaussian process(44).
Note that the above non-parametric components should be thought of again as building blocks, which can be composed into more complex models as described earlier.
The next section describes an even more powerful way of composing models — through probabilistic programming.  -->


## 3. 確率的プログラミング<!--  ## 3. Probabilistic programming -->

確率的プログラミングの基本的な考え方は、コンピュータ・プログラムを使って確率的モデル(http://probabilistic-programming.org) を表現することである (45-47)。その一つの方法は，コンピュータ・プログラムに確率モデルからのデータの生成器，つまりシミュレータを定義することである (図 2)。このシミュレータは，乱数発生器を呼び出し，シミュレータを繰り返し実行することで，モデルから異なる可能性のあるデータセットをサンプリングするようにする。コンピュータプログラムでは，再帰 (関数が自分自身を呼び出すこと) や制御フロー文 (例えば，プログラムが辿ることのできる複数の経路をもたらす if 文) など，有限グラフでは表現が困難または不可能な構成が可能なため，このシミュレーションの枠組みは，前述のグラフモデルの枠組みよりも一般的である。実際，チューリング完全言語の拡張に基づく最近の確率的プログラミング言語の多く (よく使われる言語のほとんどを含む) では，計算可能なあらゆる確率分布を確率的プログラムとして表現することが可能である (48)。
<!--The basic idea in probabilistic programming is to use computer programs to represent probabilistic models (http://probabilistic-programming.org) (45–47). One way to do this is for the computer program to define a generator for data from the probabilistic model, that is, a simulator (Fig. 2). This simulator makes calls to a random number generator in such a way that repeated runs from the simulator would sample different possible data sets from the model. This simulation framework is more general than the graphical model framework described previously since computer programs can allow constructs such as recursion (functions calling themselves) and control flow statements (for example, ‘if ’ statements that result in multiple paths a program can follow), which are difficult or impossible to represent in a finite graph. In fact, for many of the recent probabilistic programming languages that are based on extending Turing-complete languages (a class that includes almost all commonly used languages), it is possible to represent any computable probability distribution as a probabilistic program(48). -->

<div class="figcenter">
<img src="../figures/2015Ghahramani_fig2.svg" width="88%"><br/>
<div class="figcaption">
<!-- <div style="text-align:left;width:88%;background-color:cornsilk"> -->

図 2. 確率的プログラミング。
Julia による確率的プログラム (左) は，単純な 3 状態の隠れマルコフモデル (HMM) を定義するもので，文献の例から着想を得ている(62)。
HMM は逐次データや時系列データに対して広く使われている確率モデルで，データが離散的な数の隠れ状態の間を確率的に遷移することによって得られたと仮定している(98)。
最初の 4 行はモデルのパラメータとデータを定義する。
ここで trans は 3×3 の状態遷移行列，initial は初期状態分布，statesmean は 3 つの状態それぞれの平均観測値で，実際の観測値はこの平均値をガウスノイズでノイズ化したものと仮定している。
関数  hmm は HMM の定義を開始し，@assume 文で状態列を描き，@observe 文で観測データに対する条件付けを行う。
最後に @predict で状態とデータを推論する。
この推論は普遍推論エンジンによって自動的に行われ，このコンピュータプログラムの構成に対して推論を行う。
このプログラムは HMM のパラメータを固定ではなく，未知とするように変更することは容易である。
HMM 確率モデルに対応するグラフモデル (右)。パラメータ(青)，隠れ状態変数 (緑)，観測データ (黄) 間の依存性を示す。
このグラフィカルなモデルは，確率モデルの構成的な性質を強調している。
<!-- Figure 2. Probabilistic programming.
A probabilistic program in Julia (left) defining a simple three-state hidden Markov model (HMM), inspired by an example in ref. 62.
The HMM is a widely used probabilistic model for sequential and time-series data, which assumes the data were obtained by transitioning stochastically between a discrete number of hidden states(98).
The first four lines define the model parameters and the data. Here ‘trans’ is the 3 × 3 state-transition matrix, ‘initial’ is the initial state distribution, and ‘statesmean’ are the mean observations for each of the three states; actual observations are assumed to be noisy versions of this mean with Gaussian noise. The function hmm starts the definition of the HMM, drawing the sequence of states with the @assume statements, and conditioning on the observed data with the @observe statements.
Finally @predict states that we wish to infer the states and data; this inference is done automatically by the universal inference engine, which reasons over the configurations of this computer program. It would be trivial to modify this program so that the HMM parameters are unknown rather than fixed.
A graphical model (right) corresponding to the HMM probabilistic program showing dependencies between the parameters (blue), hidden state variables (green) and observed data (yellow).
This graphical model highlights the compositional nature of probabilistic models. -->
</div>
</div>
</div>

確率的プログラミングの可能性を最大限に引き出すには、観測されたデータを条件として、モデル中の観測されない変数を推測するプロセスを自動化することである (囲み記事 1)。
概念的には，観測データと一致するデータを生成するプログラムの入力状態を計算する必要がある。
通常，プログラムは入力から出力に向かうと考えるが，条件付けは，あるプログラムの出力に一致する入力 (特に乱数呼び出し) を推測するという逆問題を解くことになる。
このような条件付けは「万能推論エンジン」によって行われ，通常，観測データと一致するシミュレータプログラムの実行可能回数に対するモンテカルロ・サンプリングによって実装される。
コンピュータ・プログラムに対してこのような普遍的推論アルゴリズムを定義することが可能であるという事実はいささか驚きであるが，これは棄却サンプリング，逐次モンテカルロ法 (25)，近似ベイズ計算 (49) といったサンプリングの主要な考え方の一般性に関連している。
<!-- The full potential of probabilistic programming comes from automating the process of inferring unobserved variables in the model conditioned on the observed data (Box 1).
Conceptually, conditioning needs to compute input states of the program that generate data matching the observed data. Whereas normally we think of programs running from inputs to outputs, conditioning involves solving the inverse problem of inferring the inputs (in particular the random number calls) that match a certain program output.
Such conditioning is performed by a ‘universal inference engine’, usually implemented by Monte Carlo sampling over possible executions of the simulator program that are consistent with the observed data. The fact that defining such universal inference algorithms for computer programs is even possible is somewhat surprising, but it is related to the generality of certain key ideas from sampling such as rejection sampling, sequential Monte Carlo methods(25) and ‘approximate Bayesian computation’(49). -->

例として，未測定の転写因子と特定の遺伝子の発現レベルを関連付ける遺伝子制御モデルをシミュレートする確率的プログラムを書くとする。
モデルの各部分の不確実性は，シミュレータで使用される確率分布によって表現されるだろう。
そして，万能推論エンジンは，このプログラムの出力を測定された発現レベルに条件付けし，測定されていない転写因子の活性や他の不確実なモデルパラメータを自動的に推論することができる。
確率的プログラミングの別の応用例として，コンピュータグラフィックスプログラムの逆バージョンとしてコンピュータビジョンシステムを実装したものがある (50)。
<!-- As an example, imagine you write a probabilistic program that simulates a gene regulatory model that relates unmeasured transcription factors to the expression levels of certain genes.
Your uncertainty in each part of the model would be represented by the probability distributions used in the simulator.
The universal inference engine can then condition the output of this program on the measured expression levels, and automatically infer the activity of the unmeasured transcription factors and other uncertain model parameters.
Another application of probabilistic programming implements a computer vision system as the inverse of a computer graphics program(50).-->

確率的プログラミングが機械知能や科学的モデリングにとって革命的であると証明できる理由はいくつかある (その可能性は米国国防高等研究計画局も注目しており，現在「機械学習を進めるための確率的プログラミング」という主要プログラムに資金援助を行っている)。
第一に，汎用的な推論エンジンにより，モデルの推論方法を人手で導き出す必要がなくなる。
一般に，推論手法の導出と実装は，モデリングにおいて最も速度が遅く，バグが発生しやすいステップであり，しばしば数ヶ月かかる。
このステップを数分から数秒で済むように自動化すれば，機械学習システムの展開が大幅に加速される。
第二に，確率的プログラミングは，データのさまざまなモデルのプロトタイピングとテストを迅速に行うことができるため，科学分野での変革をもたらす可能性がある。
確率的プログラミング言語は，モデルと推論手順の間に非常に明確な分離を作り出し，モデルベース思考を促進する(51)。
確率的プログラミング言語の数は増え続けている。
BUGS(52), Stan(53), AutoBayes(54), Infer.NET(55) はチューリング完全言語に基づくシステムと比較して，限られたクラスのモデルのみを表現することが可能である。
この制限の代償として，これらの言語での推論は IBAL(57), BLOG(58), Church(59), Figaro(60), Venture(61), Anglican(62) などの一般言語(56)よりはるかに速くなることがある。
最近の研究では，一般言語における高速な推論に重点が置かれている (例えば 文献 63 参照)。
不確実性を自動推論するための他の首尾一貫した枠組みを作るのが難しいので，確率的プログラミングのほぼ全てのアプローチはベイズである。
注目すべき例外は Theano のようなシステムで，それ自身は確率的プログラミング言語ではないが，記号微分を用いてニューラルネットワークや他の確率モデルのパラメータの最適化を高速化し，自動化する (64)。
<!-- There are several reasons why probabilistic programming could prove to be revolutionary for machine intelligence and scientific modelling (its potential has been noticed by US Defense Advanced Research Projects Agency, which is currently funding a major programme called Probabilistic Programming for Advancing Machine Learning).
First, the universal inference engine obviates the need to manually derive inference methods for models.
Since deriving and implementing inference methods is generally the most rate-limiting and bug-prone step in modelling, often taking months, automating this step so that it takes minutes or seconds will greatly accelerate the deployment of machine learning systems.
Second, probabilistic programming could be potentially transformative for the sciences, since it allows for rapid prototyping and testing of different models of data.
Probabilistic programming languages create a very clear separation between the model and the inference procedures, encouraging model-based thinking(51).
There are a growing number of probabilistic programming languages.
BUGS(52), Stan(53), AutoBayes(54) and Infer.NET(55) allow only a restrictive class of models to be represented compared with systems based on Turing-complete languages.
In return for this restriction, inference in such languages can be much faster than for the more general languages(56), such as IBAL(57), BLOG(58), Church(59), Figaro(60), Venture(61), and Anglican(62).
A major emphasis of recent work is on fast inference in general languages (see for example ref.63).
Nearly all approaches to probabilistic programming are Bayesian since it is hard to create other coherent frameworks for automated reasoning about uncertainty.
Notable exceptions are systems such as Theano, which is not itself a probabilistic programming language but uses symbolic differentiation to speed up and automate optimization of parameters of neural networks and other probabilistic models(64).-->

パラメータ最適化は確率モデルの改善によく使わる。次節では，確率モデルを使って最適化を改善する方法についての最近の研究を説明する。
<!-- Although parameter optimization is commonly used to improve probabilistic models, in the next section I will describe recent work on how probabilistic modelling can be used to improve optimization.-->

## 4. ベイズ最適化<!-- ## 4. Bayesian optimization-->

評価するのに高価な未知の関数の大域的な最大値を求めるという非常に一般的な問題を考えてみよう (例えば，関数を評価するには多くの計算を行うか，実験を行う必要がある)。
数学的には，領域 $X$ 上の関数 $f$ に対して，大域的な最大値 $x^{\star}$ を求めることが目的である。
<!-- Consider the very general problem of finding the global maximum of an unknown function that is expensive to evaluate (say, evaluating the function requires performing lots of computation or conducting an experiment).
Mathematically, for a function f on a domain X, the goal is to find a global maximizer x*:-->

$$ x^{\star}=\arg\max_{x\in X} f(x). $$

ベイズ最適化では，これを逐次決定理論の問題として，未知関数 $f$ の情報量を考慮しながら $f$ を最も早く最大化するために，次にどこを評価すればよいかを考える (文献65,66)。
例えば 3 点で評価し，その点での関数の対応する値 [(x1, f(x1)), (x2, f(x2)), (x3, f(x3))] を測定した後，アルゴリズムはどの点 $x$ を次に評価すべきか，また，最大値はどこだと考えているか？
これは典型的な機械知能問題であり，薬物設計 (薬物の効能を関数とする) からロボット工学 (ロボットの歩行速度を関数とする) まで，科学や工学の幅広い分野で応用されている。
高価な関数の最適化を伴うあらゆる問題に適用できる。
「高価な」という修飾語は，ベイズ最適化が次に評価する場所を決定するためにかなりの計算資源を使う可能性があり，これらの資源と関数評価のコストとのトレードオフが必要であることに由来している。
<!-- Bayesian optimization poses this as a problem in sequential decision theory: where should one evaluate next so as most quickly to maximize f, taking into account the gain in information about the unknown function f (refs 65,66)?
For example, having evaluated at three points, measuring the corresponding values of the function at those points, [(x1, f(x1)), (x2,f(x2)),(x3,f(x3))], which point x should the algorithm evaluate next, and where does it believe the maximum to be?
This is a classic machine-intelligence problem with a wide range of applications in science and engineering, from drug design (where the function could be the drug’s efficacy) to robotics (where the function could be the speed of a robot’s gait).
It can be applied to any problem involving the optimization of expensive functions; the qualifier ‘expensive’ comes because Bayesian optimization might use substantial computational resources to decide where to evaluate next, and a trade-off for these resources has to be made with the cost of function evaluations.-->

現在，最も優れた大域的最適化手法は，最適化される不確実な関数 $f$ の確率分布のベイズ表現を維持し，この不確実性を利用して (Xの) どこに次の問い合わせを行うかを決定する(67-69)。
連続空間では，ほとんどのベイズ型最適化手法 (図3) は (ノンパラメトリック節で述べた) ガウス過程を用いて未知関数をモデル化する。
最近のインパクトのある応用例としては，ディープニューラルネットワークを含む機械学習モデルの学習過程の最適化である (70)。
これと関連する最近の研究 (71) は，機械知能を向上させるための機械知能の応用のさらなる例である。
<!-- The current best-performing global optimization methods maintain a Bayesian representation of the probability distribution over the uncertain function f being optimized, and use this uncertainty to decide where (in X) to query next(67–69).
In continuous spaces, most Bayesian optimization methods (Fig.3) use Gaussian processes (as described in the section on non-parametrics) to model the unknown function.
A recent highimpact application has been the optimization of the training process for machine-learning models, including deep neural networks(70).
This and related recent work(71) are further examples of the application of machine intelligence to improve machine intelligence.-->

<div class="figcenter">
<img src="../figures/2015Ghahramani_fig3.svg" width="66%"><br/>
<div class="figcaption">
<!-- <div style="text-align: left; width:88%;background-color: cornsilk"> -->

**図 3. ベイズ最適化**<br/>
一次元のベイズ最適化の簡単な図。目標はある真の未知関数 $f$ (図示せず) を最大化することである。 この関数に関する情報は，特定の $x$ 値における関数の評価である観測 (丸，上) を行うことによって得られる。
これらの観測は，可能な関数の分布を表す関数値 (平均，青線，および標準偏差，青斜線領域として示される) に対する事後分布を推論するために使用される; 不確実性は観測から離れるにつれて大きくなることに注意。 この関数上の分布に基づき，獲得関数が計算される (緑色の網掛け部分，下図)。 これは，異なる $x$ 値で未知の関数 $f$ を評価することによる利得を表す。 獲得関数は `期待される改善` や `情報利得` のような異なる獲得関数を使用することができる。 獲得関数のピーク (赤線) は，評価するために最適な次のポイントであり，したがって評価のために選択される (赤点，新しい観測)。 左右図は，それぞれ 3 回と 4 回の関数評価後に起こりうることの例である。
<!-- Figure 3. Bayesian optimization.
A simple illustration of Bayesian optimization in one dimension. The goal is to maximize some true unknown function f (not shown). Information about this function is gained by making observations (circles, top), which are evaluations of the function at specific x values. These observations are used to infer a posterior distribution over the function values (shown as mean, blue line; and standard deviations, blue shaded area) representing the distribution of possible functions; note that uncertainty grows away from the observations. On the basis of this distribution over functions, an acquisition function is computed (green shaded area, bottom panels), which represents the gain from evaluating the unknown function f at different x values; note that the acquisition function is high where the posterior over f has both high mean and large uncertainty. Different acquisition functions can be used such as `expected improvement` or `information gain`. The peak of the acquisition function (red line) is the best next point to evaluate, and is therefore chosen for evaluation (red dot, new observation). The left and right panels show an example of what could happen after three and four function evaluations, respectively. -->
</div>
</div>

ベイズ最適化と強化学習の間には，興味深いつながりがある。具体的には，ベイズ最適化は，決定 (評価する $x$ の選択) がシステムの状態 (実際の関数 $f$) に影響を与えない逐次決定問題である。このような状態に依存しない逐次決定問題は，強化学習問題の下位クラスであるマルチアームバンディット (72) に分類される。より広く，最近の重要な研究は，不確実な系を制御するための学習に対してベイズ的アプローチをとっている(73) (レビューは文献 74 参照)。行動の将来の結果に関する不確実性を忠実に表現することは，意思決定や制御問題において特に重要である。良い決定は，異なる結果の確率とその相対的なペイオフの良い表現に依存している。
<!-- There are interesting links between Bayesian optimization and reinforcement learning. Specifically, Bayesian optimization is a sequential decision problem where the decisions (choices of x to evaluate) do not affect the state of the system (the actual function f). Such state-less sequential decision problems fall under the rubric of multi-arm bandits(72), a subclass of reinforcement-learning problems. More broadly, important recent work takes a Bayesian approach to learning to control uncertain systems(73) (for a review see ref.74). Faithfully representing uncertainty about the future outcome of actions is particularly important in decision and control problems. Good decisions rely on good representations of the probability of different outcomes and their relative payoffs. -->

より一般的には，ベイズ最適化はベイズ数値計算 (75,76) の特殊なケースであり，非常に活発な研究分野として再浮上している (http://www.probabilistic-numerics.org)。 常微分方程式を解くことや数値積分などの話題が含まれる。 これらすべての場合において，確率論は計算の不確実性，すなわち決定論的な計算の結果について持つ不確実性を表現するために用いられている。
<!-- More generally, Bayesian optimization is a special case of Bayesian numerical computation(75,76), which is re-emerging as a very active area of research (http://www.probabilistic-numerics.org), and includes topics such as solving ordinary differential equations and numerical integration.
In all these cases, probability theory is being used to represent computational uncertainty; that is, the uncertainty that one has about the outcome of a deterministic computation. -->

<div class="figcenter">
<div class="figcaption">

図 8: 成分 4 の点ごとの事後分布 (左) と、データを含む成分の累積和の事後分布 (右)<!--Figure 8: Pointwise posterior of component 4 (left) and the posterior of the cumulative sum of components with data (right) -->
この成分は周期が約 10.7 年の周期成分である。周期を跨いで、この関数の形状は典型的な長さスケール 34.4 年で滑らかに変化する。各周期内のこの関数の形状は非常に滑らかで、正弦波に似ている。この成分は 1643 年までと 1716 年以降に適用される。この成分は残差分散の 72.4 % を説明する。これにより説明される総分散は 73.0 %から 92.5 %に増加する。この成分の追加により、交差検証誤差平均絶対値 (MAE) は 0.18 から 0.15 へ 16.67 %減少する。
<!-- This component is approximately periodic with a period of 10.7 years. Across periods the shape of this function varies smoothly with a typical length scale of 34.4 years. The shape of this function within each period is very smooth and resembles a sinusoid. This component applies until 1643 and from 1716 onwards. This component explains 72.4% of the residual variance; this increases the total variance explained from 73.0% to 92.5%. The addition of this component reduces the cross validated MAE by 16.67% from 0.18 to 0.15.-->
</div>
<img src="../figures/2015Ghahramani_fig4.svg" width="88%"><br/>
<div class="figcaption">
<!-- <div style="text-align: left;width: 88%;background-color: cornsilk"> -->

図 4. 自動統計家。
自動統計家 Automatic Statistician を説明する流れ図。
系への入力はデータであり，この場合は時系列で表現される (左上)。
モデルはベイズ推論により得点化され (箱 1)，データの良い解釈を発見するためにモデルの文法が検索される (左下)。
発見されたモデルの構成要素は，英語のフレーズに翻訳される (右下)。
最終的には，テキスト，図，表で構成される報告書が作成され，データについて推測されたことが詳細に記述され，モデルのチェックと批判に関する節が含まれる (右上)(文献 99,100)。
データおよび報告書は説明のためのもの。
<!-- Figure 4. The Automatic Statistician.
A flow diagram describing the Automatic Statistician.
The input to the system is data, in this case represented as time series (top left).
The system searches over a grammar of models to discover a good interpretation of the data (bottom left), using Bayesian inference to score the models (Box 1).
Components of the model discovered are translated into English phrases (bottom right).
The end result is a report with text, figures and tables, describing in detail what has been inferred about the data, including a section on model checking and criticism (top right)99,100.
Data and the report are for illustrative purposes only. -->

</div>
</div>

# 5. データ圧縮
<!-- # 5. Data compression-->

圧縮されたデータから元のデータを正確に復元できるように，データをできるだけ少ないビット数で通信または保存するように，データを圧縮する問題を考える。
このような可逆的なデータ圧縮の方法は，コンピュータのハードディスクからインターネット上のデータ転送まで，情報技術においていたるところに存在する。
データ圧縮と確率的モデリングは表裏一体であり，ベイズ型機械学習法は圧縮の最先端をどんどん進めている。
圧縮と確率的モデリングの関係は，数学者 Claude Shannon のソースコード定理(77) で確立された。
この定理は，データを可逆圧縮するのに必要なビット数は，データの確率分布のエントロピーによって制限されるとするものである。
一般に使用されているすべての可逆データ圧縮アルゴリズム (例えば gzip) は，記号列の確率モデルとして見ることができる。
<!-- Consider the problem of compressing data so as to communicate them or store them in as few bits as possible in such a manner that the original data can be recovered exactly from the compressed data.
Methods for such lossless data compression are ubiquitous in information technology, from computer hard drives to data transfer over the internet.
Data compression and probabilistic modelling are two sides of the same coin, and Bayesian machine-learning methods are increasingly advancing the state-of-the-art in compression.
The connection between compression and probabilistic modelling was established in the mathematician Claude Shannon’s seminal work on the source coding theorem(77), which states that the number of bits required to compress data in a lossless manner is bounded by the entropy of the probability distribution of the data.
All commonly used lossless data compression algorithms (for example, gzip) can be viewed as probabilistic models of sequences of symbols.-->

ベイズ機械学習との関連は，学習する確率モデルが優れていればいるほど，圧縮率を高くすることができるということである(78)。
異なる種類の系列 (例えばシェークスピアの劇やコンピュータのソースコード) は非常に異なる統計パターンを持っているので，これらのモデルは柔軟で適応的である必要がある。
世界最高の圧縮アルゴリズム (例えば Sequence Memoizer(79) や動的パラメータ更新によるPPM(80)) は，系列のベイズ型ノンパラメトリックモデルと同等であることが判明し，系列の統計構造を学習する方法を理解することで圧縮の改善が進んでいる。
今後の圧縮の進歩は，画像やグラフなどの構造化された非配列データに対する特殊な圧縮方法など，確率的機械学習の進歩に伴ってもたらされるであろう。
<!-- The link to Bayesian machine learning is that the better the probabilistic model one learns, the higher the compression rate can be(78).
These models need to be flexible and adaptive, since different kinds of sequences have very different statistical patterns (say, Shakespeare’s plays or computer source code).
It turns out that some of the world’s best compression algorithms (for example, Sequence Memoizer(7)9 and PPM with dynamic parameter updates(80)) are equivalent to Bayesian non-parametric models of sequences, and improvements to compression are being made through a better understanding of how to learn the statistical structure of sequences.
Future advances in compression will come with advances in probabilistic machine learning, including special compression methods for non-sequence data such as images, graphs and other structured objects.-->

# 6. データから解釈可能なモデルを自動発見する
<!-- # 6. Automatic discovery of interpretable models from data-->

機械学習の大きな挑戦の一つは，データから統計モデルを学習し説明するプロセスを完全に自動化することである。
これは 自動統計家 Automatic Statistician (http://www.automaticstatistician.com) の目標である。
この系は，データからもっともらしいモデルを自動的に発見し，発見したものを平易な英語で説明することができる(81)。
これは，データから知識を抽出することに依存しているほとんどすべての分野の努力に役立つ可能性がある。
機械学習の文献では，カーネル法，ランダムフォレスト，深層学習などの手法を使ってパターン認識問題の性能をどんどん向上させることに焦点が当てられているが，それと対照的に，自動統計学家は解釈可能な要素からなるモデルを構築し，データからモデル構造に関する不確実性を表現する原理的方法を備えている。
また，大きなデータセットだけでなく，小さなデータセットに対しても，合理的な答えを出すことができる。
ベイズアプローチはモデルの複雑さとデータの複雑さをトレードオフするエレガントな方法を提供し，確率モデルは既に述べたように構成的で解釈可能である。
<!-- One of the grand challenges of machine learning is to fully automate the process of learning and explaining statistical models from data.
This is the goal of the Automatic Statistician (http://www.automaticstatistician.com), a system that can automatically discover plausible models from data, and explain what it has discovered in plain English(81).
This could be useful to almost any field of endeavour that is reliant on extracting knowledge from data.
In contrast to the methods described in much of the machinelearning literature, which have been focused on extracting increasing performance improvements on pattern-recognition problems using techniques such as kernel methods, random forests or deep learning, the Automatic Statistician builds models that are composed of interpretable components, and has a principled way of representing uncertainty about model structures given the data.
It also gives reasonable answers not just for big data sets, but also for small ones.
Bayesian approaches provide an elegant way of trading off the complexity of the model and the complexity of the data, and probabilistic models are compositional and interpretable, as already described. -->

自動統計学者のプロトタイプ版は，時系列データを取り込み，発見したモデルを説明する 5～15 ページのレポートを自動生成する (図 4)。
この系は，確率的なビルディングブロックを文法を通して組み合わせることで，モデルのオープンエンドな言語を構築することができるという考えに基づいている (82)。
方程式学習 (例えば文献 83 参照) とは対照的に，モデルは正確な方程式ではなく，関数の一般的な特性 (例えば滑らかさ，周期性，傾向など) を捉えようとするものである。
不確実性の取り扱いは 自動統計家 Automatic Statistician の中核をなすもので，ベイズ型ノンパラメトリックを利用して，最先端の予測性能を得るための柔軟性を与え，モデルの空間を探索するために限界尤度 (囲み記事 1) という指標を利用する。
<!-- A prototype version of the Automatic Statistician takes in time-series data and automatically generates 5–15 page reports describing the model it has discovered (Fig. 4).
This system is based on the idea that probabilistic building blocks can be combined through a grammar to build an open-ended language of models(82).
In contrast to work on equation learning (see for example ref. 83), the models attempt to capture general properties of functions (for example, smoothness, periodicity or trends) rather than a precise equation.
Handling uncertainty is at the core of the Automatic Statistician; it makes use of Bayesian non-parametrics to give it the flexibility to obtain state-of-the-art predictive performance, and uses the metric marginal likelihood (Box 1) to search the space of models.-->

重要な先行研究としては、統計的エキスパートシステム(84,85)や、機械学習と科学的発見を微生物学の実験プラットフォームと閉ループで統合し、新しい実験の設計と実行を自動化するロボットサイエンティスト(86)がある。
また Auto-WEKA は，既に述べたベイズ最適化技術を多用して，分類器の学習を自動化する最近のプロジェクトである (71)。
データへの機械学習手法の適用を自動化する取り組みは最近活発になっており，最終的にはデータサイエンスのための人工知能系が実現されるかもしれない。
<!--Important earlier work includes statistical expert systems(84,85) and the Robot Scientist, which integrates machine learning and scientific discovery in a closed loop with an experimental platform in microbiology to automate the design and execution of new experiments86.
Auto-WEKA is a recent project that automates learning classifiers, making heavy use of the Bayesian optimization techniques already described(71).
Efforts to automate the application of machine-learning methods to data have recently gained momentum, and may ultimately result in artificial intelligence systems for data science. -->


# 7. 展望<!-- # 7. Perspective-->

情報革命により，これまで以上に大規模なデータの集合が利用できるようになった。
このようなビッグデータのモデリングにおいて，不確実性はどのような役割を果たすのだろうか。
古典的な統計学の結果によれば，ある規則性の条件の下で，大規模データセットの極限において，ベイズ・パラメトリック・モデルのパラメータの事後分布は，最尤推定値の周りの 1 点に収束することが示されている。
これは，ベイズの確率論的な不確実性モデリングは，多くのデータを持っていれば不要ということだろうか？
<!-- The information revolution has resulted in the availability of ever larger collections of data.
What is the role of uncertainty in modelling such big data? Classic statistical results state that under certain regularity conditions, in the limit of large data sets the posterior distribution of the parameters for Bayesian parametric models converges to a single point around the maximum likelihood estimate.
Does this mean that Bayesian probabilistic modelling of uncertainty is unnecessary if you have a lot of data? -->

そうでない理由は少なくとも二つある(87)。
第一に，これまで見てきたように，ベイズ型ノンパラメトリックモデルは基本的に無限に多くのパラメータを持っているので，どんなに多くのデータを持っていても，学習能力は飽和せず，その予測は向上し続けるはずである。
<!-- There are at least two reasons this is not the case(87).
First, as we have seen, Bayesian non-parametric models have essentially infinitely many parameters, so no matter how many data one has, their capacity to learn should not saturate, and their predictions should continue to improve.-->

第二に，多くの大規模データセットは，実は小規模データセットの大規模な集合体である。
例えば，個別化医療やレコメンデーション系などの分野では，データ量は多くても，患者や顧客それぞれのデータ量は比較的少ないと考えられる。
各人の予測をカスタマイズするためには，各人に固有の不確実性を含んだモデルを構築し，そのモデルを階層的に結合して，他の類似した人から情報を借用できるようにする必要がある。
我々はこれをモデルのパーソナライゼーションと呼び，階層的ディリクレ過程 (36) やベイズマルチタスク学習 (88,89) などの階層的ベイズアプローチを使って自然に実装されている。
<!-- Second, many large data sets are in fact large collections of small data sets.
For example, in areas such as personalized medicine and recommendation systems, there might be a large amount of data, but there is still a relatively small amount of data for each patient or client, respectively.
To customize predictions for each person it becomes necessary to build a model for each person — with its inherent uncertainties — and to couple these models together in a hierarchy so that information can be borrowed from other similar people.
We call this the personalization of models, and it is naturally implemented using hierarchical Bayesian approaches such as hierarchical Dirichlet processes36, and Bayesian multitask learning(88,89). -->

機械学習と知能に対する確率的アプローチは，従来のパターン認識問題にとどまらない幅広い影響を持つ，非常に活発な研究分野である。
これまで概説したように，これらの問題には，データ圧縮，最適化，意思決定，科学的モデルの発見と解釈，パーソナライゼーションなどが含まれる。
確率的アプローチが重要な問題と，非確率的な機械学習アプローチで解決できる問題の重要な違いは，不確実性が中心的な役割を担っているかどうかという点である。
さらに，従来の最適化ベースの機械学習アプローチには，より原理的に不確実性を扱う確率論的な類似がある。
例えば，ベイズニューラルネットワークはニューラルネットワークのパラメータの不確実性を表現し (44)，混合モデルはクラスタリング手法の確率的相同物である(78)。
確率的機械学習では，原理的にどのように問題を解くかを定義することが多いが，この分野の中心的な課題は，実際に計算効率の良い方法で解く方法を見つけることである (90,91)。
計算困難な推論問題を効率的に近似するためのアプローチは数多く存在する。
最近の推論手法では，数百万のデータ点への拡張が可能となり，確率的手法は従来の手法と計算上競争力を持つようになった (92-95)。
最終的に，知能は不完全に感知された不確実な世界を理解し，行動することに依存する。
確率的モデリングは，これまで以上に強力な機械学習や人工知能システムの開発において，今後も中心的な役割を担っていくことだろう。
<!-- Probabilistic approaches to machine learning and intelligence are a very active area of research with wide-ranging impact beyond conventional pattern-recognition problems.
As I have outlined, these problems include data compression, optimization, decision making, scientific model discovery and interpretation, and personalization.
The key distinction between problems in which a probabilistic approach is important and problems that can be solved using non-probabilistic machine-learning approaches is whether uncertainty has a central role.
Moreover, most conventional optimization-based machine-learning approaches have probabilistic analogues that handle uncertainty in a more principled manner.
For example, Bayesian neural networks represent the parameter uncertainty in neural networks(44), and mixture models are a probabilistic analogue for clustering methods(78).
Although probabilistic machine learning often defines how to solve a problem in principle, the central challenge in the field is finding how to do so in practice in a computationally efficient manner(90,91).
There are many approaches to the efficient approximation of computationally hard inference problems.
Modern inference methods have made it possible to scale to millions of data points, making probabilistic methods computationally competitive with conventional methods(92–95).
Ultimately, intelligence relies on understanding and acting in an imperfectly sensed and uncertain world.
Probabilistic modelling will continue to play a central part in the development of ever more powerful machine learning and artificial intelligence systems.  -->
