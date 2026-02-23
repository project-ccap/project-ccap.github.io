---
title: Bayes for CCAP
author: Shin Asakawa
layout: home
---
<link href="/asamarkdown.css" rel="stylesheet"><!--</link>-->

$$
\newcommand{\bs}[1]{\boldsymbol{#1}}
\newcommand{\mb}[1]{\boldsymbol{#1}}
% \newcommand{\mb}[1]{\mathbf{#1}}
\newcommand{\Brc}[1]{\left(#1\right)}
\newcommand{\BRc}[1]{\left[#1\right]}
\newcommand{\Rank}{\text{rank}\;}
\newcommand{\Hat}[1]{\widehat{#1}}
\newcommand{\Prj}[1]{\mb{#1}\Brc{\mb{#1}^{\top}\mb{#1}}^{-1}\mb{#1}^{\top}}
\newcommand{\RegP}[2]{\Brc{\mb{#1}^{\top}\mb{#1}}^{-1}\mb{#1}^{\top}\mb{#2}}
\newcommand{\NSQ}[1]{\left|\mb{#1}\right|^2}
\newcommand{\Norm}[1]{\left|#1\right|}
\newcommand{\IP}[2]{\left({#1}\cdot{#2}\right)}
\newcommand{\Bar}[1]{\overline{\;#1\;}}
\newcommand{\of}[1]{\left(#1\right)}
\newcommand{\Of}[1]{\left[#1\right]}
\newcommand{\OF}[1]{\left\{#1\right\}}
\newcommand{\widebar}[1]{\overline{#1}}
$$

## 言語材料を固定効果として扱う誤謬 (Language as fixed effect fallacy, Clark, 1973)<br/>CCAP の参加者の諸兄姉が Bayes を知りたいと思う理由について浅川の邪推

そもそも，というか歴史的な文脈としては，言語材料を固定効果として扱う誤謬 (Language as fixed effect fallacy, Clark, 1973) が出発点なのだろう。
Clark (1973) は、心理学研究において、言語材料を固定効果として扱うことの誤謬を指摘し、これが統計的推論の根本的な問題であると主張した。彼は、言語材料が無数に存在する中から「たまたま」選ばれた一部であることを考慮せずに分析を行うことが、研究結果の一般化可能性を損なうと警告した。

このように Clark は (札幌農学校で，「青年よ大志を抱け」と言っただけでなく ^^;)，言語材料を安易に固定効果として扱った分析の危うさを指摘した。

<!--
“Boys, be ambitious! Be ambitious not for money or for selfish aggrandizement, not for that evanescent thing which men call fame. Be ambitious for the attainment of all that a man ought to be.”

この言葉がこのように広まったのは，昭和 39.3.16 の朝日新聞「天声人語」欄によるものと思われる。「天声人語」はその出典として稲富栄次郎著「明治初期教育思想の研究」(昭 19) をあげ，さらに次のような訳文を添えている。「青年よ大志をもて。それは金銭や我欲のためにではなく，また人呼んで名声という空しいもののためであってはならない。人間として当然そなえていなければならぬあらゆることを成しとげるために大志をもて」
出典: https://www.lib.hokudai.ac.jp/collections/clark/boys-be-ambitious/
-->

Clark (1973) の指摘した Language as fixed effect fallacy は、被験者と刺激の両方をランダム効果として扱う **線形混合効果モデル** (LME: Linear Mixed Effect Modeling, 文献によっては LMEM とも表記される) の発展につながった。LME は、被験者と言語材料の両方のバラツキを考慮することで、より一般化可能な結論を導くことができるようになった。
線形混合効果モデルは、被験者と刺激の両方をランダム効果として扱う。Clark (1973) の指摘した language-as-fixed-effect fallacy を回避する方法としては，最も知られた方法であろう。
CCAP メンバとしては，玉岡 (2022)，あるいは，橋本，上間，三盃 (2022) の文献の親密度が高いだろう。

近年の発展としては，Yarkoni (2022) の Generalizability Crisis が挙げられる。Yarkoni (2022) は Clark (1973) の指摘した問題をさらに拡大して、心理学研究全体の一般化可能性の危機を論じている。Yarkoni は、研究者が特定のサンプルや条件に過度に依存する傾向があることを指摘し、これが研究結果の再現性や一般化可能性を損なう原因となっていると主張している。

それでも，なお，ベイズを知りたいという動機は，LME では限界があるということを感じているからではないかと邪推してみた。たとえば、LME では、モデルの構造をあらかじめ決めておく必要がある（例：ランダム効果の構造）。しかし、ベイズ統計学では、モデルの構造をデータから学習することができる。また、ベイズ統計学では、モデルの不確実性を自然に扱うことができる。これらの理由から、CCAP の参加者の皆様は、ベイズ統計学に興味を持っているのではないかと邪推する。

* [Language as fixed effect fallacy](1973Clark_ja/){:target="_blank"}: Clark, H. H. (1973) The language-as-fixed-effect fallacy: A critique of language statistics in psychological research. Journal of Verbal Learning and Verbal Behavior, 12(4), 335–359.
* [Generalizability Crisis: Yarkoni (2022)](2022Yarkoni_ja/){:target="_blank"} The Generalizability Crisis, Behavioral and Brain Sciences, 45, e1, 1–37, DOI:10.1017/S0140525X2000168X
* 玉岡 (2022) チュートリアル：線形混合効果モデル（LME）による分析法,
* 橋本，上間，三盃 (2022) 線形混合効果モデリングによる解析例 － 成人・子どもを対象にした読み処理に関する研究から －
* [lme4 パッケージのドキュメント](https://cran.r-project.org/web/packages/lme4/lme4.pdf)


<!-- そこで，以下に，一般の線形回帰モデルと線形混合効果モデルの違いを、数式で示してみる -->

<!-- ## 回帰モデルの違い，背景にある異なる考え方

$$\begin{aligned}
y &= Xw + \epsilon, & w\sim\mathcal{N}\of{\mu_{w},\sigma^2}, \epsilon\sim\mathcal{N}\of{0,\sigma^2}     & \hspace{20mm}\text{(頻度論的線形回帰モデル) $H_0:\mu_w=0$} \\
y &= Xw + \epsilon, & X\sim\mathcal{N}\of{\mu_{x},\sigma^2_{s}}, w\sim\mathcal{N}\of{\mu_{w},\sigma^2}, \epsilon\sim\mathcal{N}\of{0,\sigma^2}, & \hspace{20mm}\text{(線形混合効果モデル) かつ (ベイズ回帰モデル)} \\
y &= f\of{X} + \epsilon, & f\sim\mathcal{GP}\of{0,k}, \epsilon\sim\mathcal{N}\of{0,\sigma^2}, & \hspace{20mm}\text{(ガウス過程回帰モデル)}
\end{aligned}$$

今回の依頼が，上式 1 行目の頻度論的線形回帰モデルと上式 2 行目の線形混合効果モデルやベイズ回帰モデルの相違を説明せよということなのか，それとも，上式 3 行目のガウス過程回帰モデルの説明まで含むのかはわからない。とりあえず上式 1 行目の頻度論的線形回帰モデルの説明から始めてみることとしたい。 -->


## ベイズの定理

ちなみになぜベイズの定理が重要かというと、ベイズ統計学の基礎となるからです。ベイズの定理は、ある事象 A が起こる確率を、別の事象 B が起こる確率を用いて計算する方法を提供します。これにより、データから未知のパラメータや仮説の確率を更新することができる。
日本語版ウィキペディアでは，次式がベイズの定理として紹介されていまる：

$$
P\of{A\vert B} = \frac{P\of{B\vert A}P\of{A}}{P\of{B}}
$$

上式は，次のように発音する：
ピー オブ エー ギブン ビー
イコール
ピー オブ ビー ギブン エー タイムズ ピー オブ エー
オーバー
ピー オブ ビー

上式は，次のように考えた方が後々の考え方に馴染むかもしれない：

$\displaystyle P\of{A\vert B} = \frac{P\of{B\vert A}P\of{A}}{P\of{B}} = \frac{P\of{B\vert A}P\of{A}}{P\of{B\vert A}P\of{A}+P\of{B\vert\neg A}P\of{\neg A}}$, 
すなわち，上式最右辺の分母は，$p\of{A} + p\of{\neg A} = 1$ を利用して，$P\of{B} = P\of{B\vert A}P\of{A}+P\of{B\vert\neg A}P\of{\neg A}$ と書き換えられることを示している。
このようにしておけば，事象 $A$ とその補事象 $\neg A$ の両方だけでなく，取りうる状態が $n$ であっても，$p\of{A_i}, i\in\{1, \dots, n\}$ を考慮して，事象 $B$ が起こる確率を計算することができる。

---
See also:

* [初めてのベイズ学習](2023_1123var_basic/){:target="_blank"}
* [確率的機械学習と人工知能, Gharamani+(2015)](2015Gharamani_BayesianML_ja/){:target="_blank"}
* [ベイズ推定のための変分近似法: EM アルゴリズム以降の生活, Tzikas+(2008)](2008Tzikas_Variational_Bayes_ja/){:target="_blank"}
<!-- * [ベイズ統計学の基礎, Gelman+(2013)](2013Gelman_Bayesian_Statistics_ja.md) -->


<!-- ## 研究史略と浅川の雑感 -->

## Language as fixed effect はなぜ問題なのか？

**たまたま選んだその単語リストでうまくいったからといって、すべての単語で同じことが言えると思うな** という警告。
<!-- この language-as-fixed-effect fallacy（言語固定効果の誤謬）について、3 つのポイントで分かりやすく解説します。 -->

### 1. なぜ「固定（Fixed）効果」だとダメなのか？

統計学において、要因の扱いには大きく分けて 2 種類:

* **固定効果 (Fixed Effect):** その実験で選んだ項目そのものに興味がある場合。（例：男性と女性の比較、投与量 0 mgと10 mg の比較）
* **変量効果 (Random Effect):** 無数にある候補の中から「たまたま」抽出された一部にすぎない場合。（例：実験に参加したAさん、Bさん……といった「個人」）

従来の心理言語学では、**参加者（人間）** については「人によってバラツキがある」として変量効果（サンプリングの不確実性）を考慮していたが、**刺激（単語）** については「固定されたもの」として扱っていた。

しかし、本来「単語」も辞書にある膨大な数の中から抽出されたサンプルに過ぎない。特定の単語だけで見られた傾向を、言語全体に広げてしまうのは、**10 人の日本人の意見を聞いて、日本国民全員がそう思っていると結論づける** のと同じミスにあたる。

### 2. クラーク（1973）の批判の本質

1973年 に Clark がこの問題を指摘するまで、多くの研究者が誤った分散分析を行っていた。
Clark は、結果が一般化可能であると言うためには、以下の **両方のバラツキ** を同時に乗り越えなければならないと主張した。

1. **人によるバラツキ**（たまたまこの被験者群だったからではないか？）
2. **単語群によるバラツキ**（たまたまこの単語リストだったからではないか？）

これを解決するために、彼は $F^\prime$ という統計量を導入し、人と言葉の両方の不確実性を合算して計算することを提唱した。

### 3. 現代における解決策

現在、この問題はさらに進化した **線形混合モデル LMM:Linear Mixed Effects Models** によって処理。
論文中に Random intercepts for Subjects and Items（被験者と言葉の両方にランダム切片を設定した）という記述があれば、それはまさにこの「言語固定効果の誤謬」を回避するための処理を行っていることを意味する。

#### 3.1 語彙研究への適用（2020年代）

最近の語彙研究では、線形混合効果モデルが理想的だとされながらも、固定効果のみのモデルに依存する研究者が多いことが問題視されている。
[Nicklin & Vitta (2025)](https://onlinelibrary.wiley.com/doi/10.1111/lang.12715) の研究では、固定効果のみの分析が真の効果量を過大評価する可能性を示唆。

#### 3.2 神経科学への展開

EEG や MEG 研究にも混合効果モデルを適用する [lmeEEG DOI:10.1016/j.jneumeth.2023.109991](https://www.biorxiv.org/content/10.1101/2023.01.18.524560v4){:target="_blank"} などの新しい手法が開発されており、心理言語学以外の分野にも広がっている。

<!-- 実践的な課題 -->
混合効果モデルの複雑さが障壁となっており、データシミュレーションを通じた理解促進やチュートリアルが[公開](https://eprints.gla.ac.uk/223488/1/223488.pdf) されている。
<!-- また、モデルの収束問題やオプティマイザの設定など、実装上の課題も議論されている。 -->
<!-- Clark (1973)の論文は 50 年以上経った現在も、心理学研究における統計的推論の根本的な問題を提起し続けており、再現性危機や一般化可能性の問題と密接に関連する重要な文献として再評価されています。 -->

### 4. LME は福音なのか？それとも徒花（あだばな）なのか

* でも LME って，結局，頻度論統計学の枠組みの中での解決策の提案でしかないよね？
* LME だって正しいとは限らないのでは？
* 知りたいことは，有意差ではなくて，言語活動を行っている人間の頭の中では何が起こっているのか，であろう。
* [Spieler & Balota (1997) 基準](1997Spieler_Balota_ja/){:target="_blank"} を考えてモデル化


## [Kruschke(2013) Bayesian Estimation Supersedes the t Test 要約](2013Kruschke_BEST_ja/){:target="_blank"}

## ベイズモデルに関するレクチャの提案
<!-- # ベイズ統計学とベイズ機械学習の相違-->

1. 推測統計学における帰無仮説検定 (NHST: Null Hypothesis Statistical Tests) と ベイズ統計学におけるベイズ因子
2. 回帰分析の違い
    1. 推測統計学における回帰分析
    2. ベイズ回帰分析
    3. ガウス過程
3. パス解析 (構造方程式モデル) とグラフィカルモデルとの相違

実際 [Gelman の教科書](https://www.amazon.co.jp/dp/4627097034){:target="_blank"} では、21 章に GP モデルの記述が認められる。
ガウス過程まで含めて考えることによって，Clark (1973) の指摘した language-as-fixed-effect fallacy の抜本的解決策とみなせることになる。

逆に言えば，線形混合効果モデルの枠組みは，いわゆる頻度論的統計学による拡張であり，本質的な解決策になっていないとも言えるからである。
とは言え，固定量効果を用いた分散分析では，いわゆる p 値のインフレが起こる可能性があり，このままでは行き詰まるという危機感をお持ちの諸兄姉も多いのではないかと邪推する。
