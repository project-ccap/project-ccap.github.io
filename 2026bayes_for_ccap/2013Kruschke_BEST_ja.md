---
layout: home
---
<link href="/asamarkdown.css" rel="stylesheet">
<!-- <div style="background-color:#F8F8F8;"> -->

## 論文: Bayesian Estimation Supersedes the t Test, Kruschke(2013) 要約

<!-- <div class="abstract" style="background-color:white;"> -->
<!-- <div class="abstract" style="background-color:#FFFFCC;"> -->

* **原題:** Bayesian Estimation Supersedes the *t* Test<br/>
* **著者:** John K. Kruschke (Indiana University)<br/>
* **DOI:** 10.1037/a0029146

### 1. はじめに

従来の「t 検定（NHST: Null Hypothesis Significance Testing）」は心理学や他の科学分野で広く使われているが、深刻な欠陥を抱えている。
本稿は t 検定の代替として、より情報量が多く堅牢な **「ロバスト・ベイズ推定（Robust Bayesian Estimation）」** を提案。この手法を **BEST (Bayesian Estimation Supersedes the t-test)** と呼び、その利点を論じる。

### 2. 従来手法 t 検定（NHST）の問題点

* **p 値は「サンプリングの意図」に依存:** p 値は、観測されたデータだけでなく、「観測されなかったが起こり得たデータ」にも依存。研究者がいつデータ収集を止めるつもりだったか（固定のサンプルサイズか、一定の期間か、あるいは有意になるまでか）によって、同じデータであっても p 値は変化してしまう。
* **信頼区間（CI）の情報不足:** 95 % 信頼区間は、パラメータの確率分布を示すものではない。区間の中央が端よりも確からしいといった情報は含まれておらず、単に「棄却されない値の範囲」に過ぎない。
* **帰無仮説を受容できない:** t 検定では帰無仮説を棄却することはできても、証拠として「差がない」ことを積極的に支持（受容）する仕組みがありません（信頼区間がゼロを含んでいても、それは単に「差があると言えない」だけ）。
* **検出力（Power）の不確実性:** NHST の事後的な検出力分析は、点推定値に基づくため非常に不確実性が高く、実質的に役に立たないことが多い。

### 3. ロバスト・ベイズ推定（BEST）のアプローチ

* **記述モデル:** 群ごとの平均値（$\mu$）と標準偏差（$\sigma$）、および外れ値に対応するための「正規性パラメータ（$\nu$）」を持つ t 分布を用いてデータを記述。t 分布を使うことで、外れ値の影響を受けにくいロバストな推定が可能になる。
* **事前分布:** 恣意性を避けるため、データに対して影響力の少ない、幅の広い（無情報の）事前分布を設定。
* **事後分布:** ベイズ推論により、パラメータの信用できる値の完全な分布（事後分布）が得られる。これにより、平均値の差、標準偏差の差、効果量などの確率分布を直接評価できる。


図 2 は記述モデルとそのパラメータの事前分布を示したものである。群 $j$ からの $i$ 番目のデータは，図の下部で $y_{ji}$ と示されている。データは t 分布で記述され，図の中央に描かれている。事前分布は図の上部に示されている。特に，平均パラメータ $\mu_1$ と $\mu_2$ に関する事前分布は，非常に広い正規分布であると仮定され，図では象徴的な正規分布の形で描かれている。データの任意のスケールに対して事前分布を広く保つために，$\mu$ に関する事前分布の標準偏差 $S$ を，プールされたデータの標準偏差の 1,000 倍に設定した。$\mu$ に関する事前分布の平均 $M$ は，プールされたデータの平均に任意に設定される。この設定は，単にデータの任意のスケールに対して事前分布を適切にスケーリングしておくために行われる。したがって、もしyが距離の尺度であれば、スケールはナノメートルでも光年でもよく、事前分布は同じように非妥協的である。標準偏差パラメータの事前分布も非妥協的であると仮定され、プールされたデータの標準偏差の 1,000 分の 1 に設定された低い値Lから、プールされたデータの標準偏差の 1,000 倍に設定された高い値 H までの一様分布として表現される。最後に、$\nu$ パラメータは指数分布の事前分布を持ち，これは事前信頼性をほぼ正規データと重尾部データにかなり均等に広げる。$\nu$ の正確な事前分布を付録 A に示す。
<!-- Figure 2 depicts the descriptive model along with the prior distribution on its parameters. The ith datum from group j is denoted as yji at the bottom of the diagram. The data are described by t distributions, depicted in the middle of the figure. The prior distribution is indicated at the top of the figure. In particular, the prior on the mean parameters, mu_1 and mu_2, is assumed to be a very broad normal distribution, depicted in the diagram by an iconic normal shape. To keep the prior distribution broad relative to the arbitrary scale of the data, I have set the standard deviation S of the prior on mu to 1,000 times the standard deviation of the pooled data. The mean M of the prior on mu is arbitrarily set to the mean of the pooled data; this setting is done merely to keep the prior scaled appropriately relative to the arbitrary scale of the data. Thus, if y were a measure of distance, the scale could be nanometers or light-years and the prior would be equally noncommittal. The prior on the standard deviation parameter is also assumed to be noncommittal, expressed as a uniform distribution from a low value L, set to one thousandth of the standard deviation of the pooled data, to a high value H, set to one thousand times the standard deviation of the pooled data. Finally, the nu parameter has a prior that is exponentially distributed, which spreads prior credibility fairly evenly over nearly normal and heavy tailed data. The exact prior distribution on nu is shown in Appendix A. -->

<div class="figcenter">
<img src="../figures/2013Kruschke_Bayesian_fig2.svg" width="49%;">
<div class="figcaption">

図 2. ロバストベイズ推定の記述モデルの階層図<br/>
図の下部では，群 1 のデータを $y_{1i}$，群 2 のデータを $y_{2i}$ と表記する。 t 分布のアイコンからデータに下る矢印で示されるように，データは t 分布で記述されると仮定される。各矢印のチルダ記号 ($\tilde{}$) は，データがランダムに分布していることを示し，下矢印の $\cdots$ 記号 は，すべての $y_i$ が同一かつ独立に分布していることを示す。2 群は異なる平均パラメータ ($\mu_1$ と $\mu_2$) と異なる標準偏差パラメータ ($\sigma_1$ と $\sigma_2$) を持ち，分割された矢印で示されるようにパラメータは両群で共有され，合計 5 つのパラメータが推定される。パラメータは，図の上部のアイコンで示されるように，広範で非妥協的な事前分布で提供される。事前分布には，非常に大きな無作為標本による表現と，図 3-5 の事後分布のヒストグラムとの対応を示すために，ヒストグラム・バーが重ねられている。
<!-- Figure 2. Hierarchical diagram of the descriptive model for robust Bayesian estimation. At the bottom of the diagram, the data from Group 1 are denoted y1i and the data from Group 2 are denoted y2i. The data are assumed to be described by t distributions, as indicated by the arrows descending from the t-distribution icons to the data. The tilde symbol (tilde) on each arrow indicates that the data are randomly distributed, and the “. . .” symbol (ellipsis) on the lower arrows indicates that all the yi are distributed identically and independently. The two groups have different mean parameters (mu_1 and mu_2) and different standard deviation parameters (sigma_1 and sigma_2), and the   parameter is shared by both groups, as indicated by the split arrow, for a total of five estimated parameters. The parameters are provided with broad, noncommittal prior distributions, as indicated by the icons in the upper part of the diagram. The prior distributions have histogram bars superimposed on them to suggest their representation by a very large random sample and their correspondence to the histograms of the posterior distributions in Figures 3–5. -->
S:standard deviation; M:mean; L:low value; H: high value; R:rate; unif:uniform; shifted exp: shifted exponenti
al; distrib.:distribution.
</div></div>


t 検定が「計算手順（統計量の算出とp値の確認）」に依存するのに対し、このベイズ的アプローチは **「データの生成モデル（図2）を定義し、そこからパラメータの事後分布を推定する」** というプロセスそのものが推論の本体となる。

### 4. 推論過程：頻度論的検定との違い

頻度論的検定（t 検定）が「帰無仮説の下でそのデータが得られる確率（p値）」を計算するのに対し、BEST は以下のプロセスで推論を行います。

1. **確率の再配分（ベイズ更新）:** 図2で定義されたモデルと観測データに基づき、ベイズの定理を用いて、パラメータの確からしさ（信用度）を再配分します。
2. **MCMC 法によるサンプリング:** 数学的に解くことが難しい積分計算の代わりに、マルコフ連鎖モンテカルロ法（MCMC）を用いて、事後分布から大量のパラメータ値のサンプル（例: 100,000個）を生成します。
3. **事後分布の評価:** 得られたサンプル全体の分布（事後分布）を見ることで、平均値の差（$\mu_1 - \mu_2$）や標準偏差の差（$\sigma_1 - \sigma_2$）、効果量などの**「あり得る値の範囲と確率」**を直接可視化します。

### 5. 意思決定のルール（p値の代替）
p値による「有意・非有意」の二分法の代わりに、事後分布の形状に基づいて判断を下します。

*   **HDI (Highest Density Interval):** パラメータの最も信用できる値の範囲（95% HDIなど）。
    *   **差があるか:** 平均値の差の95% HDIが**ゼロを含まなければ**、「信用できる差がある」と判断します。
*   **ROPE (Region of Practical Equivalence):** 「実質的に等価とみなせる範囲」（例: 効果量 $\pm0.1$）。
    *   **差がないか:** 95% HDIが完全にROPEの中に収まる場合、「実質的に差がない（帰無仮説を受容する）」と判断します。これは従来のt検定では不可能な判断です。

### 6. 結論

* 図 2 のモデルに基づく BEST は、外れ値の影響を受けにくく、サンプルサイズや停止規則（いつ実験を止めたか）に依存しない一貫した結果を提供。
* 単なる検定（棄却の可否）を超えて、パラメータの豊かな情報を提供するこの手法への移行を推奨。


### 7. 意思決定のルール：HDI と ROPE
ベイズ推定では、p 値の代わりに以下の概念を用いて意思決定を行う。

* **HDI (Highest Density Interval):** パラメータの最も信用できる値の範囲（例えば95% HDI）。
    * **差の検定:** 平均値の差の95% HDIがゼロを含まなければ、グループ間に「信用できる差がある」と判断します。
* **ROPE (Region of Practical Equivalence):** 「実質的に等価とみなせる範囲」（例: 効果量 $\pm0.1$）。
    * **帰無仮説の受容:** 95% HDIが完全にROPEの中に収まる場合、その値は実用上ゼロと等しいとみなせ、「帰無仮説を受容」します。

### 8. 比較事例

論文では、以下のケースで t 検定と BEST の結果がどう異なるかを示している。

1. **外れ値がある場合:** t 検定は外れ値に弱く、有意差を見逃すことがありますが、BEST は t 分布を用いるため外れ値の影響を適切に処理し、群間の差や分散の違いを正確に検出できる。
2. **サンプルサイズが小さい場合:** BEST は推定の不確実性を HDI の広さとして正直に表現。
3. **サンプルサイズが大きい場合:** 差が極めて小さい場合、BEST は ROPE を用いて「実質的に差がない」と結論付けることができるが、t 検定はサンプルサイズが大きすぎると微小な差でも有意と判定してしまう（あるいは帰無仮説を受容できない）。

### 9. まとめ

* ベイズ推定は、t 検定よりも豊かな情報（完全なパラメータ分布）を提供し、外れ値に対して頑健。
* サンプリングの意図に依存しない一貫した結果をもたらす。
* t 検定の代わりにこのベイズ推定アプローチを使用すべき。
