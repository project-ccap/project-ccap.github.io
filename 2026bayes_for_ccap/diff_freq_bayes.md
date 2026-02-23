---
layout: home
---
<link href="/asamarkdown.css" rel="stylesheet">

## 頻度論的 t 検定とベイズ流推論: 2 群（A,B）の平均差 

データ: $y_{Ai}, i=1,\dots,n_A、y_{Bj}, j=1,\dots,n_B$。標本平均 $\bar{y}_A,\bar{y}_B$, 標本分散 $s_A^2,s_B^2$。

### (1) 頻度論（t 検定）: 「仮説が真だとしたとき、これ以上に極端なデータが出る確率」を計算

典型は帰無仮説 $H_0:\mu_A=\mu_B$ に対し、対立仮説 $H_1:\mu_A\neq\mu_B$。

#### 等分散を仮定する場合（Student）：

プール分散：$\displaystyle s_p^2=\frac{(n_A-1)s_A^2+(n_B-1)s_B^2}{n_A+n_B-2}$ を用いて，t 統計量

$$
t=\frac{\bar y_A-\bar y_B}{s_p\sqrt{\frac{1}{n_A}+\frac{1}{n_B}}}
$$

自由度 $\nu=n_A+n_B-2$。p 値: $\displaystyle p = \Pr$  $\left(\left\|T_{\nu}\right\| \ge \left\|t_{\text{obs}}\right\| \mid H_0\right)$.


#### 等分散が仮定できない場合（Welch）：

$$
t=\frac{\bar y_A-\bar y_B}{\sqrt{\frac{s_{A}^2}{n_{A}}+\frac{s_{B}^2}{n_{B}}}}
$$

自由度は Welch–Satterthwaite 近似で決める。

この枠組みの出力は「p 値」と「（頻度論的）信頼区間」。たとえば平均差 $\Delta=\mu_A-\mu_B$ の 95 %信頼区間は
$$
\hat\Delta \pm t_{0.975,\nu},\mathrm{SE},
\quad \hat\Delta=\bar y_A-\bar y_B.
$$
重要点：95 %信頼区間は「$\Delta$ が 95 %の確率で入る区間」ではない。手続きの長期頻度の主張。

### (2) ベイズは「未知量（平均差や効果量）の事後分布」を計算

ベイズの焦点は $\Delta=\mu_A-\mu_B$ や効果量 $d$ の事後分布 $p(\Delta\mid D)$ を直接得ること。

最小構成（正規モデル、等分散）：
$$
y_{Ai}\sim\mathcal N(\mu_A,\sigma^2),\qquad
y_{Bj}\sim\mathcal N(\mu_B,\sigma^2).
$$

事前分布を置く（例）：

$$
\mu_A,\mu_B \sim \mathcal N(\mu_0,\tau^2),\qquad
\sigma^2\sim \mathrm{Inv\text{-}Gamma}(a,b)
$$

（あるいは $\sigma$ に half-Cauchy 等）。すると事後 $p(\mu_A,\mu_B,\sigma^2\mid D)$ が得られ、平均差は
$$
\Delta=\mu_A-\mu_B
$$
を事後サンプルから計算できる。出力は

* 事後平均・中央値 $\mathbb{E}[\Delta\mid D]$
* 信用区間（credible interval）

例：95 %信用区間 $[q_{0.025},q_{0.975}]$ は「$\Delta$ がこの区間にある確率が 0.95」という意味で読める（このモデルと事前の下で）。

効果量でやるなら
$$
d=\frac{\mu_A-\mu_B}{\sigma}.
$$
同様に事後から d の分布 p(d\mid D) を得る。

### (3) 仮説検定の対比：p値 vs Bayes factor（あるいは事後確率）

頻度論の「検定」は、p 値が閾値（例 0.05）以下かどうか。

ベイズで「$H_0$ と $H_1$ を比較する」すなわち Bayes factor：
$$
\mathrm{BF}_{10}=\frac{p(D\mid H_1)}{p(D\mid H_0)}.
$$
ここで
$$
p(D\mid H_k)=\int p(D\mid\theta_k,H_k),p(\theta_k\mid H_k),d\theta_k
$$

が周辺尤度（エビデンス）。$\mathrm{BF}_{10}>1$ ならデータは $H_1$ を相対的に支持、$\mathrm{BF}_{10}<1$ なら $H_0$ を相対的に支持。

「 t 検定と同じ入力で動くベイズ検定」をするなら、Rouder et al. 型の “Bayesian t-test”。
効果量 $d$ に Cauchy 事前を置いて $\mathrm{BF}_{10}$ を閉形式/数値で計算する流儀が広く使われる（実装も豊富）。

重要点：p 値は $p(H_0\mid D)$ ではない。一方で Bayes factor は「H0とH1の相対的証拠」を直接出す。ただし BF は事前（特に H_1 側の効果量事前）に感度を持つ。

### (4) 「差があるか」以外に、ベイズが自然に答える問い

心理・神経系で効くのはここ。例：

* 「$\Delta>0$ の確率は？」
$$
\Pr(\Delta>0\mid D)=\int \mathbb{I}(\Delta>0),p(\Delta\mid D),d\Delta
$$

* 「臨床的に意味のある差（ROPE）を超える確率は？」
ROPE を $[-\epsilon,\epsilon]$ とすると
$$
\Pr(|\Delta|\le \epsilon\mid D)
$$
を出せる。
* 階層化（被験者差・刺激差）を入れても同じ形式で推論できる（t検定はここで破綻しやすい）。

### (5) 何が「同じ」で何が「違う」か
同じ：平均差を標準誤差で割った統計量や、正規性などの仮定の部分は共有されうる。線形ガウスモデルでは、ベイズの事後平均が頻度論推定と一致/近似する状況も多い。
違う：解釈の対象。頻度論は「データの珍しさ」を測り、ベイズは「パラメータ（差）の不確実性」を更新する。検定出力も、p値 vs 事後（信用区間・確率・BF）で異なる。

### (6) 実務的な使い分け（2群差に限定した、無駄のない指針）
*  「差の大きさと不確実性」を報告したい、あるいは「意味のある差を超えた確率」を言いたい → ベイズ（事後＋信用区間＋ROPE）が直截。
*  「H0支持（差がない方向の証拠）も示したい」→ p値では無理が出るので Bayes factor が有利。
*  ただし BF は事前感度があるので、効果量事前の根拠（弱情報・スケール）を明示するのが最低条件。

参考として、t検定とベイズ検定の対比を前面に出した代表的な系統は「Bayesian t-test（Rouder/Wagenmakers周辺）」と、推定重視の「BEST（Kruschke）」あたりが典型。必要なら「どれを引用すべきか（心理学向け/工学向け）」を、あなたの用途（実験・階層化・モデル比較）に合わせて絞って提示する。