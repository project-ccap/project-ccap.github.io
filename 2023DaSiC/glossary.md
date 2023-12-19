---
title: 用語集
layout: default
author: CCAP プロジェクト
---

#### 分野によって異なる訳語を持つ用語

* argument : 機械学習においては，引数 (ひきすう)，言語学においては 項 (こう)
* LDA: 統計学においては，線形判別分析 Linear Discriminant Analysis, 機械学習においては潜在ディレクリ配置 Latent Dirichllet Allocation を表す場合もある。
* Lemma: 数学においては，補題，言語学においては，辞書項目，原形
* ML: 機械学習の頭文字 ML:machine learning, または最尤推定 maximam likelihood, 一般にはメーリングリスト mailing list,

# 用語集 glossary

**太字** は今回のキーワードを表す

* BERT (**B**idirectional **E**ncoder **R**epresentations from **T**ransformers): Google が開発した Transformer に基づく言語モデル。マスク化言語モデルと次文予測課題によって事前訓練を行い，各下流課題に対して微調整 (fine turing) を行うことで SOTA を達成した。
* GPT (Generative Pretrained Transformer): [OpenAI 社](https://openai.com/) の開発した Transformer に基づく生成 AI モデルの一つ。
* LLM (Large Language Model): 大規模言語モデル。大規模コーパスによって訓練された言語モデル。近年の BERT, GPT 等の言語モデルはすべて LLM である。
* LM (Language Model): 言語モデル。伝統的には，直前までの単語から次単語を予測するモデルを指す。
* ML: 機械学習 machine learning を指す場合と，最大尤度 Maximum Likelihood を指す場合とがある。まれにメーリングリスト mailing list を指す場合もある。
* DL: 深層学習 deep learning，多層ニューラルネットワークのこと。
* LangChain: LLM の API を提供している。
* PPO (Proxical Policy Gradient): 近位方針勾配法。強化学習の最適化手法の一つ。価値勾配法 に対して，方針 (policy) 勾配を用いる。[Schullman+2017](https://arXiv.org/abs/1707.06347)
* RNN (Recurrent Neural Networks): 再帰的ニューラルネットワーク。系列情報を扱うためのニューラルネットワークモデル。Elman ネット，Jordan ネット，LSTM, GRU などが含まれる。
* SOTA (State of the Art): 現時点での最高性能のこと。
* Transformer: [Vaswani+2017](https://arXiv.org/abs/1706.03762) によって提案された RNN の代替モデル。マルチヘッド注意 (MHSA) に基づく処理機構。
* **双対性 duality** あるいは **双対問題 dual problem**: 多くの場合，双対問題とは Lagrange の双対問題を指す。Lagrange の双対問題は，非負の乗数を用いて目的関数を変形し，異なる目的関数を用いて最適化問題を解くことを指す。最小化問題の Lagrange 方程式は，Lagrange 乗数で表現される問題の最大化である。このため，目的関数の最小化は，Langrage 乗数に用いられる関数の最大化とみなしうる。このような一方の関数の最大化が，他方の関数の最小化であるような問題を双対問題と呼ぶ。発話エラーに即して考えれば，発話意図を正しく音声化する問題と，発話意図が伝わらないリスクを最小化するという制約条件とに相当すると考えることが可能であろう。
* カルバック=ライブラー・ダイバージェンス (Kullback–Leibler divergence): 2 つの確率密度関数の差異を定義する値。機械学習においては，目的関数とモデル出力との間で，カルバック=ライブラー・ダイバージェンスを用いる場合がある。
<!-- 確率分布 A と B とのカルバック=ライブラー・ダイバージェンスを $KL(A||B)$ などと表記する。確率分布間の距離に相当する。$KL(A||B)\ne KL(B||A)$ すなわち，A から見た B の距離と，B から見た A の距離とが等しいとは限らないため，偽距離と呼ばれることもある。 -->
* **ソフトマックス softmax**: 実数値を要素とするベクトルを，離散記号に変換する場合，最大値の値を大きくし，他の要素は 0 に近づける操作を行う場合がある。このときに用いられる変換がソフトマックス変換，あるいはソフトマックス関数という。ソフトマックス関数は，識別や分類を行う機械学習モデルの最終層に用いられワンホットベクトルを得る場合用いられる。また，その性質から Transformer ベースの注意機構の実装にも用いられる。物理学のボルツマン分布，エネルギー関数と式としては同一。$\displaystyle f(x_i)=\frac{e^{x_i}}{\sum e^{x_i}}$. 左辺 LHS の 分母 the denominator は，分配関数 partition function と呼ばれる。
* **ワンホットベクトル one-hot vector**: 言語処理に用いられるニューラルネットワークモデルでは，入出力が，単語や文字など記号表現である場合が多い。任意の記号を表す場合に，その記号に該当する位置だけが 1 で，他の要素はすべて 0 であるベクトルを用いる。このようにして作成されたベクトルを 1 つだけが熱く，他の要素がすべて冷たい，ワンホットベクトル，あるいはワンホット表現と呼ぶ。
* **単語埋め込み word embeddings**: 単語のベクトル表現。古くは潜在意味解析 (Latent Semantic Analysis) なども含まれるが，[word2vec](https://arXiv.org/abs/1301.3781) や [GloVe](http://nlp.stanford.edu/projects/glove/) などが代表的モデルである。
* **微調整 Fine-tuning**: 事前学習を施したモデルに対して，課題に合わせて再学習を行うこと。
* 文脈注入 (Context Injection): プロンプトの一つ。
* **転移学習 Transfer learnign**: 従来は最終層のみを課題に合わせて入れ替えて，最終層と最終直下層 (penultimate layers) の結合係数のみを調整することを指した。微調整 (fine-tuning) との相違は，再学習させる層の相違である。
* ロジスティック回帰: 回帰と名がつくが，分類問題を解くための手法。出力を確率と仮定して シグモイド関数 (logistic sigmoid functions) を用いる。
* シグモイド関数: $f(x)=\left(1 + e^{-x}\right)^{-1}$ 連続量を確率的判断に変換する。すなわち 2 値 (真偽値 true or false, 裏表 head or tail, p であるか p でないか $p$ or $1-p$ など。ニューラルネットワークでは伝統的に用いられてきた経緯がある。理由は，微分が極端に簡単になることが挙げられる。現在では ハイパータンジェント tanh や，整流線形関数 ReLU (Recutified Linear Unit) が用いられる場合が多い。理由は，勾配消失問題対策のため。
- 交差エントロピー損失: エントロピー $- p\log p$ は，熱力学と情報論とで用いられる概念。熱力学の第二法則，時間の矢 に関連。情報理論では，情報量の定義。機械学習では，分類問題の損失関数として頻用される。$-\left(t \log p + (1-t) \log(1-p)\right)$
- [次元圧縮 t-SNE](https://komazawa-deep-learning.github.io/t-SNE/) 2008 年の提案以来，よく見かけるようになった次元圧縮手法。
- サポートベクターマシン: ウラジミール・ヴァプニク (Vapnik) による 教師あり学習 (Vapnik 1999, 1998). ディープラーニング以前に主流であった。2 群分類で特徴を最もよく (マージン最大化) 分離する境界面決定アルゴリズム。カーネルトリック，スラック変数の導入。線形回帰，線形判別に比べて性能が出ると考えられていた。今でも，最終層における判別に応用されることがある。
- 重み崩壊 weight decay: カラス=クーン=タッカー条件(KKT Karush-Kuhn-Tucker condition)を ラグランジェ未定乗項 Lagrange's multpliers 付きで解くことであり，変分問題と一応用とみなしうる。

<!-- <center>-->
<!--<img src="figures/2015scikit-learn-0.16_svm_p150.jpg" style="width:66%"><br/>-->
<!-- 出典: scikit-learn マニュアル-->
<!-- </center> -->

* 確率的勾配降下法 SGD: stochastic gradient descent methods. Bottou+2007 によって導入された機械学習における学習法の工夫の一つ。ミニバッチの導入を特徴とする。オンライン学習とバッチ学習の中間で，学習データをランダムサンプリングして学習に用いる。精度改善の手法ではないが，計算時間の削減に貢献。ビッグデータの使用を前提とする現代的な実用手段

<!-- <center>-->
<!-- <img src="figures/2007Bottou_NIPSpage30.svg" style="width:77%">  -->
<!-- </center> -->

* ニューラルネットワーク neural networks: 脳の神経細胞の活動を模した処理単位から構成される情報処理モデルのこと。一つ一つの処理単位は，複数個の入力を受け取って，一つの値を出力する素子である。複数個の入力とは他の神経細胞から与えられる信号である。これら入力信号を神経細胞間の結合の強度を表す重み (結合係数) に従って各入力信号が重み付けされる。出力信号は，これら入力信号の重み付け荷重和に基づいて算出される。
* パーセプトロン perceptron: Rosenblatt によって提案された初期の画像認識モデル。第一次 AI ブーム 1950-1960 年代。パーセプトロンにおける学習は，最終層と最終直下層の間の結合係数に限定される。
* 多層パーセプトロン multi-layer perceptron: 中間層を持つニューラルネットワークの総称。畳み込み層やリカレント結合を含まず，層間の素子が全て結合しているような全結合を指す場合が多い。
* フィードフォワード feed-forward: 信号が次層または次の素子 (神経細胞，ニューロン) へ送られる結合，あるいはデータの流れ。入力データはフィードフォワードで次層へ伝播するが，誤差信号は逆に上位層から下位層へ向かって逆伝播する。
* 畳み込み convolution: 主に視覚信号処理で用いられる演算。ヒューベルとウィーゼルによって発見された視覚野にみられる特徴検出細胞を模した演算処理を言う。畳み込み演算を複数層に渡って積み重ねたモデルを深層ニューラルネットワークまたはディープニューラルネットワークと呼ぶ。
* 信用割当問題 credit assignment problem: 強化学習においては，一連の経験において，どれ事象が重要だったのかを判定する問題。一方，多層ニューラルネットワークにおいては，下位層のどの素子が判断にとって重要であったかを知る方法ことが難しいことを指す。
* 特徴エンジニアリング feature engineering: 視覚的あるいは聴覚的な特徴を抽出する際の技法，設計手法，理念のこと。
* RNN リカレントニューラルネットワーク (SRN, Elman, Jordan, LSTM, GRU): 単純再帰型ニューラルネットワークのこと。自分自身の出力に対してフィードバック結合を有するネットワークであり，時系列処理，言語処理などに用いられる。
* SOTA: State of the arts (現時点での) 最高性能の意。
* 活性化関数 activation functions: ニューラルネットワークにおいて，入力信号を出力信号に変換する場合に非線形変換が行われるが，その変換に用いられる関数のこと。シグモイド (sigmoid) 関数，ハイパータンジェント (tanh)，整流線形 (ReLU) などが代表的。
* 誤差逆伝播法 back-propagation {gradient descent algorithm}: 関数最適化に用いられる手法の一つ。多層ニューラルネットワークにおいては各素子の結合状態を，素子間の結合を信号とは逆方向にたどることで誤差の伝播させるため，逆伝播と呼ばれる。実際の神経系においては，誤差信号が逆伝播する証拠は得られていない。しかし，脳神経細胞の結合にはしばしば，順方向の結合のみならず，逆方向の結合が観察できることから，誤差逆伝播法と同等の処理が行われている可能性が指摘されている。
* 勾配降下法 gradient descent methods: 最小化すべき関数をそのパラメータ関する勾配の逆方向に向かってパラメータを逐次変化させることで求める最適値を探すための手法である。盲目の登山者アナロジー (blind hiker's analogy) として知られる。
* 勾配消失問題，勾配爆発問題 gradient vanishing problems, gradient exploding problems
* 目的関数，誤差関数，損失関数 objective/error/loss functions: 平均自乗誤差，負の対数尤度，交差エントロピーなどで定義され，出力信号と教師信号との乖離を表す項と正則化項との和で定義される場合が多い。
* 平均自乗誤差 MSE (mean square errors), 負の対数尤度 NLL (negitive log likelihood), 交差エントロピー CE (cross entropy), 最大尤度 ML (maximum likelihood), カルバック・ライブラーダイバージェンズ KL-divergence,
* EM アルゴリズム EM algorithm: Dempster+1977 によって提唱された，パラメータの反復推定法。現在までのデータと知識に基づいて，求めるパラメータの推定値を算出し (E ステップ)，推定した値に基づいてモデルの尤度を最大化するようパラメータを更新する (M ステップ)。これら E, M 両ステップを反復することで，パラメータ推定を行う手法である。Neal&Hinton1993 による EM アルゴリズムの変分解釈によりニューラルネットワークと機械学習への応用範囲が広がった。
* 最適化手法 optimization methods (SGD, AdaGrad, AdaDelta, RMSprop, Adam): パラメータを調整するために用いられる勾配降下法や最尤推定法などを指す。目的関数を最適化するために用いられる。
* データセット dataset: 機械学習においては，データを訓練データ，検証データ，テストデータの 3 種類に分ける場合がある。訓練データを用いてモデルのパラメータ更新を行い，検証データを用いてモデルの汎化性能の検証する。最後にテストデータを用いて最終的なモデルの性能評価を行う。所与のデータを訓練，検証，テストのいずれに振り分けるのかは，コンテンスト主催者によって予め定められている場合もあれば，勝手に定めて良い場合もある。モデルの性能評価で用いられる検証，テストデータに，パラメータ推定に用いた訓練データが含まれている場合をデータ漏洩 (data leakage) と呼び，モデルの性能を不当に見積もることに繋がる。
* 正則化項，あるいはモデルの制約と呼ばれる場合もある: L0, L1, L2 などが区別される。L に続く数字の意味は制約項の肩に乗る数の意味。すなわち L0 であれば $|x|^{0}$, L1 であれば $|x|^2$ などとなる。
* ドロップアウト dropout: ニューラルネットワークの結合をランダムに間引くこと。これにより，モデルの冗長性，堅牢性，汎化性能の向上が期待できる。
* 過学習，未学習 over/under fitting: 学習のし過ぎで，訓練データには当てはまるが，未学習データ，すなわち検査データや検証データに対しての当てはまりが悪くなる場合を過学習と呼び，反対に，訓練データに対しても改善の余地がある場合を未学習と呼ぶ。
* バッチ正則化 batch normalization: データを正則化 (この場合，平均 0 分散 1 に変換すること) して入力信号とすれば，学習が容易である場合が多い。このため，ミニバッチに対して，毎回正則化を行うこと。
* 自己符号化器 auto-encoders: 入力信号を復元するような処理に用いられる。内部表象に入力情報の圧縮表現が得られる。
* 強化学習 reinforcement learning: 他のニューラルネットワークモデルが，教師信号を正しく予測することを目的関数として居るのに対して，強化学習のそれは，環境から与えられる報酬を最大化することを目的関数とする。環境へより良く適応することと考えれば生物学，進化理論と関連し，不適応と考えれば精神病理への応用もなされている。DQN と呼ばれるモデルが 2014 年に囲碁で世界レベルの人間を破ったことで話題となった。を通常の機械学習分野での用語と異なる専門用語が用いられるが，不自然な仮定ではない。頻用される用語としては，以下のものが挙げられる: 環境 environment, 状態 state, 行為 action, 報酬 reward, 価値 value, Q 値 q-value, 方針 policy, 時間遅延 temporal difference TD, REINFORCE (初期から用いられているアルゴリズム名), マルコフ決定過程 MDP: Markov Decision Process, 部分観察マルコフ決定過程 Partially Observed Markov Decision Process:POMDP, SARSA (アルゴリズム名 State-Action-Reward-State-Action の頭文字), 経験再生 experice replay, アドバンテージ advantage, 血糖 ネットワーク duealing network, 二重 Q アルゴリズム double-Q, 非同期アドバンテージ行為者‐批評家 (アルゴリズム) A3C: Asyncronious Advantage Actor-Critic, 近位方針最適化 PPO: Proximal Policy Optimization なお PPO は chatGPT でも採用されて 2022 年から 2023 年にかけて話題となった。

<!-- # 用語集 glossary
* BERT (**B**idirectional **E**ncoder **R**epresentations from **T**ransformers): Google が開発した Transformer に基づく言語モデル。マスク化言語モデルと次文予測課題によって事前訓練を行い，各下流課題に対して微調整 (fine turing) を行うことで SOTA を達成した。
* GPT (Generative Pretrained Transformer): [OpenAI 社](https://openai.com/) の開発した Transformer に基づく生成 AI モデルの一つ。
* LLM (Large Language Model): 大規模言語モデル。大規模コーパスによって訓練された言語モデル。近年の BERT, GPT 等の言語モデルはすべて LLM である。
* LM (Language Model): 言語モデル。伝統的には，直前までの単語から次単語を予測するモデルを指す。
* LangChain: LLM の API を提供している。
* PPO (Proxical Policy Gradient): 近位方針勾配法。強化学習の最適化手法の一つ。価値勾配法 に対して，方針 (policy) 勾配を用いる。[Schullman+2017](https://arXiv.org/abs/1707.06347)
* RNN (Recurrent Neural Networks): 再帰的ニューラルネットワーク。系列情報を扱うためのニューラルネットワークモデル。Elman ネット，Jordan ネット，LSTM, GRU などが含まれる。
* SOTA (State of the Art): 現時点での最高性能のこと。
* Transformer: [Vaswani+2017](https://arXiv.org/abs/1706.03762) によって提案された RNN の代替モデル。マルチヘッド注意 (MHSA) に基づく処理機構。
* カルバック=ライブラー・ダイバージェンス (Kullback–Leibler divergence): 2 つの確率密度関数の差異を定義する値。確率分布 A と B とのカルバック=ライブラー・ダイバージェンスを $KL(A||B)$ などと表記する。確率分布間の距離に相当する。機械学習においては，目的関数とモデル出力との間で，カルバック=ライブラー・ダイバージェンスを用いる場合がある。$KL(A||B)\ne KL(B||A)$ すなわち，A から見た B の距離と，B から見た A の距離とが等しいとは限らないため，偽距離と呼ばれることもある。
* ソフトマックス: 実数値を要素とするベクトルを，離散記号に変換する場合，最大値の値を大きくし，他の要素は 0 に近づける操作を行う場合がある。このときに用いられる変換がソフトマックス変換，あるいはソフトマックス関数という。ソフトマックス関数は，識別や分類を行う機械学習モデルの最終層に用いられワンホットベクトルを得る場合用いられる。また，その性質から Transformer ベースの注意機構の実装にも用いられている。
* ワンホットベクトル (one-hot vector): 言語処理に用いられるニューラルネットワークモデルでは，入出力が，単語や文字など記号表現である場合が多い。任意の記号を表す場合に，その記号に該当する位置だけが 1 で，他の要素はすべて 0 であるベクトルを用いる。このようにして作成されたベクトルを 1 つだけが熱く，他の要素がすべて冷たい，ワンホットベクトル，あるいはワンホット表現と呼ぶ。
* 単語埋め込み (word embeddings): 単語のベクトル表現。古くは潜在意味解析 (Latent Semantic Analysis) なども含まれるが，[word2vec](https://arXiv.org/abs/1301.3781) や [GloVe](http://nlp.stanford.edu/projects/glove/) などが代表的モデルである。
* 微調整 (Fine-tuning): 事前学習を施したモデルに対して，課題に合わせて再学習を行うこと。
* 文脈注入 (Context Injection): プロンプトの一つ。
* 転移学習 (Transfer learnign): 従来は最終層のみを課題に合わせて入れ替えて，最終層と最終直下層 (penultimate layers) の結合係数のみを調整することを指した。微調整 (fine-tuning) との相違は，再学習させる層の相違である。
* ロジスティック回帰: 回帰と名がつくが，分類問題を解くための手法。出力を確率と仮定して シグモイド関数 (logistic sigmoid functions) を用いる。
* シグモイド関数: $f(x) = \left( 1 + e^{-x}\right)^{-1}$ 連続量を確率的判断に変換する。すなわち 2 値 true or false, head or tail, $p$ or $1-p$ など。ニューラルネットワークでは伝統的に用いられてきた経緯がある。理由は，微分が極端に簡単になることが挙げられる。現在では ハイパータンジェント tanh や，整流線形関数 ReLU (Recutified Linear Unit) が用いられる場合が多い。理由は，勾配消失問題対策のため。
* ソフトマックス関数 (softmax function): 多値分類に用いられる。物理学のボルツマン分布，エネルギー関数と式としては同一。$\displaystyle f(x_i)=\frac{e^{x_i}}{\sum e^{x_i}}$. 左辺 LHS の 分母 the denominator は，分配関数 partition function と呼ばれる。
- 交差エントロピー損失: エントロピー $- p\log p$ は，熱力学と情報論とで用いられる概念。熱力学の第二法則，時間の矢 に関連。情報理論では，情報量の定義。機械学習では，分類問題の損失関数として頻用される。$-\left(t \log p + (1-t) \log(1-p)\right)$
- [次元圧縮 t-SNE](https://komazawa-deep-learning.github.io/t-SNE/) 2008 年の提案以来，よく見かけるようになった次元圧縮手法。
- サポートベクターマシン: ウラジミール・ヴァプニク(Vapnik) による 教師あり学習 (Vapnik 1999, 1998). ディープラーニング以前に主流であった。2 群分類で特徴を最もよく (マージン最大化) 分離する境界面決定アルゴリズム。カーネルトリック，スラック変数の導入。線形回帰，線形判別に比べて性能が出ると考えられていた。今でも，最終層における判別に応用されることがある。カラス=クーン=タッカー条件(KKT Karush-Kuhn-Tucker condition)を ラグランジェ未定乗項 Lagrange's multpliers 付きで解く
<center>
<img src="figures/2015scikit-learn-0.16_svm_p150.jpg" style="width:66%"><br/>
出典: scikit-learn マニュアル
</center>
* 確率的勾配降下法 SGD: stochastic gradient descent methods. Bottou ら (2007) によって導入された機械学習における学習法の工夫の一つ。ミニバッチの導入を特徴とする。オンライン学習とバッチ学習の中間で，学習データをランダムサンプリングして学習に用いる。精度改善の手法ではないが，計算時間の削減に貢献。ビッグデータの使用を前提とする現代的な実用手段

<center>
<img src="figures/2007Bottou_NIPSpage30.svg" style="width:77%">
</center>

- 多層パーセプトロン MLP: multi-layer perceptron:
* neural networks,
* perceptron, multi-layer perceptron, feed-forward
* convolution, feature engineering
* RNN (SRN, Elman, Jordan, LSTM, GRU)
* activation functions {sigmoide,tanh,ReLU}
* back-propagation {gradient descent algorithm}
* gradient vanishing problems, gradient exploding problems
* objective/error/loss functions
* MSE (mean square errors), NLL (negitive log likelihood), CE (cross entropy), ML (maximam likelihood), KL-divergence, EM algorithm
* optimization methods (SGD, AdaGrad, AdaDelta, RMSprop, Adam)
* dataset {training/validation/test}
* L0, L1, L2 normalization
* dropout
* over/under fitting
* batch normalization, skip-connection, ResNet
* auto-encoders
* reinforcement learning {environment, state, action, reward, value, q-value, policy, TD, REINFORCE, MDP, POMDP, SARSA, experice replay, advantage, duealing, double-Q, A3C}
 -->
