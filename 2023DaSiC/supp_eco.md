---
title: "DaSiC7 (2023) 発表資料"
author: 浅川伸一
layout: default
codemirror_mode: python
codemirror_mime_type: text/x-cython
---

# Colaboratory 事始め

- [Python](https://www.python.org/){target="_blank"} って何？
    - AI や 機械学習 分野の共同体で使われることが多いコンピュータ言語のことです。下記に示すように高等学校の情報で採択されます。
    - [StackOverFlow におけるコンピュータ言語のトレンド](https://insights.stackoverflow.com/trends?tags=r%2Cpython%2Cjavascript%2Cjava%2Cc%2B%2B%2Cc%23){target="_blank"}

- [Jupyter notebook](https://jupyter.org/){target="_blank"} って何？
    - Python をブラウザ上で動かすシステム，あるいはその環境を指します。
    - 木星を表す ジュピター jupiter とは綴りが異なります。ですが由来は 木星 から来て言います。
- [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb){target="_blank"} って何？
    - Jupyter notebook をクラウド上で実行する環境です


- ブラウザを立ち上げて [https://colab.research.google.com/notebooks/welcome.ipynb?hl=ja](https://colab.research.google.com/notebooks/welcome.ipynb?hl=ja) にアクセスしてください。

- [Colaboratory へようこそ](https://colab.research.google.com/notebooks/intro.ipynb?hl=ja)
- [Google Colaboratory のチュートリアルビデオ](https://youtu.be/inN8seMm7UI) も参照してください
- [外部データとのやりとり，ダウンロード，アップロードなど](https://colab.research.google.com/notebooks/io.ipynb)

# 簡単な説明

- Colab とはブラウザ上 [jupyter notebook](https://jupyter.org/) ([jupyter lab](https://jupyterlab.readthedocs.io/en/stable/)) を実行するクラウド計算環境です。
- 従って，ブラウザさえあれば，特別なインストール作業を必要ありません。
- jupyter notebook とは [ipython](https://ipython.org/notebook.html) をブラウザ上で実行する環境です。
- ipython とは [python](https://pytorch.org/) をインタラクティブに実行する環境です。
- python とは機械学習やニューラルネットワークのコミュニティで使われるコンピュータ言語です。
- したがって colab の使い方は jupyeter notebook をご存知であれば，ほぼ同じです。
- クラウドベースの実行系ですので，インストールの手間は不要です。

# 簡単な操作方法

- notebook は セル と呼ばれる単位から成り立っています。
- セルは コード，テキスト，画像で構成されます。
- Google Doc に保存可能で Google Drvie 経由シェアできます
- テキストはマークダウン形式で書きます
- [少し時間がかかりますが Keras によるニューラル画像変換を実行](https://colab.research.google.com/github/ten
sorflow/models/blob/master/research/nst_blogpost/4_Neural_Style_Transfer_with_Eager_Execution.ipynb)  してみ
ることをお勧めします
- [https://research.google.com/seedbank/seeds](https://research.google.com/seedbank/seeds)
- [How to Train Your Models in the Cloud](https://youtu.be/Bgwujw-yom8)
- [https://colab.research.google.com/notebooks/basic_features_overview.ipynb](https://colab.research.google.
com/notebooks/basic_features_overview.ipynb)
- [http://colab.research.google.com/](http://colab.research.google.com/)

---
author:
- 'Google[^1]'
title: |
    Google Colaboratory Frequently Asked Questions)
---

オリジナル URL: <https://research.google.com/colaboratory/faq.html>

- Colaroboratory とは何ですか (What is Colaboratory?)

Google codelaboratory は械学習教育研究用ツールです。環境構築に必要な事前のセットアップが不要なjuypter notebook 環境です。

- サポートしているブラウザは何ですか (What browsers are supported?)

Google Colaboratory は大抵のブラウザで動作します。動作テスト済ブラウザは，
[Chrome](https://www.google.com/chrome/browser/desktop/index.html) と
[Firefox](https://www.mozilla.org/ja/firefox/)です。

- 無料で使えますか (Is it free to use?)

無料です。

- ジュピター(jupyter)と colaboratory の違いは何ですか (What is the difference between Jupyter and Colaboratory?)

Colaboratory はオープンソースの
[ジュピター(Jupyter)](https://jupyter.org/) を元にしています。
ジュピターノートブック(Juypter
notebook)を他のユーザと共有することができます。
ローカルな環境に対する，ダウンロード，インストール，実行，などは必要ありません。

- colaboratory.jupyter.org との関係ありますか (How is this related to colaboratory.jupyter.org?)

2014 年に jupyter 開発チームと我々は本ツールの初期バージョンを共同で開発していました。
以来 colaboratory は発展し続け，グーグル内部用途となっています。

- Notebook はどこに保存されますか，また，シェアできますか ( Where are my notebooks stored, and can I share them?)


Colaboratory ノートブックは [Google Drive](https://drive.google.com/) に保存されます。
Google Docs や sheets と同様にシェア可能です。右上のシェアボタンを押してください。
Google Drive の
[ファイル共有の手引き](https://support.google.com/drive/answer/2494822?co=GENIE.Platform%3DDesktop&hl=en) に従ってください。

- Notebook をシェアする場合，何がシェアされますか (If I share my notebook, what will be shared?)

Notebook のシェアを選択すると，テキスト，コード，出力という noteobook の全内容が共有されます。
コードセルの出力を抑制させることは可能です。出力をシェアするには，保存時に
*編集 \> ノートブック設定 \> コードセル出力の抑制* を選択してください。
仮想環境の使用時は，使用時の設定ファイルやライブラリはシェアされません。ですので，ライブラリのインストールやカスタマイズをセル内に含めておくと良いでしょう。
[libraries](https://colab.research.google.com/notebooks/snippets/importing_libraries.ipynb)
や [files](https://colab.research.google.com/notebooks/io.ipynb)
を参照してください。

- 二人のユーザが同時に同じ notebook を編集したらどうなりますか (What happens if two users edit the same notebook at the same time?)

変更の反映は即時に全員へなされます。グーグル Docs
の編集結果が編集中全ユーザに視認可能なのと同様です。

- 以前作成したジュピター(IPython)ファイルをインポートすることはできますか (Can I import an existing Jupyter/IPython notebook into Colaboratory?)

できます。ファイルメニューからノートブックのアップロードを選んでください

- Python3 (R, Scale)って何ですか (What about Python3? (or R, Scala, \...)

Colaboratory は，Python バージョン 2.7 と Python バージョン 3.6
をサポートしています。 これら以外の R や Scale
といったジュピターカーネルの使用を希望するユーザがいることは承知しています。
将来的にはサポートするつもりですが，時間の制約が実現していません。

- Colaboratory ノートブックを検索するには？(How can I search Colaboratory notebooks?)

検索ボックスの[ドライブ](https://drive.google.com)を選んでください。
左上にある colaboratory のロゴをクリックすればグーグルドライブ上の全てのファイルを閲覧できます。
**ファイル-\>最新のファイル**を開けば，最近閲覧したファイルを開くことができます。

- コードはどこで実行されるのですか。ブラウザのウィンドウを閉じてしまったら実行中のコードはどうなりますか (Where is my code executed? What happens to my execution state if I close the browser window?)

コードは仮想マシンで実行されます。仮想マシンはしばらく放置するとリフレッシュされます。
仮想マシンがリフレッシュされるまでの最長寿命はシステムに寄ります[^2]。

- 自分のデータを出力できますか (How can I get my data out?)

 colaboratory 上で作成したノートブックをグーグルドライブからダウンロードできます。
[instructions](https://support.google.com/drive/answer/2423534) や
 colaboratory のファイルメニューの中を御覧ください。
 colaboratory ノートブックは，オープンソースのジュピターノートブック形式(拡張子
.ipynb)で保存されています。

- GPU は利用できますか。どうして GPU が使えない場合があるのですか (How may I use GPUs and why are they sometimes unavailable?)

 colaboratory はインタラクティブな利用を想定しています。バックグラウンドで
GPU
を長時間実行すると停止させる場合があります。 colaboratory を仮想通貨のマイニングに使わないでください。仮想通貨マイニングはサポート対象外です。
長時間の継続利用に際は[ローカルランタイムe](https://research.google.com/colaboratory/local-runtimes.html)を推奨しています。

- 仮想マシンの実行をリセットできますか。どうしてリセットできない時があるのですか (How can I reset the virtual machine(s) my code runs on, and why is this sometimes unavailable?)

"ランタイム" メニュー内の "全てのランタイムをリセット"
は割り当てられた管理下の全仮想マシン
に対して実施されます。これは仮想マシンに不都合があった場合，たとえばシステムファイルを誤って上書きしてしまった場合など，に有効です。 colaboratory は計算資源を消費するこのような事態に制約を課しています。このような事態が発生したら，しばらく待ってから再試行してください。

- バグをみつけました/質問があります。問い合わせ先を教えてください (I found a bug or have a question, who do I contact?)

colaboratory を開いて，「ヘルプ」メニューから「フィードバックを送る」を選んでください

