---
title: "Roelofs (2019) Anomia cueing, WEAVER++ のシミュレーションプログラムまとめ"
author: Shin Asakawa
---


# WEAVER++ (Roelofs, 2019) Anomia Cueing のシミュレーションプログラムまとめ

1. WEAVER++ シミュレーションの本体は， https://project-ccap.github.io/ から辿れる colab ファイル
[WEVER++ デモ 2020-1205 更新 Reolofs(2019) Anomia cueing](https://colab.research.google.com/github/project-ccap/project-ccap.github.io/blob/master/notebooks/2020ccap_Roelofs2019_Anomia_cueing_demo.ipynb) である。
2. 本体は `5. シミュレーションの実施` の項 `main()` を実行することでシミュレーションが実行される。
その直下には，グラフを描いたり，元論文の出力結果がある。
3. その下 `追加のシミュレーション` を数値を変えて実行することで，理解を深めることができる
4. `main()` は 2 重ループになっていて，最外ループは，BEFORE と AFTER という 2 回の繰り返しである。
5. BEFORE では LEX_RATE 分だけ結合間係数を減じる。従って，損傷患者のシミュレーションという意味である
6. AFTER は treatment 後なので 1.0/LEX_RATE しているので，元に戻している。治療後でもあるし，健常者のシミュレーションでもある
7. 2 重ループの内側 `compute_prob_functions()` が WEAVER++ 本体の実体である。この実体を 4 条件繰り返している
8. `compute_prob_functions()` を 4 回繰り返しているのは，4 つの音韻手がかり条件である。このままプライミング実験のシミュレーションにも使えそうです。
    * 単語全体を音韻手がかりとして与える条件 `WHOLE`, 
    * 語頭音を手がかりとして与える条件 `INITIAL`,
    * 語尾尾を手がかりとして与える条件 `FINISH`,
    * 音だが言語ではない条件 `NOISE`
9. 結果の表示は，`NOISE` 条件から各条件がどの程度改善したかを示している。
10. 結果は ms 単位で，正解単語 `CAT` と発話し終えるまでの累積確率が十分大きくなるまでの時間を示している。


11. `compute_prob_functions()` 内部では，0 から 1000 まで `STEP_SIZE` 刻みで繰り返す。
12. 繰り返された内部では，以下の 2 つの関数 (処理) を逐次繰り返し呼び出している。
	1. `compute_hazard_rate()`: syllabification がどこまで進んだかの状態を更新する。/k/ ---> /ae/ ---> /t/ と順番に状態が変化することを表現
	2. `update_network()`: Dell や Levelt, Roelofs らの論文で必ず最初に出てくる数式 ($A(j,t)=(1-d)(A(j,t-1) + \sum w A(i,t-1)$) に基づくネットワークの更新
13. ネットワークの更新において更新される情報は以下の通り
	* `M_node_act`: Morpheme 形態層の活性化状態
	* `P_node_act`: Phoneme 音韻層の活性化状態
	* `S_node_act`: Syllable 音素層の活性化状態
14. 上の 3 つの層 `M_node_act`, `P_node_act`, `S_node_act` に影響を与えるものは以下の $3\times2=6$ つ
	* `input_M`: Morpheme 層への入力
	* `input_P`: Phoneme 層への入力
	* `input_S`: Syllable 層への入力
	* `MP_con`: Morpheme 層と Phoneme 層との間の結合係数行列
	* `PS_con`: Phoneme 層と Syllable 層との間の結合係数行列


<center>
<img src="https://raw.githubusercontent.com/project-ccap/project-ccap.github.io/master/figures/2019Roelofs_Aphasiology_fig1.png" style="width:66%"><br/>
<div align="left" style="width:66%">
Roelofs (2019) Phonological cueing of word finding in aphasia: insights from simulations of immediate and treatment effects, APHASIOLOGY
https://doi.org/10.1080/02687038.2019.1686748, Fig. 1<br/>
</div>
</center>

<center>
<img src="https://raw.githubusercontent.com/project-ccap/project-ccap.github.io/master/figures/2019Roelofs_Aphasiology_fig2.png" style="width:66%"><br/>
<div align="left" style="width:66%">
Roelofs (2019) Phonological cueing of word finding in aphasia: insights from simulations of immediate and treatment effects, APHASIOLOGY
https://doi.org/10.1080/02687038.2019.1686748, Fig. 2<br/>
 </div>
</center>

<center>
<img src="https://raw.githubusercontent.com/project-ccap/project-ccap.github.io/master/figures/2019Roelofs_Aphasiology_fig3.png" style="width:66%"><br/>
<div align="left" style="width:94%">
<font size="+2" color="teal">
右上の図での数値は，<br/>
Whole,Untreated:295, Whole,Treated: 268, 
Initial,Untreated:295, Intial,Treated:268,
Final,Untreated:297, Final,Treated:268,
Noise,Untreated:316, Noise,Treated,291<br/>
</font><br/>
From Roelofs (2019) Phonological cueing of word finding in aphasia: insights from simulations of immediate and treatment effects, APHASIOLOGY
https://doi.org/10.1080/02687038.2019.1686748, Fig. 3<br/>
 </div>
</center>
