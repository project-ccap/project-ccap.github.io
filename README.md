# project-ccap.github.io

https://project-ccap.github.io/
<br />

## ページの更新方法

- git, GitHubの設定を行っていることが前提です．  

```bash
$ git clone https://project-ccap.github.io.git
$ cd project-ccap.githuh.io.git
$ git checkout -b 適当なブランチ名

編集作業を行う

$ git commit -m '編集した内容' -a
$ git push -u origin ブランチ名

GitHub上で pull request を作成する
```

- mergeは管理者が行います．merge後はローカルで以下を実行してください．

```bash
$ git checkout master
$ git pull
$ git branch -d ブランチ名
```

## GitHubの設定・使い方

- [Getting started with GitHub](https://help.github.com/en/github/getting-started-with-github)  
- [SSHによるGitHubへのアクセス](https://help.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh)  
- [pull requestの作成](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests)

## localでページの見た目を確認するための手順

- 環境により異なると思いますのでその都度調べてください．以下はmacOS Catalinaの場合です．  
- 最初にやっておくこと．  

command line toolsのインストール

```bash
$ cd project-ccap.github.io.git
$ gem install bundler
$ bundle init
$ vim Gemfile (他のテキストエディアでも可)
Gemfileを開いて，一番下に以下の行を追記する
gem "github-pages", group: :jekyll_plugins

$ bundle install
```

- 以下を実行するとローカルでサーバが立ち上がるので，ブラウザでhttp://localhost:4000/ にアクセスする．  

```bash
$ cd project-ccap.github.io.git
$ bundle exec jekyll serve
```
