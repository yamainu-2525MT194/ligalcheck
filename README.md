# 契約書リーガルチェックアプリ (Legal Contract Review App)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 概要 (Overview)
このアプリは、PDF契約書をアップロードし、機械学習とAIを使用してリーガルリスクを分析するWeb アプリケーションです。

### 主な機能 (Features)
- 📄 PDF契約書アップロード
- 🤖 機械学習によるリスク評価
- 🧠 AIによる詳細な契約書分析
- 📊 リスクレベルの可視化
- 🖥️ Web機械学習トレーニングプラットフォーム

## 主な機能 (Features)
- PDFアップロード
- 機械学習による初期リスク評価
- AIによる詳細な契約書分析
- リスクレベルの可視化
- Webベースの機械学習トレーニングプラットフォーム

## 主な機能 (Detailed Features)
- データアップロード
- モデル設定
- モデルトレーニング
- モデル評価
- モデル保存

## 必要条件 (Requirements)
- Python 3.8+
- OpenAI API Key

## インストール手順 (Installation)
1. リポジトリをクローン
2. 仮想環境を作成
```bash
python3 -m venv venv
source venv/bin/activate
```

3. 依存関係をインストール
```bash
pip install -r requirements.txt
```

4. `.env`ファイルを作成し、OpenAI APIキーを設定
```
OPENAI_API_KEY=your_openai_api_key_here
```

5. アプリケーションを起動
```bash
python app.py
```

## アクセス方法 (Access)
- 契約書分析: `http://localhost:5000`
- 機械学習トレーニング: `http://localhost:5000/ml_trainer`

## データ学習方法 (How to Train)
1. `ml_trainer`ページにアクセス
2. CSVまたはExcelファイルをアップロード
3. モデルとパラメータを選択
4. トレーニングを開始
5. 評価結果を確認

## 注意事項 (Caution)
- このアプリは法的助言の代替にはなりません
- 機密情報を扱う際は十分注意してください
- モデルの精度は学習データに依存します
