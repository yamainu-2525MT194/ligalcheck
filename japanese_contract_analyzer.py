import re
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import fugashi
from fugashi import GenericTagger
import os

# MeCab辞書パスの設定
os.environ['MECAB_PATH'] = '/usr/local/lib/mecab/dic/ipadic'

class JapaneseContractAnalyzer:
    def __init__(self, model_path='models/contract_risk_model.pkl'):
        self.tagger = GenericTagger()
        self.tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
        self.bert_model = AutoModelForSequenceClassification.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
        
        try:
            self.model = joblib.load(model_path)
        except FileNotFoundError:
            self.model = None
    
    def preprocess_text(self, text):
        # 形態素解析による前処理
        tokens = [word.surface for word in self.tagger(text)]
        return ' '.join(tokens)
    
    def extract_features(self, texts):
        vectorizer = TfidfVectorizer(max_features=5000)
        return vectorizer.fit_transform(texts)
    
    def train_model(self, contract_data_path='datasets/raw_contracts/sample_contracts.csv'):
        # データ読み込み
        df = pd.read_csv(contract_data_path)
        
        # リスクレベルのエンコーディング
        risk_level_map = {
            '低リスク': 0,
            '中リスク': 1,
            '高リスク': 2
        }
        df['risk_level_encoded'] = df['risk_level'].map(risk_level_map)
        
        # 前処理
        df['preprocessed_text'] = df['text'].apply(self.preprocess_text)
        
        # 特徴量抽出とモデル学習
        X = self.extract_features(df['preprocessed_text'])
        y = df['risk_level_encoded']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = MLPClassifier(
            hidden_layer_sizes=(100, 50), 
            max_iter=1000,  # 収束回数を増やす
            random_state=42,
            verbose=True  # 学習遍程の詳細出力
        )
        self.model.fit(X_train, y_train)
        
        # モデル保存
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, 'models/contract_risk_model.pkl')
        
        # モデル評価
        train_accuracy = self.model.score(X_train, y_train)
        test_accuracy = self.model.score(X_test, y_test)
        
        print(f'Training Accuracy: {train_accuracy:.4f}')
        print(f'Test Accuracy: {test_accuracy:.4f}')
        
        return test_accuracy
    
    def analyze_contract(self, contract_text):
        if self.model is None:
            self.train_model()
        
        # 前処理
        preprocessed_text = self.preprocess_text(contract_text)
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform([preprocessed_text])
        
        # リスクレベル予測
        risk_prediction = self.model.predict(X)[0]
        
        # BERTによる追加分析
        inputs = self.tokenizer(contract_text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        
        # キーワードベースの詳細分析
        risk_details = self._generate_risk_explanation(contract_text, risk_prediction)
        
        return {
            'risk_level': risk_prediction,
            'risk_score': self.model.predict_proba(X)[0].max(),
            'explanation': risk_details
        }
    
    def _generate_risk_explanation(self, text, risk_level):
        risk_keywords = {
            '高リスク': ['違約金', '損害賠償', '即時解除', '機密情報違反', '契約違反'],
            '中リスク': ['契約期間', '報酬', '義務', '条件変更', '制限事項'],
            '低リスク': ['協力', '誠実', '善管注意義務', '相互理解', '和解']
        }
        
        detected_keywords = []
        for level, keywords in risk_keywords.items():
            if level.startswith(risk_level):
                for keyword in keywords:
                    if keyword in text:
                        detected_keywords.append(keyword)
        
        if detected_keywords:
            return f"{risk_level}リスクの可能性。検出されたキーワード：{', '.join(detected_keywords)}"
        else:
            return f"{risk_level}リスクと判断されます。詳細なキーワードは検出されませんでした。"
