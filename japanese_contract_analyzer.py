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
import logging
from functools import lru_cache, partial
from concurrent.futures import ThreadPoolExecutor
import asyncio
from typing import Optional, Dict, Any, List, Tuple

# MeCab辞書パスの設定
os.environ['MECAB_PATH'] = '/usr/local/lib/mecab/dic/ipadic'

class JapaneseContractAnalyzer:
    _instance = None
    _initialized = False
    _lock = asyncio.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(JapaneseContractAnalyzer, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, model_path='models/contract_risk_model.pkl', vectorizer_path='models/tfidf_vectorizer.pkl'):
        if self._initialized:
            return
            
        self._model_path = model_path
        self._vectorizer_path = vectorizer_path
        self._tagger = None
        self._tokenizer = None
        self._bert_model = None
        self._model = None
        self._vectorizer = None
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._initialized = True
    
    @property
    def tagger(self):
        if self._tagger is None:
            self._tagger = GenericTagger()
        return self._tagger
    
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
        return self._tokenizer
    
    @property
    def bert_model(self):
        if self._bert_model is None:
            self._bert_model = AutoModelForSequenceClassification.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
        return self._bert_model
    
    @property
    def model(self):
        if self._model is None and os.path.exists(self._model_path):
            self._model = joblib.load(self._model_path)
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def vectorizer(self):
        if self._vectorizer is None and os.path.exists(self._vectorizer_path):
            self._vectorizer = joblib.load(self._vectorizer_path)
        return self._vectorizer

    @vectorizer.setter
    def vectorizer(self, value):
        self._vectorizer = value
    
    @lru_cache(maxsize=1000)
    def preprocess_text(self, text: str) -> str:
        """テキストを前処理し、キャッシュする"""
        if not text or not isinstance(text, str):
            return ""
        tokens = [word.surface for word in self.tagger(text)]
        return ' '.join(tokens)
    
    def extract_features(self, texts):
        # 新しいvectorizerを作成しfit_transform
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(texts)
        return X, vectorizer
    
    def train_model(self, contract_data_path='datasets/raw_contracts/sample_contracts.csv'):
        # データ読み込み
        logging.info(f'Loading training data from {contract_data_path}')
        df = pd.read_csv(contract_data_path)
        logging.info(f'Loaded DataFrame shape: {df.shape}')
        logging.info(f'DataFrame columns: {list(df.columns)}')
        logging.info(f'DataFrame head:\n{df.head()}')
        logging.info(f'Unique risk levels: {df["risk_level"].unique()}')
        
        # リスクレベルのエンコーディング
        risk_level_map = {
            '低リスク': 0,
            '中リスク': 1,
            '高リスク': 2
        }
        df['risk_level_encoded'] = df['risk_level'].map(risk_level_map)
        logging.info(f'Risk level mapping:\n{df["risk_level"].value_counts()}')
        
        # 前処理
        logging.info('Starting text preprocessing')
        df['preprocessed_text'] = df['text'].apply(self.preprocess_text)
        logging.info('Text preprocessing complete')
        
        # 特徴量抽出とモデル学習
        logging.info('Extracting features')
        X, vectorizer = self.extract_features(df['preprocessed_text'])
        logging.info(f'Feature matrix shape: {X.shape}')
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
        joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
        self.vectorizer = vectorizer
        
        # モデル評価
        train_accuracy = self.model.score(X_train, y_train)
        test_accuracy = self.model.score(X_test, y_test)
        
        print(f'Training Accuracy: {train_accuracy:.4f}')
        print(f'Test Accuracy: {test_accuracy:.4f}')
        
        return test_accuracy
    
    async def analyze_contract_async(self, contract_text: str) -> Dict[str, Any]:
        """非同期で契約書を分析する"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(self.analyze_contract, contract_text)
        )
    
    def analyze_contract(self, contract_text: str) -> Dict[str, Any]:
        """同期版の契約書分析"""
        logger = logging.getLogger('contract_analyzer')
        
        try:
            if self.model is None:
                logger.info('Model is None. Training model...')
                self.train_model()
            if self.vectorizer is None:
                logger.error('Vectorizer is None. Please train the model first.')
                raise RuntimeError('Vectorizer is not loaded. モデルを再学習してください。')
            
            logger.info(f'Contract text length: {len(contract_text)} characters')
            
            # 前処理
            preprocessed_text = self.preprocess_text(contract_text)
            
            # 特徴量抽出と予測
            X = self.vectorizer.transform([preprocessed_text])
            risk_prediction = self.model.predict(X)[0]
            risk_prob = float(self.model.predict_proba(X)[0].max())
            
            # リスク詳細の生成（非同期で実行）
            risk_details = self._generate_risk_explanation(contract_text, risk_prediction)
            
            # リスクレベルのマッピング
            risk_level_map = {
                0: '低リスク',
                1: '中リスク',
                2: '高リスク'
            }
            
            result = {
                'risk_level': risk_level_map.get(risk_prediction, '不明'),
                'risk_score': risk_prob,
                'explanation': risk_details
            }
            
            return result
            
        except Exception as e:
            logger.error(f'Error in analyze_contract: {str(e)}', exc_info=True)
            raise
    
    @lru_cache(maxsize=1000)
    def _generate_risk_explanation(self, text: str, risk_level: int) -> str:
        """リスク説明を生成し、キャッシュする"""
        risk_level_map = {
            0: '低リスク',
            1: '中リスク',
            2: '高リスク'
        }
        
        risk_level_str = risk_level_map.get(risk_level, '不明')
        
        # キーワードマッチングの最適化
        risk_keywords = {
            '高リスク': ['違約金', '損害責償', '即時解除', '機密情報違反', '契約違反'],
            '中リスク': ['契約期間', '報酬', '義務', '条件変更', '制限事項'],
            '低リスク': ['協力', '誠実', '善管注意義務', '相互理解', '和解']
        }
        
        # 対象のリスクレベルのキーワードのみをチェック
        target_keywords = risk_keywords.get(risk_level_str, [])
        detected_keywords = [kw for kw in target_keywords if kw in text]
        
        if detected_keywords:
            return f"{risk_level_str}リスクの可能性。検出されたキーワード：{', '.join(detected_keywords)}"
        return f"{risk_level_str}リスクと判断されます。詳細なキーワードは検出されませんでした。"
