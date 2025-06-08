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
import openai
import csv
from load_env import ensure_env_loaded

ensure_env_loaded()

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
    
    def analyze_contract(self, contract_text: str) -> dict:
        """
        OpenAI GPTを使って契約書本文からリーガルチェック結果を生成する。
        常にraw_openai_responseを含む辞書を返すように変更。
        """
        # Ensure necessary imports are available (some were method-local)
        import logging
        import json # For potential future JSON parsing from OpenAI, not used in this version directly for parsing OpenAI response
        import traceback
        import os
        import csv
        import openai # Ensure openai is imported

        logger = logging.getLogger('contract_analyzer')

        response_content_raw = None # To store raw text from OpenAI
        # Default value for raw_openai_response in case of early critical failure
        raw_openai_response_for_return = "分析プロセス開始前にエラーが発生しました。詳細はログを確認してください。"

        logger.info("=" * 50)
        logger.info(f"契約書分析開始: テキスト長 {len(contract_text)} 文字")

        try:
            pairs_csv = "datasets/processed/contract_pairs.csv"
            examples = []
            try:
                with open(pairs_csv, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row["contract_text"].strip() and row["legal_check_result"].strip():
                            examples.append(row)
                logger.info(f"Few-shotサンプル: {len(examples)}件")
            except FileNotFoundError:
                logger.warning(f"Few-shotサンプルファイルが見つかりません: {pairs_csv}")
            except Exception as e:
                logger.error(f"Few-shotサンプル読み込みエラー: {str(e)}")

            messages = [
                {"role": "system", "content": "あなたは契約書を分析し、リーガルチェック結果を提供する法務専門家です。以下の形式で回答してください：\n\n1. 問題箇所：契約書の具体的な条項や文言を引用し、問題のある箇所を明示してください\n2. リスク内容：各問題箇所がもたらす具体的なリスクや法的問題点を説明してください\n3. 改善提案：問題箇所をどのように修正すべきか、具体的な代替文言や条項を提案してください\n4. 総合評価：契約書全体のリスクレベルと主な懸念事項をまとめてください\n\n回答は箇条書きで明確に、かつ実務的に役立つ内容にしてください。"}
            ]
            if len(examples) >= 2:
                for i in range(min(2, len(examples))): # Use min to prevent index error if fewer than 2 examples loaded
                    messages.append({"role": "user", "content": examples[i]["contract_text"][:1000]})
                    messages.append({"role": "assistant", "content": examples[i]["legal_check_result"][:1000]})
            
            messages.append({"role": "user", "content": contract_text[:3000]})
            logger.info(f"入力テキスト先頭100文字: {contract_text[:100].replace(chr(10), ' ')}")
            
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.error("OPENAI_API_KEYが設定されていません")
                raw_openai_response_for_return = "APIキーが設定されていないため分析できません。環境変数 OPENAI_API_KEY を確認してください。"
                return {
                    "legal_check_result": "APIキー設定エラー", 
                    "error": "API key not found",
                    "risk_level": None, "risk_score": 0.0, "explanation": "APIキーが設定されていません。",
                    "summary": "設定エラー", "problems": [], "risks": [], "suggestions": [],
                    "raw_openai_response": raw_openai_response_for_return
                }

            logger.info(f"OpenAI API呼び出し開始 (APIキー先頭4文字: {api_key[:4] if api_key else 'None'}...)")
            client = openai.OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=1500,
                temperature=0.1,
                presence_penalty=0.1,
                frequency_penalty=0.1
            )
            
            response_content_raw = response.choices[0].message.content.strip()
            raw_openai_response_for_return = response_content_raw # Successfully got raw response
            logger.info(f"OpenAI API応答先頭100文字: {response_content_raw[:100].replace(chr(10), ' ')}...")
            
            risk_level, risk_score, explanation = self._assess_risk_level(response_content_raw)
            logger.info(f"リスク判定結果: レベル={risk_level}, スコア={risk_score}")

            parsed_elements = self._parse_openai_response(response_content_raw)

            return {
                "legal_check_result": response_content_raw, # Keep for now, might be used by other parts
                "risk_level": risk_level,
                "risk_score": float(risk_score),
                "explanation": explanation,
                "summary": parsed_elements.get("summary", "概要の抽出/解析に失敗しました。元データを確認してください。"),
                "problems": parsed_elements.get("problems", []),
                "risks": parsed_elements.get("risks", []),
                "suggestions": parsed_elements.get("suggestions", []),
                "raw_openai_response": raw_openai_response_for_return,
                "error": None # Explicitly set error to None on success
            }

        except openai.APIError as e:
            error_msg = f"OpenAI APIエラーが発生しました: {str(e)}\n詳細: {traceback.format_exc()}"
            logger.error(error_msg)
            raw_openai_response_for_return = error_msg # Provide detailed API error as raw response
            return {
                "legal_check_result": f"OpenAI APIエラー: {str(e)}", 
                "error": str(e),
                "risk_level": None, "risk_score": 0.0, "explanation": f"OpenAI APIとの通信に失敗しました: {str(e)}",
                "summary": "API通信エラー", "problems": [], "risks": [], "suggestions": [],
                "raw_openai_response": raw_openai_response_for_return
            }
        
        except Exception as e:
            error_msg = f"契約書分析中に予期せぬエラーが発生しました: {str(e)}\n詳細: {traceback.format_exc()}"
            logger.error(error_msg)
            
            if response_content_raw: # If raw content was obtained before this unrelated error
                 raw_openai_response_for_return = response_content_raw + f"\n\n--- 後続処理でのエラー ---\n{error_msg}"
            else: # No raw content, error happened before or during API call but wasn't an APIError
                 raw_openai_response_for_return = error_msg

            return {
                "legal_check_result": f"予期せぬエラー: {str(e)}", 
                "error": str(e),
                "risk_level": None, "risk_score": 0.0, "explanation": f"分析処理中に予期せぬエラーが発生しました: {str(e)}",
                "summary": "処理エラー", "problems": [], "risks": [], "suggestions": [],
                "raw_openai_response": raw_openai_response_for_return
            }

    def _parse_openai_response(self, raw_text: str) -> dict:
        """
        OpenAIからの生のテキスト応答を構造化された辞書に解析する（現時点ではプレースホルダー）。
        実際のプロジェクトでは、プロンプトで指定した形式に基づいて、
        正規表現やより高度なテキスト処理技術を使用してこの部分を実装する必要があります。
        """
        logger = logging.getLogger('contract_analyzer') # Ensure logger is accessible
        summary = "概要は元データから直接ご確認ください。"
        problems = []
        risks = []
        suggestions = []

        try:
            # This is a very basic placeholder. A robust implementation is needed for production.
            # Example: Use regex to find sections like "1. 問題箇所：", "2. リスク内容：" etc.
            # For now, we'll just indicate that parsing needs to be done from raw_text.

            if "問題箇所：" in raw_text or "リスク内容：" in raw_text or "改善提案：" in raw_text or "総合評価：" in raw_text:
                problems.append("構造化された情報の抽出は未実装です。元データをご参照ください。")
            else:
                # If no markers are found, it might be a general statement or an error message from OpenAI
                summary = raw_text[:500] + ("... (詳細は元データ)" if len(raw_text) > 500 else "") 
                problems.append("応答が期待した形式ではありません。元データをご確認ください。")

        except Exception as e:
            logger.error(f"_parse_openai_response でエラー: {str(e)}\n{traceback.format_exc()}")
            summary = "応答の解析中にエラーが発生しました。元データを確認してください。"
            problems = ["応答の解析中にエラーが発生しました。元データを確認してください。"]

        return {
            "summary": summary,
            "problems": problems,
            "risks": risks, # Placeholder, to be populated by parsing raw_text
            "suggestions": suggestions # Placeholder, to be populated by parsing raw_text
        }

    
    def _assess_risk_level(self, analysis_text: str) -> tuple:
        """契約書の分析結果からリスクレベル、リスクスコア、説明を判定する"""
        # リスクキーワードの定義
        risk_keywords = {
            '高リスク': ['違約金', '損害賠償', '即時解除', '機密情報違反', '契約違反', '訴訟', '罰則', '重大な不履行'],
            '中リスク': ['契約期間', '報酬', '義務', '条件変更', '制限事項', '通知義務', '解約予告', '責任範囲'],
            '低リスク': ['協力', '誠実', '善管注意義務', '相互理解', '和解', '協議']
        }
        
        # 各リスクレベルのキーワードの出現回数をカウント
        counts = {level: 0 for level in risk_keywords.keys()}
        detected_keywords = {level: [] for level in risk_keywords.keys()}
        
        for level, keywords in risk_keywords.items():
            for keyword in keywords:
                if keyword in analysis_text:
                    counts[level] += 1
                    detected_keywords[level].append(keyword)
        
        # リスクレベルの判定
        if counts['高リスク'] > 0:
            risk_level = 2  # 高リスク
            risk_score = 0.7 + min(counts['高リスク'] * 0.05, 0.3)  # 0.7-1.0
            level_str = '高リスク'
        elif counts['中リスク'] > 1:  # 中リスクが2つ以上
            risk_level = 1  # 中リスク
            risk_score = 0.4 + min(counts['中リスク'] * 0.05, 0.3)  # 0.4-0.7
            level_str = '中リスク'
        elif counts['中リスク'] > 0 or counts['低リスク'] > 2:  # 中リスクが1つか低リスクが3つ以上
            risk_level = 1  # 中リスク
            risk_score = 0.3 + min((counts['中リスク'] + counts['低リスク'] * 0.3) * 0.05, 0.1)  # 0.3-0.4
            level_str = '中リスク'
        elif counts['低リスク'] > 0:  # 低リスクが1つ以上
            risk_level = 0  # 低リスク
            risk_score = 0.1 + min(counts['低リスク'] * 0.05, 0.2)  # 0.1-0.3
            level_str = '低リスク'
        else:  # キーワードなし
            # 分析結果の内容から判断
            if '注意' in analysis_text or '確認' in analysis_text:
                risk_level = 1  # 中リスク
                risk_score = 0.3
                level_str = '中リスク'
            else:
                risk_level = 0  # 低リスク
                risk_score = 0.1
                level_str = '低リスク'
        
        # 説明文の生成
        explanation = f"{level_str}と判断されます。"
        
        # 検出されたキーワードがあれば追加
        for level in ['高リスク', '中リスク', '低リスク']:
            if detected_keywords[level]:
                explanation += f" {level}キーワード: {', '.join(detected_keywords[level])}"
        
        return risk_level, risk_score, explanation
    
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
