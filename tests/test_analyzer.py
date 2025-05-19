import pytest
import os
import joblib
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# テスト対象のモジュールをインポート
from japanese_contract_analyzer import JapaneseContractAnalyzer

class TestJapaneseContractAnalyzer:
    @pytest.fixture(autouse=True)
    def setup_method(self, temp_dir, test_config):
        # テスト用のモデルとベクトルライザーのパスを設定
        self.model_path = test_config["MODEL_PATH"]
        self.vectorizer_path = test_config["VECTORIZER_PATH"]
        
        # テスト用のディレクトリを作成
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # ダミーのモデルとベクトルライザーを作成
        self.create_dummy_model()
        
        # アナライザーを初期化
        self.analyzer = JapaneseContractAnalyzer(
            model_path=self.model_path,
            vectorizer_path=self.vectorizer_path
        )
    
    def create_dummy_model(self):
        """テスト用のダミーモデルとベクトルライザーを作成"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # ダミーのトレーニングデータ
        texts = [
            "これはテスト文書です。",
            "契約書のテスト文書です。",
            "違約金の条件が含まれています。"
        ]
        
        # ダミーのラベル
        labels = [0, 1, 2]  # 0: 低リスク, 1: 中リスク, 2: 高リスク
        
        # ベクトルライザーをフィット
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)
        
        # モデルをトレーニング
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, labels)
        
        # モデルとベクトルライザーを保存
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(model, self.model_path)
        joblib.dump(vectorizer, self.vectorizer_path)
    
    def test_analyze_contract(self):
        """契約書の分析テスト"""
        # テスト用の契約書テキスト
        test_text = "この契約書には違約金の条件が含まれています。"
        
        # 分析を実行
        result = self.analyzer.analyze_contract(test_text)
        
        # 結果の検証
        assert isinstance(result, dict)
        assert 'risk_level' in result
        assert 'risk_score' in result
        assert 'explanation' in result
        assert result['risk_level'] in ['低リスク', '中リスク', '高リスク']
        assert 0 <= result['risk_score'] <= 1.0
    
    @pytest.mark.asyncio
    async def test_analyze_contract_async(self):
        """非同期での契約書分析テスト"""
        # テスト用の契約書テキスト
        test_text = "この契約書には違約金の条件が含まれています。"
        
        # モックの設定
        with patch.object(self.analyzer, 'analyze_contract') as mock_analyze:
            mock_analyze.return_value = {
                'risk_level': '高リスク',
                'risk_score': 0.85,
                'explanation': 'テスト用の説明'
            }
            
            # 非同期で分析を実行
            result = await self.analyzer.analyze_contract_async(test_text)
            
            # 結果の検証
            assert isinstance(result, dict)
            assert 'risk_level' in result
            assert 'risk_score' in result
            assert 'explanation' in result
            
            # モックが正しく呼び出されたか確認
            mock_analyze.assert_called_once_with(test_text)
    
    def test_generate_risk_explanation(self):
        """リスク説明の生成テスト"""
        # テスト用のテキストとリスクレベル
        test_text = "この契約書には違約金の条件が含まれています。"
        risk_level = 2  # 高リスク
        
        # 説明を生成
        explanation = self.analyzer._generate_risk_explanation(test_text, risk_level)
        
        # 結果の検証
        assert isinstance(explanation, str)
        assert "高リスク" in explanation
        assert "違約金" in explanation
