import os
import logging
from functools import lru_cache
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import Dict, Any

class CustomT5ContractAnalyzer:
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(CustomT5ContractAnalyzer, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, model_path='models/t5_contract_analyzer'):
        if self._initialized:
            return
            
        self._model_path = model_path
        self._model = None
        self._tokenizer = None
        self._max_input_length = 1024
        self._max_output_length = 1024
        self._initialized = True
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logging.info(f"CustomT5ContractAnalyzer initialized with model path: {model_path}")
        logging.info(f"Using device: {self._device}")
    
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            try:
                logging.info(f"Loading T5 tokenizer from {self._model_path}")
                if not os.path.exists(self._model_path):
                    logging.error(f"モデルディレクトリが存在しません: {self._model_path}")
                    raise FileNotFoundError(f"モデルディレクトリが見つかりません: {self._model_path}")
                self._tokenizer = T5Tokenizer.from_pretrained(self._model_path)
            except Exception as e:
                logging.error(f"T5トークナイザーのロード中にエラーが発生しました: {str(e)}")
                # デフォルトのトークナイザーを使用（エラーメッセージ用）
                self._tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-base')
        return self._tokenizer
    
    @property
    def model(self):
        if self._model is None:
            try:
                logging.info(f"Loading T5 model from {self._model_path}")
                if not os.path.exists(self._model_path):
                    logging.error(f"モデルディレクトリが存在しません: {self._model_path}")
                    raise FileNotFoundError(f"モデルディレクトリが見つかりません: {self._model_path}")
                self._model = T5ForConditionalGeneration.from_pretrained(self._model_path)
                self._model.to(self._device)
                self._model.eval()  # 推論モード
            except Exception as e:
                logging.error(f"T5モデルのロード中にエラーが発生しました: {str(e)}")
                self._model = None  # モデルをNoneに設定
        return self._model
    
    @lru_cache(maxsize=100)
    def analyze_contract(self, contract_text: str) -> Dict[str, Any]:
        """
        訓練済みT5モデルを使用して契約書を分析し、法的メモを生成する
        
        Args:
            contract_text (str): 分析対象の契約書テキスト
            
        Returns:
            Dict[str, Any]: 分析結果を含む辞書
        """
        # モデルが利用可能かのチェック
        if self.model is None:
            error_message = "カスタムT5モデルがロードできませんでした。モデルファイルが存在しないか破損しています。"
            logging.error(error_message)
            return {
                'success': False,
                'error': True,
                'error_message': error_message,
                'raw_t5_response': "モデルが利用可能でありません",
                'analysis': {
                    'problems': [],
                    'risks': [],
                    'suggestions': [],
                    'summary': "カスタムT5モデルがロードできませんでした。"
                },
                'risk_info': {
                    'risk_level': 0,
                    'risk_score': 0.0,
                    'explanation': "分析できませんでした"
                }
            }

        try:
            logging.info("Starting contract analysis with T5 model")
            
            # 契約書テキストの検証
            if not contract_text or len(contract_text.strip()) < 10:
                error_message = "分析する契約書テキストが短すぎます。"
                logging.error(error_message)
                return {
                    'success': False,
                    'error': True,
                    'error_message': error_message,
                    'raw_t5_response': "テキストが短すぎます",
                    'analysis': {
                        'problems': [],
                        'suggestions': [],
                        'risks': [],
                        'summary': error_message
                    },
                    'risk_info': {
                        'risk_level': 0,
                        'risk_score': 0.0,
                        'explanation': error_message
                    }
                }
            
            # 長すぎる入力は切り詰める
            if len(contract_text) > self._max_input_length * 4:  # トークン数はだいたい文字数の1/4程度
                logging.warning(f"Contract text too long ({len(contract_text)} chars), truncating...")
                contract_text = contract_text[:self._max_input_length * 4]
            
            # トークン化
            inputs = self.tokenizer(
                contract_text, 
                return_tensors="pt", 
                max_length=self._max_input_length,
                truncation=True
            ).to(self._device)
            
            # 推論実行
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=self._max_output_length,
                    num_beams=4,
                    length_penalty=1.0,
                    early_stopping=True
                )
            
            # デコード
            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # テキストから概要を抽出
            summary = ""
            lines = decoded_output.split('\n')
            # 先頭の数行を概要として使用
            if lines and len(lines) > 0:
                summary = "\n".join(lines[:min(2, len(lines))])
                if len(summary) > 200:
                    summary = summary[:200] + "..."
            
            # 問題点、リスク、提案を抽出
            problems = self._extract_problems(decoded_output)
            risks = self._extract_risks(decoded_output)
            suggestions = self._extract_suggestions(decoded_output)
            
            # リスクレベルを評価
            risk_info = self._assess_risk_level(decoded_output)
            
            # 分析結果を整形
            result = {
                'success': True,
                'raw_t5_response': decoded_output,
                'analysis': {
                    'problems': problems,
                    'risks': risks,
                    'suggestions': suggestions,
                    'summary': summary
                },
                'risk_info': risk_info
            }
            
            return result
            
        except Exception as e:
            error_message = f"T5モデル分析中にエラーが発生しました: {str(e)}"
            logging.error(error_message)
            return {
                'success': False,
                'error': True,
                'error_message': error_message,
                'raw_t5_response': error_message,
                'analysis': {
                    'problems': [],
                    'risks': [],
                    'suggestions': [],
                    'summary': '分析中にエラーが発生しました'
                },
                'risk_info': {
                    'risk_level': 0,
                    'risk_score': 0.0,
                    'explanation': error_message
                }
            }
    
    def _extract_problems(self, analysis_text: str) -> list:
        """分析テキストから問題点を抽出する"""
        problems = []
        
        # キーワードやセクションから問題点を抽出
        problem_indicators = [
            "問題点:", "問題点として", "問題があります", "問題事項:", 
            "懐念点:", "懐念事項:", "この点が懐念", "不明確な点"
        ]
        
        # テキストを行ごとに分割
        lines = analysis_text.split('\n')
        
        in_problem_section = False
        current_problem = ""
        
        for line in lines:
            line = line.strip()
            
            # 問題点のセクションを検出
            if any(indicator in line.lower() for indicator in problem_indicators):
                in_problem_section = True
                # 既に問題点があれば追加
                if current_problem:
                    problems.append(current_problem.strip())
                    current_problem = ""
                # セクションタイトルが含まれていない場合は問題点として追加
                potential_problem = line
                for indicator in problem_indicators:
                    potential_problem = potential_problem.replace(indicator, "")
                if potential_problem.strip():
                    current_problem = potential_problem
            # 別のセクションへの移行を検出
            elif in_problem_section and line and ("提案:" in line or "提案事項:" in line or "リスク:" in line):
                if current_problem:
                    problems.append(current_problem.strip())
                in_problem_section = False
                current_problem = ""
            # 問題点セクション内でテキストを収集
            elif in_problem_section and line:
                if line.startswith("-") or line.startswith("*") or line.startswith("•"):
                    if current_problem:
                        problems.append(current_problem.strip())
                    current_problem = line.lstrip("-*• ")
                elif current_problem:
                    current_problem += " " + line
        
        # 最後の問題点を追加
        if current_problem:
            problems.append(current_problem.strip())
            
        # 複数の問題点が抽出できなかった場合は、シンプルなキーワードマッチングも試す
        if len(problems) == 0:
            # 開始/終了キーワードの間のテキストを問題点として出力
            if "問題点" in analysis_text and len(analysis_text) > 10:
                problems = [p.strip() for p in analysis_text.split(",") if p.strip()]
        
        # 空の結果を防ぐためのフォールバック
        if len(problems) == 0 and len(analysis_text) > 10:
            # もし問題点が何も抽出できない場合、最初の数行を返す
            first_paragraph = analysis_text.split('\n\n')[0] if '\n\n' in analysis_text else analysis_text
            if len(first_paragraph) > 200:  # 長すぎる場合は切り詰める
                first_paragraph = first_paragraph[:200] + "..."
            problems = [first_paragraph]
        
        return problems
    
    def _extract_suggestions(self, analysis_text: str) -> list:
        """分析テキストから提案を抽出する"""
        suggestions = []
        
        # キーワードやセクションから提案を抽出
        suggestion_indicators = [
            "提案:", "提案事項:", "改善点:", "改善事項:", 
            "推奨事項:", "推奨します:", "検討すべき点:"
        ]
        
        # テキストを行ごとに分割
        lines = analysis_text.split('\n')
        
        in_suggestion_section = False
        current_suggestion = ""
        
        for line in lines:
            line = line.strip()
            
            # 提案セクションを検出
            if any(indicator in line.lower() for indicator in suggestion_indicators):
                in_suggestion_section = True
                # 既に提案があれば追加
                if current_suggestion:
                    suggestions.append(current_suggestion.strip())
                    current_suggestion = ""
                # セクションタイトルが含まれていない場合は提案として追加
                potential_suggestion = line
                for indicator in suggestion_indicators:
                    potential_suggestion = potential_suggestion.replace(indicator, "")
                if potential_suggestion.strip():
                    current_suggestion = potential_suggestion
            # 別のセクションへの移行を検出
            elif in_suggestion_section and line and ("問題点:" in line or "リスク:" in line):
                if current_suggestion:
                    suggestions.append(current_suggestion.strip())
                in_suggestion_section = False
                current_suggestion = ""
            # 提案セクション内でテキストを収集
            elif in_suggestion_section and line:
                if line.startswith("-") or line.startswith("*") or line.startswith("•"):
                    if current_suggestion:
                        suggestions.append(current_suggestion.strip())
                    current_suggestion = line.lstrip("-*• ")
                elif current_suggestion:
                    current_suggestion += " " + line
        
        # 最後の提案を追加
        if current_suggestion:
            suggestions.append(current_suggestion.strip())
            
        # 複数の提案が抽出できなかった場合は、キーワードを含む文を探す
        if len(suggestions) == 0:
            for line in lines:
                if any(keyword in line.lower() for keyword in ["改善", "修正", "検討", "追加", "明確"]):
                    suggestions.append(line.strip())
        
        return suggestions
    
    def _extract_risks(self, analysis_text: str) -> list:
        """分析テキストからリスクを抽出する"""
        risks = []
        
        # キーワードやセクションからリスクを抽出
        risk_indicators = [
            "リスク:", "法的リスク:", "危険性:", "懐念事項:", 
            "リスクがあります", "懐念点があります", "注意点:", "要注意:"
        ]
        
        # テキストを行ごとに分割
        lines = analysis_text.split('\n')
        
        in_risk_section = False
        current_risk = ""
        
        for line in lines:
            line = line.strip()
            
            # リスクセクションを検出
            if any(indicator in line.lower() for indicator in risk_indicators):
                in_risk_section = True
                # 既にリスクがあれば追加
                if current_risk:
                    risks.append(current_risk.strip())
                    current_risk = ""
                # セクションタイトルが含まれていない場合はリスクとして追加
                potential_risk = line
                for indicator in risk_indicators:
                    potential_risk = potential_risk.replace(indicator, "")
                if potential_risk.strip():
                    current_risk = potential_risk
            # 別のセクションへの移行を検出
            elif in_risk_section and line and ("問題点:" in line or "提案:" in line or "提案事項:" in line):
                if current_risk:
                    risks.append(current_risk.strip())
                in_risk_section = False
                current_risk = ""
            # リスクセクション内でテキストを収集
            elif in_risk_section and line:
                if line.startswith("-") or line.startswith("*") or line.startswith("•"):
                    if current_risk:
                        risks.append(current_risk.strip())
                    current_risk = line.lstrip("-*• ")
                elif current_risk:
                    current_risk += " " + line
        
        # 最後のリスクを追加
        if current_risk:
            risks.append(current_risk.strip())
        
        # キーワードベースのリスク抽出を行う
        risk_keywords = [
            '違約金', '損害賠償', '解除', '機密情報', '違反', '訴訟', '罰則', '不履行',
            '契約期間', '制限', '通知義務', '解約', '責任', '法的', '懐念'
        ]
        
        if len(risks) == 0:
            for line in lines:
                for keyword in risk_keywords:
                    if keyword in line and line not in risks and len(line.strip()) > 5:
                        risks.append(line.strip())
                        break
        
        # リスクが見つからない場合は空配列を返す
        return risks
    
    def _assess_risk_level(self, analysis_text: str) -> Dict[str, Any]:
        """
        分析テキストからリスクレベルを評価する
        リスク評価はT5モデルの出力から、シンプルなキーワード検出で行う
        """
        # リスク関連キーワード
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
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'explanation': explanation
        }
