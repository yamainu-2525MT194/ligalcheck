import os
import logging
import re
import asyncio
from typing import Dict, Any, List, Optional
import openai
from load_env import ensure_env_loaded

# 環境変数をロード
ensure_env_loaded()

class HybridContractAnalyzer:
    """
    OpenAIのLLMをベースとし、ルールベースの分析を組み合わせたハイブリッド契約書分析クラス
    """
    _instance = None
    _initialized = False
    _lock = asyncio.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(HybridContractAnalyzer, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        logging.info("HybridContractAnalyzer: 初期化")
        
        # OpenAI APIキーの確認
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            logging.error("OpenAI APIキーが設定されていません。環境変数OPENAI_API_KEYを設定してください。")
        else:
            openai.api_key = self.api_key
            
        # リスク分析用のキーワードと重み付け
        self._risk_keywords = {
            'high': {
                'keywords': ['違約金', '損害賠償', '即時解除', '機密情報違反', '契約違反', 
                          '訴訟', '罰則', '重大な不履行', '解約金', '損失補償'],
                'weight': 3.0
            },
            'medium': {
                'keywords': ['契約期間', '報酬', '義務', '条件変更', '制限事項', 
                           '通知義務', '解約予告', '責任範囲', '支払遅延', '知的財産'],
                'weight': 1.5
            },
            'low': {
                'keywords': ['協力', '誠実', '善管注意義務', '相互理解', '和解', 
                          '協議', '通知', '連絡', '開示'],
                'weight': 0.5
            }
        }
        
        # 共通契約書条項パターン
        self._common_patterns = {
            '秘密保持': ['秘密', '機密', '開示', '守秘義務'],
            '契約期間': ['期間', '契約期間', '有効期間'],
            '支払条件': ['支払', '報酬', '料金', '費用'],
            '解約条件': ['解約', '解除', '終了'],
            '知的財産': ['知的財産', '特許', '著作権']
        }
        
        self._initialized = True
        
    async def analyze_contract_async(self, contract_text: str) -> Dict[str, Any]:
        """
        契約書を非同期分析し、OpenAI LLMの分析結果とルールベース分析を組み合わせた結果を返す
        
        Args:
            contract_text (str): 分析対象の契約書テキスト
            
        Returns:
            Dict[str, Any]: 分析結果を含む辞書
        """
        if not contract_text or len(contract_text) < 50:
            return {
                'success': False,
                'error': True,
                'error_message': "契約書テキストが短すぎます。有効な契約書を提供してください。",
                'raw_openai_response': "",
                'problems': [],
                'risks': [],
                'suggestions': [],
                'summary': "",
                'risk_level': 0,
                'risk_score': 0.0,
                'explanation': "分析できませんでした"
            }
        
        # OpenAI APIを使用して分析
        try:
            # まずOpenAI LLMによる分析を実行
            llm_analysis = await self._analyze_with_openai_async(contract_text)
            
            # ルールベースの分析を実行
            rule_based_results = self._rule_based_analysis(contract_text)
            
            # 結果を組み合わせる
            combined_results = self._combine_analysis_results(llm_analysis, rule_based_results)
            
            # リスク評価を行う
            risk_info = await self._advanced_risk_assessment(contract_text, combined_results)
            
            return {
                'success': True,
                'raw_openai_response': llm_analysis.get('raw_openai_response', ''),
                'problems': combined_results.get('problems', []),
                'risks': combined_results.get('risks', []),
                'suggestions': combined_results.get('suggestions', []),
                'summary': combined_results.get('summary', ''),
                'risk_level': risk_info.get('risk_level', 0),
                'risk_score': risk_info.get('risk_score', 0.0),
                'explanation': risk_info.get('explanation', '')
            }
            
        except Exception as e:
            logging.error(f"契約書分析中にエラーが発生しました: {str(e)}")
            return {
                'success': False,
                'error': True,
                'error_message': f"分析中にエラーが発生しました: {str(e)}",
                'raw_openai_response': "",
                'problems': [],
                'risks': [],
                'suggestions': [],
                'summary': "",
                'risk_level': 0,
                'risk_score': 0.0,
                'explanation': "分析エラー"
            }
            
    async def _analyze_with_openai_async(self, contract_text: str) -> Dict[str, Any]:
        """
        OpenAI APIを使用して契約書を分析する
        
        Args:
            contract_text: 契約書テキスト
            
        Returns:
            分析結果を含む辞書
        """
        if not self.api_key:
            return {
                'success': False,
                'error': True,
                'error_message': "OpenAI APIキーが設定されていません",
                'raw_openai_response': ""
            }
        
        # テキストが長すぎる場合は切り詰める
        max_tokens = 8000  # gpt-4の入力上限を考慮
        if len(contract_text) > max_tokens * 2:  # 大雑把な推定で文字数からトークン数を計算
            contract_text = contract_text[:max_tokens * 2]  # 前半部分のみを使用
            
        prompt = f"""
あなたは優秀な法務部長です。以下の日本語の契約書テキストを分析し、法的観点から詳細なメモを作成してください。

以下の形式で契約書の分析を行い、それぞれのセクションを明確に分けて回答してください。

# 契約書の概要
[50文字程度で契約の内容を要約]

# 問題点
- [問題点 1]
- [問題点 2]
- [問題点 3]

# リスク
- [リスク 1]
- [リスク 2]
- [リスク 3]

# 改善提案
- [改善提案 1]
- [改善提案 2]
- [改善提案 3]

# 総合評価
[契約書全体についての総合的な評価、リスクレベルを「低」「中」「高」で評価し、その理由を説明]

契約書:
{contract_text}
"""

        try:
            response = await openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "あなたは優秀な法務部長で、契約書の詳細な分析とリスク評価を行います。日本の契約関連法規についての専門知識を持ち、法的に絣密な評価を行うことができます。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # 低い温度で正確な分析を優先
                max_tokens=2000
            )
            
            raw_response = response.choices[0].message.content
            parsed_data = self._parse_openai_response(raw_response)
            
            return {
                'success': True,
                'raw_openai_response': raw_response,
                **parsed_data
            }
            
        except Exception as e:
            logging.error(f"OpenAI API呼び出し中にエラーが発生しました: {str(e)}")
            return {
                'success': False,
                'error': True,
                'error_message': f"OpenAI APIエラー: {str(e)}",
                'raw_openai_response': f"APIエラー: {str(e)}"
            }
            
    def _parse_openai_response(self, raw_text: str) -> Dict[str, Any]:
        """
        OpenAIからの生のテキスト応答を構造化された辞書に解析する
        """
        if not raw_text:
            return {
                'problems': [],
                'risks': [],
                'suggestions': [],
                'summary': ''
            }
        
        # 分析結果の初期化
        result = {
            'problems': [],
            'risks': [],
            'suggestions': [],
            'summary': ''
        }
        
        try:
            # 概要の抽出
            summary_match = re.search(r'#\s*契約書の概要\s*\n([^#]+)', raw_text, re.DOTALL)
            if summary_match:
                result['summary'] = summary_match.group(1).strip()
            
            # 問題点の抽出
            problems_match = re.search(r'#\s*問題点\s*\n([^#]+)', raw_text, re.DOTALL)
            if problems_match:
                problems_text = problems_match.group(1).strip()
                # 箇条書きを抽出
                problems = re.findall(r'-\s*(.+)(?:\n|$)', problems_text)
                result['problems'] = [p.strip() for p in problems if p.strip()]
            
            # リスクの抽出
            risks_match = re.search(r'#\s*リスク\s*\n([^#]+)', raw_text, re.DOTALL)
            if risks_match:
                risks_text = risks_match.group(1).strip()
                risks = re.findall(r'-\s*(.+)(?:\n|$)', risks_text)
                result['risks'] = [r.strip() for r in risks if r.strip()]
            
            # 改善提案の抽出
            suggestions_match = re.search(r'#\s*改善提案\s*\n([^#]+)', raw_text, re.DOTALL)
            if suggestions_match:
                suggestions_text = suggestions_match.group(1).strip()
                suggestions = re.findall(r'-\s*(.+)(?:\n|$)', suggestions_text)
                result['suggestions'] = [s.strip() for s in suggestions if s.strip()]
            
            # 総合評価の抽出して概要に追加
            evaluation_match = re.search(r'#\s*総合評価\s*\n([^#]+)', raw_text, re.DOTALL)
            if evaluation_match:
                evaluation_text = evaluation_match.group(1).strip()
                if result['summary']:
                    result['summary'] += "\n\n総合評価:\n" + evaluation_text
                else:
                    result['summary'] = evaluation_text
                    
            return result
            
        except Exception as e:
            logging.error(f"OpenAI応答の解析中にエラーが発生しました: {str(e)}")
            return {
                'problems': [],
                'risks': [],
                'suggestions': [],
                'summary': f"[解析エラー: {str(e)}]"
            }
            
    def _rule_based_analysis(self, contract_text: str) -> Dict[str, List[str]]:
        """
        ルールベースの契約書分析を行う
        """
        additional_results = {
            'problems': [],
            'risks': [],
            'suggestions': []
        }
        
        # 共通条項のチェック
        for clause_type, keywords in self._common_patterns.items():
            clause_found = False
            for keyword in keywords:
                if keyword in contract_text:
                    clause_found = True
                    break
            
            # 条項が見つからない場合の処理
            if not clause_found:
                if clause_type == '秘密保持':
                    additional_results['problems'].append(f"秘密保持条項が見つかりません。情報保護のために追加が必要です。")
                    additional_results['suggestions'].append(f"秘密保持条項を追加し、機密情報の取り扱いと罰則を明確にしてください。")
                elif clause_type == '契約期間':
                    additional_results['problems'].append(f"契約期間が明確に定義されていません。")
                    additional_results['suggestions'].append(f"契約期間と更新条件を明確に記載してください。")
                elif clause_type == '解約条件':
                    additional_results['problems'].append(f"解約条件が明確に定義されていません。")
                    additional_results['suggestions'].append(f"解約条件と解約手続きを明確に記載してください。")
        
        # リスクキーワードのチェック
        for risk_level, data in self._risk_keywords.items():
            for keyword in data['keywords']:
                if keyword in contract_text:
                    if risk_level == 'high':
                        additional_results['risks'].append(f"高リスク要素: '{keyword}' が含まれています。")
                    elif risk_level == 'medium' and contract_text.count(keyword) >= 2:  # 中リスクは複数回出現する場合のみ追加
                        additional_results['risks'].append(f"中程度のリスク要素: '{keyword}' が複数回出現します。")
        
        return additional_results
        
    def _combine_analysis_results(self, llm_results: Dict[str, Any], rule_results: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        OpenAI LLMの分析結果とルールベースの分析結果を統合する
        """
        combined = {
            'problems': [],
            'risks': [],
            'suggestions': [],
            'summary': llm_results.get('summary', '')
        }
        
        # LLM結果をベースにする
        if 'success' in llm_results and llm_results['success']:
            combined['problems'] = llm_results.get('problems', [])
            combined['risks'] = llm_results.get('risks', [])
            combined['suggestions'] = llm_results.get('suggestions', [])
        
        # ルールベースの結果を追加
        for key in ['problems', 'risks', 'suggestions']:
            # 重複を避けるために既存のアイテムを確認
            existing_items = set([item.lower() for item in combined[key]])
            
            for item in rule_results.get(key, []):
                # 既存の題目と類似していない場合のみ追加
                similar_found = False
                for existing in existing_items:
                    # 簡易的な類似度チェック: 共通のキーワードが多いかどうか
                    item_words = set(item.lower().split())
                    existing_words = set(existing.split())
                    common_words = item_words.intersection(existing_words)
                    
                    # 40%以上の単語が共通していれば類似と判断
                    if len(common_words) / max(len(item_words), len(existing_words)) > 0.4:
                        similar_found = True
                        break
                
                if not similar_found:
                    combined[key].append(f"[ルールベース分析] {item}")
                    existing_items.add(item.lower())
        
        return combined
        
    async def _advanced_risk_assessment(self, contract_text: str, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        契約書の高度なリスク評価を行う
        """
        # 初期化
        risk_score = 0.0
        risk_counts = {'high': 0, 'medium': 0, 'low': 0}
        explanation = ""
        
        # キーワードベースのスコアリング
        for risk_level, data in self._risk_keywords.items():
            for keyword in data['keywords']:
                occurrences = contract_text.count(keyword)
                if occurrences > 0:
                    # 出現数に応じて加算、出現数が多いほど影響大、ある程度で頂打ち
                    risk_score += min(occurrences, 3) * data['weight']
                    risk_counts[risk_level] += 1
        
        # 分析結果からのリスク評価
        risk_mentions = len(analysis_results.get('risks', []))
        problem_mentions = len(analysis_results.get('problems', []))
        
        # 問題点とリスクの数に応じてスコアを調整
        risk_score += risk_mentions * 2.0  # リスクは大きな影響
        risk_score += problem_mentions * 1.0  # 問題点も考慮
        
        # リスクレベルを判定 (0-10のスケール)
        risk_level = 0
        if risk_score > 15:  # 非常に高いリスク
            risk_level = 3
            explanation = "非常に高いリスクが検出されました。重大な問題が複数あります。髙度な法務レビューと修正が必要です。"
        elif risk_score > 8:  # 高いリスク
            risk_level = 2
            explanation = "高いリスクが検出されました。複数の問題があり、契約書の見直しが必要です。"
        elif risk_score > 3:  # 中程度のリスク
            risk_level = 1
            explanation = "中程度のリスクが検出されました。いくつかの問題がありますが、比較的簡単に修正可能です。"
        else:  # 低いリスク
            risk_level = 0
            explanation = "リスクは低いです。重大な問題は検出されませんでした。"
            
        # 総合評価からのリスク判定
        if 'summary' in analysis_results and analysis_results['summary']:
            if '高いリスク' in analysis_results['summary'] or '高リスク' in analysis_results['summary']:
                risk_level = max(risk_level, 2)  # 最低でも高リスク
                if risk_level < 2:
                    explanation = "総合評価で高リスクと判定されました。" + explanation
            elif '中程度のリスク' in analysis_results['summary'] or '中リスク' in analysis_results['summary']:
                risk_level = max(risk_level, 1)  # 最低でも中リスク
        
        return {
            'risk_level': risk_level,
            'risk_score': round(risk_score, 2),
            'explanation': explanation,
            'risk_counts': risk_counts
        }
        
    # 簡易同期ラッパー
    def analyze_contract(self, contract_text: str) -> Dict[str, Any]:
        """
        契約書を同期分析するラッパーメソッド
        """
        # モジュールを最初にインポートしてスコープの問題を回避
        import nest_asyncio
        import asyncio
        
        try:
            # nest_asyncioを使ってネストしたイベントループを可能にする
            nest_asyncio.apply()
            
            # イベントループの取得または作成
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            # ループが閉じられているか確認
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            return loop.run_until_complete(self.analyze_contract_async(contract_text))
            
        except Exception as e:
            # 以上の方法が失敗した場合は、別の方法を試す
            logging.warning(f"同期実行に失敗しました: {str(e)}")
            try:
                # 新しい方法で実行
                return asyncio.run(self.analyze_contract_async(contract_text))
            except RuntimeError as re:
                if "This event loop is already running" in str(re):
                    # 最終手段として非同期関数を直接実行
                    nest_asyncio.apply()
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    result = new_loop.run_until_complete(self.analyze_contract_async(contract_text))
                    return result
                else:
                    raise re
            except Exception as e2:
                logging.error(f"契約書分析中にエラーが発生: {str(e2)}")
                return {
                    'success': False,
                    'error': True,
                    'error_message': f"契約書分析中にエラーが発生しました: {str(e2)}",
                    'raw_data_content': f"エラー: {str(e2)}",
                    'problems': [],
                    'risks': [],
                    'suggestions': [],
                    'summary': "分析中にエラーが発生しました",
                    'risk_level': 0,
                    'risk_score': 0.0,
                    'raw_openai_response': f"エラー発生: {str(e2)}"
                }
