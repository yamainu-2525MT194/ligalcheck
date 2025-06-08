import os
import pytest
import json
import logging
from flask import Flask
from app import app as flask_app
from japanese_contract_analyzer import JapaneseContractAnalyzer

# ロガー設定
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('test_debug')

@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client

def test_real_contract_upload(client):
    test_pdf = os.path.join(os.path.dirname(__file__), '../uploads/test_contract.pdf')
    assert os.path.exists(test_pdf), 'テスト用契約書が存在しません'
    
    # 直接JapaneseContractAnalyzerを使用してテスト
    print('\n--- 直接JapaneseContractAnalyzerを使用したテスト ---')
    from pdfplumber import open as pdf_open
    with pdf_open(test_pdf) as pdf:
        text = '\n'.join([p.extract_text() or '' for p in pdf.pages])
    
    analyzer = JapaneseContractAnalyzer()
    direct_result = analyzer.analyze_contract(text)
    print('直接分析結果:', json.dumps(direct_result, ensure_ascii=False, indent=2))
    
    # APIを使用したテスト
    print('\n--- APIを使用したテスト ---')
    with open(test_pdf, 'rb') as f:
        data = {'file': (f, 'test_contract.pdf')}
        response = client.post('/upload', data=data, content_type='multipart/form-data')
    
    assert response.status_code == 200, f'アップロードAPIが失敗: {response.data}'
    json_data = response.get_json()
    print('API応答:', json.dumps(json_data, ensure_ascii=False, indent=2))
    
    # 詳細な検証
    assert 'analysis' in json_data, '分析結果が含まれていません'
    if json_data['analysis'] == '':
        print('警告: 分析結果が空です')
        print('直接分析結果のキー:', list(direct_result.keys()))
    
    return json_data
