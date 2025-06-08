import os
import pytest
import json
import io
from flask import Flask
from app import app as flask_app

@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client

def test_contract_upload(client):
    """標準モデル（OpenAI）を使用した契約書アップロードのテスト"""
    sample_pdf = os.path.join(os.path.dirname(__file__), '../uploads/sample_contract.pdf')
    assert os.path.exists(sample_pdf), 'サンプル契約書が存在しません'
    with open(sample_pdf, 'rb') as f:
        data = {'file': (f, 'sample_contract.pdf')}
        response = client.post('/upload', data=data, content_type='multipart/form-data')
    assert response.status_code == 200, f'アップロードAPIが失敗: {response.data}'
    json_data = response.get_json()
    assert 'analysis' in json_data, '分析結果が含まれていません'
    assert 'risk_level' in json_data, 'リスクレベルが含まれていません'
    print('標準モデルAPI応答:', json_data)

def test_contract_upload_with_hybrid_model(client):
    """ハイブリッドモデルを使用した契約書アップロードのテスト"""
    sample_pdf = os.path.join(os.path.dirname(__file__), '../uploads/sample_contract.pdf')
    assert os.path.exists(sample_pdf), 'サンプル契約書が存在しません'
    with open(sample_pdf, 'rb') as f:
        # model_idパラメータを追加してハイブリッドモデルを指定
        data = {
            'file': (f, 'sample_contract.pdf'),
            'model_id': 'hybrid'
        }
        response = client.post('/upload', data=data, content_type='multipart/form-data')
    assert response.status_code == 200, f'ハイブリッドモデルアップロードAPIが失敗: {response.data}'
    json_data = response.get_json()
    assert 'analysis' in json_data, '分析結果が含まれていません'
    assert 'risk_level' in json_data, 'リスクレベルが含まれていません'
    assert json_data.get('model_id') == 'hybrid', 'ハイブリッドモデルIDが返されていません'
    print('ハイブリッドモデルAPI応答:', json_data)

def test_contract_analysis_with_text_hybrid(client):
    """テキストコンテンツを使ってハイブリッドモデル分析をテスト"""
    # サンプルの契約書テキスト
    contract_text = """
    株式会社ウィンドサーフ（以下「甲」という）と山本太郎（以下「乙」という）は、
    以下のとおり契約を締結する。
    
    第1条（目的）
    本契約は、甲が乙に対してソフトウェア開発業務を委託することに関する基本的事項を定めることを目的とする。
    
    第2条（秘密保持）
    乙は、本契約に基づき知り得た甲の技術上、営業上の一切の秘密情報を第三者に漏洩してはならない。
    
    第10条（契約解除）
    甲または乙は、相手方が本契約に違反した場合、何らの催告なく本契約を解除することができる。
    """
    
    # JSONデータとしてリクエストを送信
    response = client.post('/api/analyze_contract',
                          data=json.dumps({
                              'contract_text': contract_text,
                              'model_id': 'hybrid'
                          }),
                          content_type='application/json')
    
    assert response.status_code == 200, f'ハイブリッドモデルAPI呼び出しが失敗: {response.data}'
    json_data = response.get_json()
    assert json_data.get('success') == True, 'APIが成功を返していません'
    assert 'raw_data_content' in json_data, '元データが含まれていません'
    print('ハイブリッドモデル分析API応答:', json_data)
