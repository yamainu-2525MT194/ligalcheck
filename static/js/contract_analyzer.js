/**
 * 契約書分析システム - フロントエンド機能
 * 複数のモデル（OpenAI API、カスタムT5）を選択して契約書分析を行う
 */

// モデル情報を取得
async function loadAvailableModels() {
    try {
        const response = await fetch('/api/model_info');
        const data = await response.json();
        
        if (data.success) {
            return data.models;
        } else {
            console.error('モデル情報取得エラー:', data.error);
            return [];
        }
    } catch (error) {
        console.error('モデル情報取得中にエラーが発生:', error);
        return [];
    }
}

// モデル選択UIを構築
async function setupModelSelector() {
    const modelSelectContainer = document.getElementById('model-select-container');
    if (!modelSelectContainer) return;
    
    const models = await loadAvailableModels();
    if (models.length === 0) {
        modelSelectContainer.innerHTML = '<div class="alert alert-warning">利用可能なモデルはありません</div>';
        return;
    }
    
    let html = `
        <div class="card mb-4">
            <div class="card-header bg-secondary text-white">
                <i class="bi bi-robot"></i> 分析モデルの選択
            </div>
            <div class="card-body">
                <div class="mb-3">
                    <label for="modelSelect" class="form-label">使用するモデル:</label>
                    <select class="form-select" id="modelSelect">
    `;
    
    models.forEach(model => {
        html += `<option value="${model.id}">${model.name} - ${model.description}</option>`;
    });
    
    html += `
                    </select>
                </div>
                <p class="text-muted small">標準モデル（OpenAI）は一般的な契約書分析に、カスタムT5モデルは独自データで訓練されたモデルです。</p>
            </div>
        </div>
    `;
    
    modelSelectContainer.innerHTML = html;
}

// 契約書分析実行
async function analyzeContract(contractText, modelId = 'standard') {
    try {
        const resultDiv = document.getElementById('result');
        resultDiv.innerHTML = '<div class="alert alert-info">分析中...</div>';
        
        let endpoint, requestData;
        
        if (modelId === 'custom_t5') {
            endpoint = '/api/analyze_contract_with_t5';
            requestData = { contract_text: contractText };
        } else if (modelId === 'hybrid') {
            // ハイブリッドモデル用のエンドポイント（標準APIと同じ）
            endpoint = '/api/analyze_contract';
            requestData = { contract_text: contractText, model_id: 'hybrid' };
        } else {
            // デフォルトは標準モデル（OpenAI）
            endpoint = '/api/analyze_contract';
            requestData = { contract_text: contractText };
        }
        
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        const data = await response.json();
        
        if (!data.success && data.error) {
            resultDiv.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
            return;
        }
        
        // モデルタイプに応じて結果表示を調整
        let analysisHTML;
        
        if (modelId === 'custom_t5') {
            analysisHTML = formatT5Results(data);
        } else if (modelId === 'hybrid') {
            analysisHTML = formatHybridResults(data);
        } else {
            analysisHTML = formatStandardResults(data);
        }
        
        resultDiv.innerHTML = analysisHTML;
    } catch (error) {
        const resultDiv = document.getElementById('result');
        resultDiv.innerHTML = `<div class="alert alert-danger">エラーが発生しました: ${error.message}</div>`;
    }
}

// T5モデル結果のフォーマット
function formatT5Results(data) {
    // バックエンドからの統一された形式に対応
    const riskLevel = data.risk_level || 0;
    
    // リスクレベルに応じたバッジの色を決定
    let riskBadgeClass, riskText;
    
    if (riskLevel === 3) {
        riskBadgeClass = 'bg-danger';
        riskText = '高リスク';
    } else if (riskLevel === 2) {
        riskBadgeClass = 'bg-warning text-dark';
        riskText = '中リスク';
    } else if (riskLevel === 1) {
        riskBadgeClass = 'bg-info text-dark';
        riskText = '低リスク';
    } else {
        riskBadgeClass = 'bg-success';
        riskText = 'リスクなし';
    }

    // 問題点リストの生成
    let problemsHTML = '<ul class="list-group">';
    if (data.problems && data.problems.length > 0) {
        data.problems.forEach(problem => {
            problemsHTML += `<li class="list-group-item list-group-item-warning"><i class="bi bi-exclamation-triangle me-2"></i>${problem}</li>`;
        });
    } else {
        problemsHTML += '<li class="list-group-item">検出された問題はありません。</li>';
    }
    problemsHTML += '</ul>';

    // リスクリストの生成
    let risksHTML = '<ul class="list-group">';
    if (data.risks && data.risks.length > 0) {
        data.risks.forEach(risk => {
            risksHTML += `<li class="list-group-item list-group-item-danger"><i class="bi bi-shield-exclamation me-2"></i>${risk}</li>`;
        });
    } else {
        risksHTML += '<li class="list-group-item">検出されたリスクはありません。</li>';
    }
    risksHTML += '</ul>';

    // 提案リストの生成
    let suggestionsHTML = '<ul class="list-group">';
    if (data.suggestions && data.suggestions.length > 0) {
        data.suggestions.forEach(suggestion => {
            suggestionsHTML += `<li class="list-group-item list-group-item-success"><i class="bi bi-lightbulb me-2"></i>${suggestion}</li>`;
        });
    } else {
        suggestionsHTML += '<li class="list-group-item">提案はありません。</li>';
    }
    suggestionsHTML += '</ul>';
    
    return `
        <div class="card">
            <div class="card-header bg-primary text-white">
                <i class="bi bi-robot me-2"></i><strong>カスタムT5モデル分析結果</strong>
                <span class="badge ${riskBadgeClass} float-end">${riskText}</span>
            </div>
            <div class="card-body">
                <div class="mb-4">
                    <h5><i class="bi bi-exclamation-triangle-fill me-2"></i>リスク評価</h5>
                    <div class="alert alert-secondary">
                        <p><strong>リスクスコア:</strong> ${(data.risk_score * 100).toFixed(1)}%</p>
                        <p><strong>説明:</strong> ${data.explanation || '説明なし'}</p>
                    </div>
                </div>
                
                <div class="mb-4">
                    <h5><i class="bi bi-file-text me-2"></i>概要</h5>
                    <div class="alert alert-light">
                        ${data.summary || '概要なし'}
                    </div>
                </div>
                
                <div class="mb-4">
                    <h5><i class="bi bi-exclamation-circle me-2"></i>問題点</h5>
                    ${problemsHTML}
                </div>
                
                <div class="mb-4">
                    <h5><i class="bi bi-shield-exclamation me-2"></i>リスク</h5>
                    ${risksHTML}
                </div>
                
                <div class="mb-4">
                    <h5><i class="bi bi-lightbulb me-2"></i>提案</h5>
                    ${suggestionsHTML}
                </div>
                
                <div class="mb-4">
                    <h5 class="d-flex justify-content-between">
                        <span><i class="bi bi-code-square me-2"></i>元データ</span>
                        <button class="btn btn-sm btn-outline-secondary" type="button" data-bs-toggle="collapse" data-bs-target="#rawResponseCollapse">
                            表示/非表示
                        </button>
                    </h5>
                    <div class="collapse" id="rawResponseCollapse">
                        <div class="card card-body">
                            <pre>${data.raw_data_content || 'データなし'}</pre>
                        </div>
                    </div>
                </div>
            </div>
            <div class="card-footer text-muted">
                <i class="bi bi-clock"></i> ${new Date().toLocaleString()}
            </div>
        </div>
    `;
}

// 標準モデル結果のフォーマット
function formatStandardResults(data) {
    // 既存のフォーマット処理をコピー
    // リスクレベルに応じたバッジの色を決定
    let riskBadgeClass = 'bg-info';
    let riskText = 'リスク不明';
    
    if (data.risk_level !== null && data.risk_level !== undefined) {
        if (data.risk_level === 3) {
            riskBadgeClass = 'bg-danger';
            riskText = '高リスク';
        } else if (data.risk_level === 2) {
            riskBadgeClass = 'bg-warning text-dark';
            riskText = '中リスク';
        } else if (data.risk_level === 1) {
            riskBadgeClass = 'bg-success';
            riskText = '低リスク';
        } else {
            riskBadgeClass = 'bg-info';
            riskText = 'リスクなし';
        }
    }
    
    return `
        <div class="card">
            <div class="card-header bg-primary text-white">
                <i class="bi bi-cloud me-2"></i><strong>OpenAI API 分析結果</strong>
                <span class="badge ${riskBadgeClass} float-end">${riskText}</span>
            </div>
            <div class="card-body">
                <div id="analysisResultSections">
                    <!-- 元データタブのみを表示（簡素化されたUI） -->
                    <ul class="nav nav-tabs" id="analysisTabs" role="tablist">
                        <li class="nav-item" role="presentation">
                            <button class="nav-link active" id="raw-data-tab" data-bs-toggle="tab" data-bs-target="#raw-data" type="button" role="tab" aria-controls="raw-data" aria-selected="true">
                                <i class="bi bi-file-earmark-text"></i> 元データ
                            </button>
                        </li>
                    </ul>
                    <!-- タブコンテンツ -->
                    <div class="tab-content" id="analysisTabsContent">
                        <!-- 元データ タブ (唯一のタブ) -->
                        <div class="tab-pane fade show active" id="raw-data" role="tabpanel" aria-labelledby="raw-data-tab">
                            <h5 class="mt-3"><i class="bi bi-file-earmark-text-fill"></i> 元データ (OpenAI応答)</h5>
                            <pre id="raw-data-content" class="bg-light p-3 rounded" style="white-space: pre-wrap; word-wrap: break-word;">${data.raw_data_content ? data.raw_data_content.replace(/</g, "&lt;").replace(/>/g, "&gt;") : '元データがありません。'}</pre>
                        </div>
                    </div>
                </div>
            </div>
            <div class="card-footer text-muted">
                <i class="bi bi-clock"></i> ${new Date().toLocaleString()}
            </div>
        </div>
    `;
}

// テキスト入力からの分析 - 契約書テキストを直接入力する場合
function analyzeContractText() {
    const contractText = document.getElementById('contractText').value.trim();
    const modelSelect = document.getElementById('modelSelect');
    const selectedModel = modelSelect ? modelSelect.value : 'standard';
    
    if (!contractText) {
        alert('契約書のテキストを入力してください');
        return;
    }
    
    analyzeContract(contractText, selectedModel);
}

// ハイブリッドモデル結果のフォーマット
function formatHybridResults(data) {
    // リスクレベルに応じたバッジの色を決定
    let riskBadgeClass = 'bg-info';
    let riskText = 'リスク不明';
    
    if (data.risk_level !== null && data.risk_level !== undefined) {
        if (data.risk_level === 3) {
            riskBadgeClass = 'bg-danger';
            riskText = '高リスク';
        } else if (data.risk_level === 2) {
            riskBadgeClass = 'bg-warning text-dark';
            riskText = '中リスク';
        } else if (data.risk_level === 1) {
            riskBadgeClass = 'bg-success';
            riskText = '低リスク';
        } else {
            riskBadgeClass = 'bg-info';
            riskText = 'リスクなし';
        }
    }
    
    return `
        <div class="card">
            <div class="card-header text-white" style="background-color: #6f42c1;">
                <i class="bi bi-stars me-2"></i><strong>ハイブリッド分析結果</strong>
                <span class="badge ${riskBadgeClass} float-end">${riskText}</span>
            </div>
            <div class="card-body">
                <div id="analysisResultSections">
                    <!-- 元データタブのみを表示（簡素化されたUI） -->
                    <ul class="nav nav-tabs" id="analysisTabsHybrid" role="tablist">
                        <li class="nav-item" role="presentation">
                            <button class="nav-link active" id="raw-data-tab-hybrid" data-bs-toggle="tab" data-bs-target="#raw-data-hybrid" type="button" role="tab" aria-controls="raw-data-hybrid" aria-selected="true">
                                <i class="bi bi-file-earmark-text"></i> 元データ
                            </button>
                        </li>
                    </ul>
                    <!-- タブコンテンツ -->
                    <div class="tab-content" id="analysisTabsHybridContent">
                        <!-- 元データ タブ (唯一のタブ) -->
                        <div class="tab-pane fade show active" id="raw-data-hybrid" role="tabpanel" aria-labelledby="raw-data-tab-hybrid">
                            <h5 class="mt-3"><i class="bi bi-file-earmark-text-fill"></i> 元データ (ハイブリッド分析)</h5>
                            <pre id="raw-data-content-hybrid" class="bg-light p-3 rounded" style="white-space: pre-wrap; word-wrap: break-word;">${data.raw_data_content ? data.raw_data_content.replace(/</g, "&lt;").replace(/>/g, "&gt;") : '元データがありません。'}</pre>
                        </div>
                    </div>
                </div>
            </div>
            <div class="card-footer text-muted">
                <i class="bi bi-clock"></i> ${new Date().toLocaleString()}
            </div>
        </div>
    `;
}

// ファイルアップロードからの分析
function uploadContractFile() {
    const fileInput = document.getElementById('fileInput');
    const modelSelect = document.getElementById('modelSelect');
    const selectedModel = modelSelect ? modelSelect.value : 'standard';
    
    if (!fileInput.files.length) {
        alert('契約書ファイル（PDFまたはDOCX）を選択してください');
        return;
    }
    
    const resultDiv = document.getElementById('result');
    resultDiv.textContent = '分析中...';
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('model_id', selectedModel);
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            resultDiv.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
        } else if (selectedModel === 'custom_t5') {
            // カスタムT5モデル用の結果表示
            resultDiv.innerHTML = formatT5Results(data);
        } else {
            // 標準モデル用の結果表示
            resultDiv.innerHTML = formatStandardResults(data);
        }
    })
    .catch(error => {
        resultDiv.innerHTML = `<div class="alert alert-danger">エラー: ${error}</div>`;
    });
}

// ページ読み込み時に初期化
document.addEventListener('DOMContentLoaded', () => {
    setupModelSelector();
});
