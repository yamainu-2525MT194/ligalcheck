from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

def create_sample_contract(filename):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # タイトル
    title = Paragraph("業務委託契約書", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))

    # 契約当事者
    parties = Paragraph("甲: 株式会社テクノロジーソリューションズ", styles['Normal'])
    story.append(parties)
    parties = Paragraph("乙: イノベーション株式会社", styles['Normal'])
    story.append(parties)
    story.append(Spacer(1, 12))

    # 契約目的
    purpose = Paragraph("第1条 (目的)", styles['Heading3'])
    story.append(purpose)
    purpose_text = Paragraph("本契約は、甲が乙に対して、AIを活用した契約書分析サービスの開発業務を委託することを目的とする。", styles['Normal'])
    story.append(purpose_text)
    story.append(Spacer(1, 12))

    # 業務範囲
    scope = Paragraph("第2条 (業務範囲)", styles['Heading3'])
    story.append(scope)
    scope_text = Paragraph("1. 乙は、AIモデルを用いた契約書の自動分析システムを開発する。", styles['Normal'])
    story.append(scope_text)
    scope_text = Paragraph("2. 開発するシステムは、契約書のリスク評価、重要条項の抽出を行うものとする。", styles['Normal'])
    story.append(scope_text)
    story.append(Spacer(1, 12))

    # 契約期間
    period = Paragraph("第3条 (契約期間)", styles['Heading3'])
    story.append(period)
    period_text = Paragraph("本契約の有効期間は、2025年6月1日から2026年5月31日までとする。", styles['Normal'])
    story.append(period_text)
    story.append(Spacer(1, 12))

    # 機密保持
    confidentiality = Paragraph("第4条 (機密保持)", styles['Heading3'])
    story.append(confidentiality)
    confidentiality_text = Paragraph("甲および乙は、本契約に関連して知り得た相手方の機密情報を厳重に管理し、第三者に開示しないものとする。", styles['Normal'])
    story.append(confidentiality_text)
    story.append(Spacer(1, 12))

    # 署名
    signature = Paragraph("2025年5月11日", styles['Normal'])
    story.append(signature)
    story.append(Spacer(1, 24))

    signature_block = Paragraph("甲: 株式会社テクノロジーソリューションズ 代表取締役 山田太郎 (印)", styles['Normal'])
    story.append(signature_block)
    signature_block = Paragraph("乙: イノベーション株式会社 代表取締役 佐藤次郎 (印)", styles['Normal'])
    story.append(signature_block)

    doc.build(story)

# サンプル契約書を生成
create_sample_contract('/Users/yamamotoyuuki/Documents/windsurf_dev/legal_review_app/uploads/sample_contract.pdf')
print("サンプル契約書を生成しました: /Users/yamamotoyuuki/Documents/windsurf_dev/legal_review_app/uploads/sample_contract.pdf")
