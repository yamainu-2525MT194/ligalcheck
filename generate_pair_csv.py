import os
import csv
from pathlib import Path
from PyPDF2 import PdfReader

PDF_DIR = Path("datasets/raw_contracts/pdfs")
ANNOTATION_DIR = Path("datasets/raw_contracts/annotations")
OUT_CSV = Path("datasets/processed/contract_pairs.csv")

pairs = []

for pdf_file in PDF_DIR.glob("*.pdf"):
    base = pdf_file.stem  # e.g. 'A.docx'
    txt_file = ANNOTATION_DIR / f"{base}.txt"
    if not txt_file.exists():
        print(f"[WARN] No annotation for {pdf_file.name}")
        continue
    # Extract text from PDF
    try:
        with open(pdf_file, "rb") as f:
            reader = PdfReader(f)
            contract_text = " ".join([p.extract_text() or '' for p in reader.pages])
    except Exception as e:
        print(f"[ERROR] Failed to extract text from {pdf_file.name}: {e}")
        continue
    # Read legal check result
    with open(txt_file, "r", encoding="utf-8") as f:
        legal_check_result = f.read().strip()
    pairs.append({
        "contract_text": contract_text,
        "legal_check_result": legal_check_result
    })

# Write to CSV
os.makedirs(OUT_CSV.parent, exist_ok=True)
with open(OUT_CSV, "w", newline='', encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["contract_text", "legal_check_result"])
    writer.writeheader()
    writer.writerows(pairs)

print(f"[INFO] Wrote {len(pairs)} pairs to {OUT_CSV}")
