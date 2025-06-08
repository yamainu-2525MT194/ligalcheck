import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    T5Tokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import torch
import os
import evaluate # For ROUGE score calculation

# グローバル変数または設定可能なパラメータ
MODEL_NAME = "sonoisa/t5-base-japanese"
CSV_FILE_PATH = "compiled_training_data.csv"
MODEL_OUTPUT_DIR = "./models/t5_contract_analyzer"
TRAIN_BATCH_SIZE = 2 # データが少ないため小さく設定
EVAL_BATCH_SIZE = 2
NUM_TRAIN_EPOCHS = 3 # データが少ないため小さく設定
LEARNING_RATE = 5e-5
MAX_SOURCE_LENGTH = 1024 # 契約書の最大トークン長
MAX_TARGET_LENGTH = 512  # メモの最大トークン長

def load_and_preprocess_data(file_path, tokenizer):
    """CSVファイルを読み込み、トークナイズしてデータセットを作成する"""
    df = pd.read_csv(file_path)
    
    # NaNや空のテキストを除外
    df.dropna(subset=['contract_text', 'memo_text'], inplace=True)
    df = df[df['contract_text'].str.strip().astype(bool) & df['memo_text'].str.strip().astype(bool)]

    if len(df) == 0:
        raise ValueError("No valid data found after cleaning. Please check your CSV file.")

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42) # 80% train, 20% validation

    def tokenize_function(examples):
        # T5の場合、入力にプレフィックスを追加することが推奨される場合がある
        # ここではシンプルに契約書テキストを入力とする
        inputs = tokenizer(
            examples["contract_text"],
            max_length=MAX_SOURCE_LENGTH,
            truncation=True,
            padding="max_length" # または "longest"
        )
        # ターゲット（ラベル）のトークナイズ
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["memo_text"],
                max_length=MAX_TARGET_LENGTH,
                truncation=True,
                padding="max_length" # または "longest"
            )
        inputs["labels"] = labels["input_ids"]
        return inputs

    # Pandas DataFrameをHugging Face Datasetに変換
    from datasets import Dataset
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
    tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=val_dataset.column_names)
    
    return tokenized_train_dataset, tokenized_val_dataset

def compute_metrics(eval_preds):
    """評価指標（ROUGEスコア）を計算する"""
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # -100は無視するようにデコード
    # tokenizerはグローバルスコープで定義されている想定
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    # ROUGEスコアの計算
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    return results

def train_model(train_dataset, val_dataset, current_tokenizer):
    """モデルの学習を実行する"""
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # データコレータ
    data_collator = DataCollatorForSeq2Seq(current_tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        evaluation_strategy="epoch", # エポックごとに評価
        save_strategy="epoch",       # エポックごとに保存
        logging_dir=f"{MODEL_OUTPUT_DIR}/logs",
        logging_steps=10, # ログ出力頻度
        load_best_model_at_end=True, # 最後に最良モデルをロード
        metric_for_best_model="eval_loss", # または "rouge1", "rouge2", "rougeL" など
        predict_with_generate=True, # 生成タスクでは必須
        fp16=torch.cuda.is_available(), # GPUが利用可能ならFP16を使用
        # report_to="tensorboard" # TensorBoard連携
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=current_tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if val_dataset and len(val_dataset) > 0 else None, # 検証セットがある場合のみ評価
    )

    trainer.train()
    trainer.save_model(MODEL_OUTPUT_DIR)
    print(f"Model and tokenizer saved to {MODEL_OUTPUT_DIR}")

# グローバルスコープでtokenizerを初期化 (compute_metricsやtrain_modelで使うため)
# この初期化はmainガードの外で行い、compute_metricsからアクセスできるようにする
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

if __name__ == "__main__":
    if not os.path.exists(CSV_FILE_PATH):
        print(f"Error: Training data file not found at {CSV_FILE_PATH}")
        exit(1)
        
    print("Loading and preprocessing data...")
    try:
        train_dataset, val_dataset = load_and_preprocess_data(CSV_FILE_PATH, tokenizer)
        print(f"Training with {len(train_dataset)} samples, validating with {len(val_dataset)} samples.")
    except ValueError as e:
        print(f"Error during data loading: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        # For more detailed debugging, you might want to print the full traceback
        # import traceback
        # traceback.print_exc()
        exit(1)

    if len(train_dataset) == 0:
        print("No training data available after preprocessing. Exiting.")
        exit(1)

    print("Starting model training...")
    # train_modelにグローバルなtokenizerを渡す
    train_model(train_dataset, val_dataset, tokenizer)
    print("Training finished.")
