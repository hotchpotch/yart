#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import gc
import sys
import time

import datasets
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# モデルリストとそれぞれに適したバッチサイズ（BS）のマッピング
MODEL_BS_PAIRS = [
    {"model": "hotchpotch/japanese-reranker-tiny-v2", "batch_size": 4096 * 4},
    {"model": "hotchpotch/japanese-reranker-xsmall-v2", "batch_size": 4096 * 4},
    {
        "model": "hotchpotch/japanese-reranker-cross-encoder-xsmall-v1",
        "batch_size": 4096 * 4,
    },
    {
        "model": "hotchpotch/japanese-reranker-cross-encoder-small-v1",
        "batch_size": 4096,
    },
    {"model": "hotchpotch/japanese-reranker-cross-encoder-base-v1", "batch_size": 2048},
    {
        "model": "hotchpotch/japanese-reranker-cross-encoder-large-v1",
        "batch_size": 1024,
    },
    {"model": "hotchpotch/japanese-bge-reranker-v2-m3-v1", "batch_size": 1024},
    {"model": "BAAI/bge-reranker-v2-m3", "batch_size": 1024},
    {"model": "cl-nagoya/ruri-v3-reranker-310m", "batch_size": 2048},
]


def benchmark_with_dataset(model_name, batch_size=1024, num_samples=None):
    """実際のデータセットを使用してベンチマークを実行する関数

    Args:
        model_name (str): ベンチマークするモデルの名前
        batch_size (int): バッチサイズ
        num_samples (int, optional): データセットから使用するサンプル数。Noneの場合は全件使用
    """
    print(f"データセットを使用したベンチマーク開始: {model_name}")

    # デバイスの設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用デバイス: {device}")

    # モデルとトークナイザーをロード
    print(f"モデルとトークナイザーをロード中: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # GPUが利用可能な場合は半精度で実行
    if device == "cuda":
        model.half()
        print("半精度（FP16）で実行します")

    # データセットのロード
    try:
        print("データセットをロード中...")

        # 代替データを準備（データセットのロードに失敗した場合に使用）
        queries = []
        passages = []

        # 簡単なテストデータを作成
        test_queries = [
            "日本の首都はどこですか？",
            "富士山の高さは？",
            "日本で一番長い川は？",
            "京都の有名な観光地は？",
            "日本の国花は？",
        ]

        test_passages = [
            "東京は日本の首都です。政治、経済、文化の中心地として機能しています。",
            "富士山の高さは3,776メートルで、日本で最も高い山です。",
            "信濃川（しなのがわ）は、日本で最も長い川で、全長367kmです。",
            "京都の有名な観光地には、金閣寺、清水寺、伏見稲荷大社などがあります。",
            "日本の国花は桜（サクラ）と菊（キク）の2つとされています。",
        ]

        # 指定されたサンプル数または全件のデータを生成
        sample_count = num_samples if num_samples is not None else None

        # データセットのロードを試みる
        try:
            # データセットをロード
            print("cl-nagoya/ruri-v3-dataset-ftデータセットをロード中...")
            ds = datasets.load_dataset(
                "cl-nagoya/ruri-v3-dataset-ft", "auto-wiki-qa-nemotron", split="train"
            )

            # データセットの種類を確認
            print(f"データセットの種類: {type(ds).__name__}")

            # データセットがイテレーション可能かどうかを確認
            is_iterable = hasattr(ds, "__iter__")
            print(f"イテレーション可能: {is_iterable}")

            # データセットがインデックスでアクセス可能かどうかを確認
            has_getitem = hasattr(ds, "__getitem__")
            print(f"インデックスアクセス可能: {has_getitem}")

            # データセットの処理
            count = 0

            # イテレーション可能な場合
            if is_iterable:
                print("イテレーションによるデータ取得を試みます")
                try:
                    for item in ds:
                        # データ項目のキーを確認
                        if count == 0:
                            if isinstance(item, dict):
                                print(f"データ項目のキー: {list(item.keys())}")
                            else:
                                print(f"データ項目の型: {type(item).__name__}")

                        # データセットの構造に応じてキーを調整
                        if isinstance(item, dict):
                            if "anc" in item and "pos" in item:
                                queries.append(str(item["anc"]))
                                passages.append(str(item["pos"]))
                            elif "query" in item and "passage" in item:
                                queries.append(str(item["query"]))
                                passages.append(str(item["passage"]))
                            elif "question" in item and "answer" in item:
                                queries.append(str(item["question"]))
                                passages.append(str(item["answer"]))

                        count += 1

                        # サンプル数が指定されていて、その数に達したら終了
                        if sample_count is not None and count >= sample_count:
                            break
                except Exception as e:
                    print(f"イテレーション中にエラーが発生しました: {e}")
        except Exception as e:
            print(f"データセットのイテレーションに失敗しました: {e}")

        # データが取得できなかった場合は、テストデータを繰り返し使用
        if len(queries) == 0:
            print(
                "データセットからデータを取得できませんでした。テストデータを使用します。"
            )
            # sample_countがNoneの場合は1万件のテストデータを生成
            test_sample_count = 10000 if sample_count is None else sample_count
            for _ in range(test_sample_count):
                idx = _ % len(test_queries)
                queries.append(test_queries[idx])
                passages.append(test_passages[idx])

    except Exception as e:
        print(f"データセットのロードに失敗しました: {e}")
        # 代替データの作成
        print("代替データを使用します")
        queries = ["日本の首都はどこですか？"] * 1000
        passages = ["東京は日本の首都です。"] * 1000

    # サンプル数の指定があれば制限、なければ全件使用
    if num_samples is not None:
        print(f"{model_name}: {num_samples}サンプルのみ使用")
        queries = queries[:num_samples]
        passages = passages[:num_samples]
    else:
        print(f"{model_name}: 全サンプル使用 ({len(queries)}件)")

    # ペア作成 - 文字列として明示的に作成
    queries = [str(q) for q in queries]
    passages = [str(p) for p in passages]
    total_samples = len(queries)

    # まず全データをトークナイズ
    print(f"{model_name}: トークナイズ開始...")
    tokenize_start_time = time.time()

    # トークナイズ処理
    # バッチごとにトークナイズして、メモリ効率を向上
    tokenized_inputs = []
    tokenize_batch_size = 1000  # トークナイズのバッチサイズ

    for i in tqdm(range(0, total_samples, tokenize_batch_size), desc="トークナイズ中"):
        batch_end = min(i + tokenize_batch_size, total_samples)
        batch_queries = queries[i:batch_end]
        batch_passages = passages[i:batch_end]

        # トークナイズ - 正しい形式でクエリとパッセージのペアを渡す
        inputs = tokenizer(
            batch_queries,
            batch_passages,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        tokenized_inputs.append(inputs)

    tokenize_end_time = time.time()
    tokenize_time = tokenize_end_time - tokenize_start_time
    print(f"{model_name}: トークナイズ完了 (時間: {tokenize_time:.4f}秒)")

    # ウォームアップ実行（小さいバッチで）
    print(f"{model_name}: ウォームアップ実行...")
    warmup_size = min(10, total_samples)
    warmup_inputs = tokenizer(
        queries[:warmup_size],
        passages[:warmup_size],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    warmup_inputs = {k: v.to(device) for k, v in warmup_inputs.items()}
    with torch.no_grad():
        model(**warmup_inputs)

    # 本番ベンチマーク - モデル推論のみ
    print(f"{model_name}: バッチ処理によるベンチマーク開始")
    start_time = time.time()

    try:
        # 効率的なバッチ処理のためのループ
        effective_batch_size = min(batch_size, total_samples)

        # 進捗バーの設定
        progress_bar = tqdm(total=total_samples, desc=f"{model_name} 処理中")

        # トークナイズされたバッチごとに処理
        processed_samples = 0

        for tokenized_batch in tokenized_inputs:
            batch_size = tokenized_batch["input_ids"].size(0)

            # さらに小さなバッチに分割して処理
            for mini_start in range(0, batch_size, effective_batch_size):
                mini_end = min(mini_start + effective_batch_size, batch_size)

                # ミニバッチを抽出
                mini_batch = {
                    k: v[mini_start:mini_end].to(device)
                    for k, v in tokenized_batch.items()
                }

                # 推論実行
                with torch.no_grad():
                    _ = model(**mini_batch)

                # 進捗バーを更新
                mini_batch_size = mini_end - mini_start
                progress_bar.update(mini_batch_size)
                processed_samples += mini_batch_size

        progress_bar.close()
    except Exception as e:
        print(f"ベンチマーク実行中にエラーが発生しました: {e}")

    end_time = time.time()
    inference_time = end_time - start_time
    total_time = tokenize_time + inference_time

    # メモリ解放
    del model, tokenizer, tokenized_inputs
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(
        f"完了: {model_name}, トークナイズ時間: {tokenize_time:.4f}秒, 推論時間: {inference_time:.4f}秒, 合計時間: {total_time:.4f}秒 (データセット: {total_samples}ペア)"
    )
    return {
        "model_name": model_name,
        "batch_size": effective_batch_size,
        "tokenize_time": tokenize_time,
        "inference_time": inference_time,
        "total_time": total_time,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="transformersを使用した日本語リランカーモデルのベンチマーク"
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        default=None,
        help="データセットから使用するサンプル数（指定なしの場合は全件使用）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="結果を出力するCSVファイル（指定なしの場合は標準出力のみ）",
    )
    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        help="ベンチマークするモデル名（指定なしの場合は全モデル実行）",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="詳細な実行情報を表示"
    )

    args = parser.parse_args()

    # 引数からnum_samplesを取得
    num_samples = args.num_samples
    output_file = args.output
    specified_models = args.models

    # データセットでのベンチマーク結果を格納するリスト
    results = []

    # モデル指定があれば、該当するモデルだけをフィルタリング
    model_pairs_to_run = MODEL_BS_PAIRS
    if specified_models:
        model_pairs_to_run = [
            model_data
            for model_data in MODEL_BS_PAIRS
            if model_data["model"] in specified_models
        ]
        if not model_pairs_to_run:
            print(
                f"エラー: 指定されたモデル {specified_models} は定義されたモデルリストに含まれていません"
            )
            sys.exit(1)
        print(f"指定されたモデルのみ実行: {[m['model'] for m in model_pairs_to_run]}")

    # 各モデルをペアで定義されたバッチサイズでベンチマーク
    for model_data in model_pairs_to_run:
        model_name = model_data["model"]
        batch_size = model_data["batch_size"]

        try:
            # データセットでベンチマーク実行（指定されたサンプル数、モデル固有のバッチサイズ使用）
            print(f"モデル: {model_name}, バッチサイズ: {batch_size}")
            result = benchmark_with_dataset(model_name, batch_size, num_samples)
            result["batch_size"] = batch_size  # バッチサイズも結果に含める
            results.append(result)
        except Exception as e:
            print(f"エラー発生 {model_name}: {str(e)}")
            results.append(
                {
                    "model_name": model_name,
                    "batch_size": batch_size,
                    "tokenize_time": -1,
                    "inference_time": -1,
                    "total_time": -1,
                }
            )

    # 結果をCSV形式で標準出力に表示
    print("model_name,batch_size,tokenize_time,inference_time,total_time")
    for result in results:
        print(
            f"{result['model_name']},{result['batch_size']},{result['tokenize_time']:.6f},{result['inference_time']:.6f},{result['total_time']:.6f}"
        )

    # 出力ファイルが指定されている場合はCSVファイルにも保存
    if output_file:
        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "model_name",
                "batch_size",
                "tokenize_time",
                "inference_time",
                "total_time",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in results:
                writer.writerow(result)
        print(f"ベンチマーク結果が {output_file} に保存されました")
