#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import gc
import sys
import time

import datasets
import torch
from sentence_transformers import CrossEncoder

# モデルリストとそれぞれに適したバッチサイズ（BS）のマッピング
MODEL_BS_PAIRS = [
    {"model": "hotchpotch/japanese-reranker-tiny-v2", "batch_size": 2048},
    {"model": "hotchpotch/japanese-reranker-xsmall-v2", "batch_size": 2048},
    {
        "model": "hotchpotch/japanese-reranker-cross-encoder-xsmall-v1",
        "batch_size": 2048,
    },
    {
        "model": "hotchpotch/japanese-reranker-cross-encoder-small-v1",
        "batch_size": 2048,
    },
    {"model": "hotchpotch/japanese-reranker-cross-encoder-base-v1", "batch_size": 1024},
    {"model": "hotchpotch/japanese-reranker-cross-encoder-large-v1", "batch_size": 512},
    {"model": "hotchpotch/japanese-bge-reranker-v2-m3-v1", "batch_size": 512},
    {"model": "BAAI/bge-reranker-v2-m3", "batch_size": 512},
    {"model": "cl-nagoya/ruri-v3-reranker-310m", "batch_size": 1024},
]


def benchmark_model(model_name, batch_size=1024):
    """指定されたリランカーモデルのベンチマークを実行する関数"""
    print(f"ベンチマーク開始: {model_name}")

    # モデルをロード
    model = CrossEncoder(model_name, max_length=512)

    # GPUが利用可能な場合は半精度で実行
    if model.device == "cuda" or model.device == "mps":
        model.model.half()

    # テスト用データの準備
    query = "感動的な映画について"
    passages = [
        "深いテーマを持ちながらも、観る人の心を揺さぶる名作。登場人物の心情描写が秀逸で、ラストは涙なしでは見られない。",
        "重要なメッセージ性は評価できるが、暗い話が続くので気分が落ち込んでしまった。もう少し明るい要素があればよかった。",
        "どうにもリアリティに欠ける展開が気になった。もっと深みのある人間ドラマが見たかった。",
        "アクションシーンが楽しすぎる。見ていて飽きない。ストーリーはシンプルだが、それが逆に良い。",
    ]
    pairs = [(query, passage) for passage in passages]

    # ウォームアップ実行
    model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
    )

    # 本番ベンチマーク
    start_time = time.time()
    model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=True,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time

    # メモリ解放
    del model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"完了: {model_name}, 処理時間: {elapsed_time:.4f}秒")
    return {
        "model_name": model_name,
        "time_seconds": elapsed_time,
    }


def benchmark_with_dataset(model_name, batch_size=1024, num_samples=None):
    """実際のデータセットを使用してベンチマークを実行する関数

    Args:
        model_name (str): ベンチマークするモデルの名前
        batch_size (int): バッチサイズ
        num_samples (int, optional): データセットから使用するサンプル数。Noneの場合は全件使用
    """
    print(f"データセットを使用したベンチマーク開始: {model_name}")

    # モデルをロード
    model = CrossEncoder(model_name, max_length=512)

    # GPUが利用可能な場合は半精度で実行
    if model.device == "cuda" or model.device == "mps":
        model.model.half()

    # データセットのロード
    ds = datasets.load_dataset(
        "cl-nagoya/ruri-v3-dataset-ft", "auto-wiki-qa-nemotron", split="train"
    )
    queries = ds["anc"]
    passages = ds["pos"]

    # サンプル数の指定があれば制限、なければ全件使用
    if num_samples is not None:
        print(f"{model_name}: {num_samples}サンプルのみ使用")
        queries = queries[:num_samples]
        passages = passages[:num_samples]
    else:
        print(f"{model_name}: 全サンプル使用 ({len(queries)}件)")

    # ペア作成
    pairs = [(query, passage) for query, passage in zip(queries, passages)]

    # トークナイザー事前処理
    print(f"{model_name}: トークナイザー事前処理開始...")
    features = model.tokenize(pairs)
    print(f"{model_name}: トークナイザー事前処理完了")

    # ウォームアップ実行（小さいバッチで）
    small_batch = {k: v[: min(10, len(pairs))] for k, v in features.items()}
    model.model(**small_batch)

    # 推論のみのベンチマーク
    print(f"{model_name}: 推論処理のみのベンチマーク開始")
    start_time = time.time()

    # batchごとに処理
    batch_size = min(batch_size, len(pairs))
    total_samples = len(pairs)

    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        batch_features = {k: v[start_idx:end_idx] for k, v in features.items()}

        # バッチ処理を実行（推論のみ）
        with torch.no_grad():
            model.model(**batch_features)

    end_time = time.time()
    inference_time = end_time - start_time

    # メモリ解放
    del model, features
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(
        f"完了: {model_name}, 推論処理時間: {inference_time:.4f}秒 (データセット: {len(pairs)}ペア)"
    )
    return {
        "model_name": model_name,
        "batch_size": batch_size,
        "time_seconds": inference_time,
    }


# 不要な関数を削除

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="日本語リランカーモデルのベンチマーク")
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
                    "time_seconds": -1,  # エラーの場合は-1を記録
                }
            )

    # 結果をCSV形式で標準出力に表示
    print("model_name,batch_size,time_seconds")
    for result in results:
        print(
            f"{result['model_name']},{result['batch_size']},{result['time_seconds']:.6f}"
        )

    # 出力ファイルが指定されている場合はCSVファイルにも保存
    if output_file:
        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["model_name", "batch_size", "time_seconds"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in results:
                writer.writerow(result)
        print(f"ベンチマーク結果が {output_file} に保存されました")
