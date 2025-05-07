#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
import sys
import time

import datasets
import torch
from sentence_transformers import CrossEncoder

# モデルリストとそれぞれに適したバッチサイズ（BS）のマッピング
MODEL_BS_PAIRS = [
    {"model": "hotchpotch/japanese-reranker-tiny-v2", "batch_size": 4096},
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


def benchmark_with_dataset(model_name, batch_size=1024):
    """実際のデータセットを使用してベンチマークを実行する関数"""
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

    # 全件使用
    pairs = [(query, passage) for query, passage in zip(queries, passages)]

    # ウォームアップ実行
    small_batch = pairs[: min(10, len(pairs))]
    model.predict(
        small_batch,
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

    print(
        f"完了: {model_name}, 処理時間: {elapsed_time:.4f}秒 (データセット: {len(pairs)}ペア)"
    )
    return {
        "model_name": model_name,
        "time_seconds": elapsed_time,
    }


def main():
    # データセットでのベンチマーク結果を格納するリスト
    results = []

    # 各モデルをペアで定義されたバッチサイズでベンチマーク
    for model_data in MODEL_BS_PAIRS:
        model_name = model_data["model"]
        batch_size = model_data["batch_size"]

        try:
            # データセットでベンチマーク実行（全件、モデル固有のバッチサイズ使用）
            print(f"モデル: {model_name}, バッチサイズ: {batch_size}")
            result = benchmark_with_dataset(model_name, batch_size)
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


if __name__ == "__main__":
    main()
