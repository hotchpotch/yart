# yart
Yet Antother Reranker Trainer

このプロジェクトは、Reranker モデルを学習するためTrainer実装です。

LLMがコードを書いて実行を試す場合、以下のように実行します。configファイルは適切なものを選ぶこと。

```
WANDB_MODE=offline DEBUG=1 uv run python yart/run.py --config examples/exp337.yaml
```