"""Portable demo and BERT/SVM/LSTM reproduction entry point (Python 3.10-3.12)."""
import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "examples/comments.csv"
LABELS = ["negative", "neutral", "positive"]
CHINESE = ["负面", "中性", "正面"]
MODEL_ID = "google-bert/bert-base-chinese"
MODEL_REVISION = "84b432f646e4047ce1b5db001d43a348cd3f6bd0"
MODEL_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "model.safetensors"]


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_data(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"content", "sentiment", "split", "topic", "time"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV must contain content,sentiment,split,topic,time")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV is empty")
    seen = {}
    for index, row in enumerate(rows, 2):
        if not row["content"].strip() or not row["topic"].strip():
            raise ValueError(f"Empty content/topic at CSV line {index}")
        if row["sentiment"] not in LABELS or row["split"] not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid sentiment/split at CSV line {index}")
        datetime.strptime(row["time"], "%Y-%m-%d %H:%M:%S")
        key = row["content"].strip()
        if key in seen and seen[key] != row["split"]:
            raise ValueError("Same comment appears in different splits")
        seen[key] = row["split"]
    for split in ("train", "validation", "test"):
        if {r["sentiment"] for r in rows if r["split"] == split} != set(LABELS):
            raise ValueError(f"{split} must contain all three sentiment classes")
    return rows


def rule_predict(text):
    positive = ("很好", "棒", "喜欢", "满意", "推荐", "不错", "优秀", "开心", "支持", "精美", "赞", "顺畅")
    negative = ("差", "失望", "糟糕", "浪费", "不清楚", "混乱", "不满意", "贵", "后悔", "出错", "慢", "生气", "故障", "太久")
    # Avoid counting the positive substring in a simple negation.
    cleaned = text.replace("不满意", "差")
    score = sum(word in cleaned for word in positive) - sum(word in cleaned for word in negative)
    return "positive" if score > 0 else "negative" if score < 0 else "neutral"


def export_dashboard(rows, labels, target, method, lstm_epochs=0):
    if len(rows) != len(labels) or not rows or any(label not in LABELS for label in labels):
        raise ValueError("Every input row must have one valid predicted label")
    target = Path(target).resolve()
    source = ROOT / "可交互的可视化大屏"
    # Never overwrite an existing dashboard or original local results.
    if target.exists():
        raise FileExistsError(f"Output already exists; choose a new --output: {target}")
    target.mkdir(parents=True)
    for name in ("index.html", "dashboard.js"):
        shutil.copyfile(source / name, target / name)
    shutil.copytree(source / "vendor", target / "vendor")
    events = []
    grouped = defaultdict(list)
    for row, label in zip(rows, labels):
        grouped[row["topic"]].append((row, label))
    for index, (topic, records) in enumerate(grouped.items(), 1):
        counts = Counter(label for _, label in records)
        hourly = defaultdict(list)
        for row, label in records:
            stamp = datetime.strptime(row["time"], "%Y-%m-%d %H:%M:%S").replace(minute=0, second=0)
            hourly[stamp].append(LABELS.index(label) - 1)
        # Fill missing hours with neutral values so LSTM steps represent equal durations.
        points = []
        stamp, last = min(hourly), max(hourly)
        if (last - stamp).total_seconds() > 24 * 366 * 3600:
            raise ValueError("Time span exceeds one year; use a smaller input window")
        while stamp <= last:
            values = hourly.get(stamp, [])
            points.append({"time": str(stamp), "sentiment_score": sum(values) / len(values) if values else 0,
                           "comment_count": len(values)})
            stamp += timedelta(hours=1)
        scores = [p["sentiment_score"] for p in points]
        forecast_method = "最近三小时均值衰减（演示基线）"
        average = sum(scores[-3:]) / len(scores[-3:])
        future = [average * 0.85 ** i for i in range(1, 7)]
        if lstm_epochs:
            from lstm_predictor import LSTMPredictor
            predictor = LSTMPredictor()
            if predictor.train(scores, epochs=lstm_epochs, seq_length=12):
                future = predictor.predict(scores, future_steps=6, seq_length=12)
                forecast_method = "LSTM（探索性预测，未进行未来区间回测）"
            else:
                forecast_method += "；时间点不足 13 个，未训练 LSTM"
        keywords = {word: sum(row["content"].count(word) for row, _ in records)
                    for word in ("展览", "服务", "参观", "体验", "讲解", "预约", "时间", "活动", "展厅", "设备")}
        data = {
            "event_info": {"name": topic, "analysis_time": str(last), "total_comments": len(records),
                           "sentiment_summary": f"{method}；{forecast_method}",
                           **{f"{label}_percentage": round(100 * counts[label] / len(records), 1) for label in LABELS}},
            "sentiment_distribution": {"categories": CHINESE, "values": [counts[label] for label in LABELS],
                                       "colors": ["#ff4d4f", "#faad14", "#52c41a"]},
            "time_series": points, "keywords": {k: v for k, v in keywords.items() if v},
            "predictions": [{"time": str(last + timedelta(hours=i)), "sentiment_score": float(score)}
                            for i, score in enumerate(future, 1)]}
        visual, results = f"event-{index}-visual.json", f"event-{index}-results.json"
        write_json(target / visual, data)
        write_json(target / results, [{"comment": row["content"], "timestamp": row["time"],
                                     "sentiment": CHINESE[LABELS.index(label)]} for row, label in records])
        events.append({"id": f"event{index}", "name": topic, "visualFile": visual, "resultFile": results})
    write_json(target / "events.json", {"notice": method, "events": events})
    print(f"Dashboard written to {target}")


def download_model(args):
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    from huggingface_hub import snapshot_download
    target = Path(args.model).resolve()
    source = {"repo_id": MODEL_ID, "revision": MODEL_REVISION}
    marker = target / "download-source.json"
    # Resume only directories created by this command, without touching existing weights.
    if target.exists() and any(target.iterdir()):
        if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != source:
            raise FileExistsError("Model directory belongs to another source; select a new --model directory")
    write_json(marker, source)
    snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=target, allow_patterns=MODEL_FILES,
                      token=False, etag_timeout=30)
    missing = [name for name in MODEL_FILES if not (target / name).is_file()]
    if missing:
        raise RuntimeError(f"Incomplete download: {missing}")
    print(f"Base model ready: {target}")


def ml_modules():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sklearn.metrics import classification_report
    torch.set_num_threads(2)
    random.seed(42)
    torch.manual_seed(42)
    return torch, AutoTokenizer, AutoModelForSequenceClassification, classification_report


def predict(rows, tokenizer, model, torch, device):
    model.eval()
    labels = []
    with torch.no_grad():
        for start in range(0, len(rows), 4):
            tokens = tokenizer([r["content"] for r in rows[start:start + 4]], padding=True,
                               truncation=True, max_length=128, return_tensors="pt").to(device)
            labels.extend(LABELS[i] for i in model(**tokens).logits.argmax(-1).cpu().tolist())
    return labels


def train_model(args):
    torch, AutoTokenizer, AutoModel, report = ml_modules()
    rows = read_data(args.data)
    target = Path(args.model).resolve()
    if target.exists():
        raise FileExistsError("Output model directory exists; choose a new --model")
    base = Path(args.base_model).resolve()
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    model = AutoModel.from_pretrained(base, num_labels=3, local_files_only=True,
                                    id2label=dict(enumerate(LABELS)), label2id={v: k for k, v in enumerate(LABELS)})
    device = torch.device(args.device)
    model.to(device)
    if args.head_only:
        for parameter in model.base_model.parameters():
            parameter.requires_grad = False
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-5)
    training = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "validation"]
    best = -1
    for epoch in range(args.epochs):
        model.train()
        random.shuffle(training)
        for start in range(0, len(training), 4):
            batch = training[start:start + 4]
            tokens = tokenizer([r["content"] for r in batch], padding=True, truncation=True,
                               max_length=128, return_tensors="pt").to(device)
            truth = torch.tensor([LABELS.index(r["sentiment"]) for r in batch], device=device)
            optimizer.zero_grad()
            loss = model(**tokens, labels=truth).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        guessed = predict(validation, tokenizer, model, torch, device)
        metrics = report([r["sentiment"] for r in validation], guessed, labels=LABELS, output_dict=True, zero_division=0)
        score = metrics["macro avg"]["f1-score"]
        if score > best:
            best = score
            target.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(target, safe_serialization=True)
            tokenizer.save_pretrained(target)
            write_json(target / "validation-metrics.json", metrics)
        print(f"Epoch {epoch + 1}/{args.epochs}: validation macro F1={score:.4f}", flush=True)
    write_json(target / "training-run.json", {"seed": 42, "epochs": args.epochs, "head_only": args.head_only,
               "base_model": str(base), "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
               "note": "Synthetic sample verifies execution only; it is not a real-world quality benchmark."})
    print(f"Saved best validation checkpoint: {target}")


def analyze(args):
    torch, AutoTokenizer, AutoModel, report = ml_modules()
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import SVC
    if Path(args.output).exists():
        raise FileExistsError("Output already exists; choose a new --output")
    rows = read_data(args.data)
    model_path = Path(args.model).resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True)
    if [model.config.id2label.get(i) for i in range(3)] != LABELS:
        raise ValueError("Expected a trained negative/neutral/positive classifier. Run the train command first.")
    device = torch.device(args.device)
    model.to(device)
    predicted = predict(rows, tokenizer, model, torch, device)
    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]
    svm = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(1, 2)), SVC(kernel="linear", random_state=42))
    svm.fit([r["content"] for r in train_rows], [r["sentiment"] for r in train_rows])
    truth = [r["sentiment"] for r in test_rows]
    bert_test = [label for row, label in zip(rows, predicted) if row["split"] == "test"]
    metrics = {"note": "Held-out test split only. Synthetic sample scores do not measure real-world performance.",
               "bert": report(truth, bert_test, labels=LABELS, output_dict=True, zero_division=0),
               "svm": report(truth, svm.predict([r["content"] for r in test_rows]), labels=LABELS, output_dict=True, zero_division=0)}
    sample = Path(args.data).read_bytes() == SAMPLE.read_bytes()
    method = ("虚构示例数据 · " if sample else "用户提供的数据 · ") + "BERT 情感分类 / SVM 留出集评估"
    export_dashboard(rows, predicted, args.output, method, args.lstm_epochs)
    write_json(Path(args.output) / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Offline synthetic-data demo; no third-party Python dependencies")
    demo.add_argument("--output", type=Path, default=ROOT / "artifacts/demo")
    download = commands.add_parser("download", help="Download pinned public base BERT")
    download.add_argument("--model", type=Path, default=ROOT / "artifacts/base-model")
    train = commands.add_parser("train", help="Train a classifier on train split; select on validation split")
    train.add_argument("--data", type=Path, default=SAMPLE)
    train.add_argument("--base-model", type=Path, default=ROOT / "artifacts/base-model")
    train.add_argument("--model", type=Path, default=ROOT / "artifacts/trained-model")
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    train.add_argument("--head-only", action="store_true", help="Freeze encoder for a faster smoke run; not full fine-tuning")
    analysis = commands.add_parser("analyze", help="BERT inference, SVM evaluation, LSTM and dashboard export")
    analysis.add_argument("--data", type=Path, default=SAMPLE)
    analysis.add_argument("--model", type=Path, default=ROOT / "artifacts/trained-model")
    analysis.add_argument("--output", type=Path, default=ROOT / "artifacts/bert-dashboard")
    analysis.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    analysis.add_argument("--lstm-epochs", type=int, default=30)
    serve = commands.add_parser("serve", help="Serve generated dashboard on loopback only")
    serve.add_argument("--directory", type=Path, default=ROOT / "artifacts/demo")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        if getattr(args, "epochs", 1) < 1 or getattr(args, "lstm_epochs", 0) < 0:
            raise ValueError("epochs must be positive and lstm-epochs non-negative")
        if args.command == "demo":
            rows = read_data(SAMPLE)
            export_dashboard(rows, [rule_predict(r["content"]) for r in rows], args.output,
                             "虚构示例数据 · 关键词规则分类（非 BERT）")
        elif args.command == "download":
            download_model(args)
        elif args.command == "train":
            train_model(args)
        elif args.command == "analyze":
            analyze(args)
        elif args.command == "serve":
            if not (args.directory / "index.html").is_file():
                raise FileNotFoundError("Dashboard missing; run demo or analyze first")
            server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(SimpleHTTPRequestHandler, directory=str(args.directory.resolve())))
            print(f"Open http://127.0.0.1:{args.port} (Ctrl+C to stop)", flush=True)
            try:
                server.serve_forever()
            finally:
                server.server_close()
    except KeyboardInterrupt:
        pass
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
