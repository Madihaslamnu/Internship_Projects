
from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from day3.data_utili import ID2LABEL, LABEL2ID, LABELS, get_split
from text_preprocessing import MENTION_RE, URL_RE, WHITESPACE_RE

ART = Path(__file__).parent / "artifacts"


def clean_for_transformer(text: str) -> str:
    s = html.unescape(str(text)).replace("\u2019", "'")
    s = URL_RE.sub("http", s)
    s = MENTION_RE.sub("@user", s)
    return WHITESPACE_RE.sub(" ", s).strip()


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def build_tiny_tokenizer_and_model(train_texts: list[str]):
    """Smoke-test only: a from-scratch WordPiece tokenizer + a ~1M-param random DistilBERT."""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
    from transformers import DistilBertConfig, DistilBertForSequenceClassification, PreTrainedTokenizerFast

    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.train_from_iterator(
        train_texts,
        trainers.WordPieceTrainer(vocab_size=2000, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]),
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]",
        cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
    )
    cfg = DistilBertConfig(
        vocab_size=fast.vocab_size, dim=64, hidden_dim=128, n_layers=2, n_heads=2,
        max_position_embeddings=128, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID,
    )
    return fast, DistilBertForSequenceClassification(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=64, help="tweets are <=36 words; 64 word-pieces is plenty")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--class_weights", action="store_true",
                    help="inverse-frequency weighted loss (helps minority-class F1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke_test", action="store_true", help="tiny random model on 400 tweets; no download")
    ap.add_argument("--out", default=str(ART / "distilbert"))
    args = ap.parse_args()

    set_seed(args.seed)
    ART.mkdir(exist_ok=True)
    out = Path(args.out)

    train_full, test = get_split()
    
    # Map text labels to integers if they are currently strings (negative/neutral/positive)
    if train_full["airline_sentiment"].dtype == object:
        train_full["label"] = train_full["airline_sentiment"].map(LABEL2ID)
        test["label"] = test["airline_sentiment"].map(LABEL2ID)

    # validation slice from TRAIN only
    train, val = train_test_split(train_full, test_size=0.1, stratify=train_full["label"], random_state=args.seed)
    if args.smoke_test:
        train, val, test_run = train.head(400), val.head(100), test.head(200)
    else:
        test_run = test
    print(f"train={len(train)} val={len(val)} test={len(test_run)} | device={'cuda' if torch.cuda.is_available() else 'cpu'}")

    # model + tokenizer 
    if args.smoke_test:
        tokenizer, model = build_tiny_tokenizer_and_model([clean_for_transformer(t) for t in train["text"]])
        args.epochs = 1
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model={'tiny-random' if args.smoke_test else args.model}  params={n_params / 1e6:.1f}M")

    # ---- data 
    def to_ds(df: pd.DataFrame) -> Dataset:
        ds = Dataset.from_dict({"text": [clean_for_transformer(t) for t in df["text"]],
                                "label": df["label"].tolist()})
        return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=args.max_len),
                      batched=True, remove_columns=["text"])

    ds_train, ds_val, ds_test = to_ds(train), to_ds(val), to_ds(test_run)
    lens = [len(x) for x in ds_train["input_ids"]]
    print(f"word-piece length: median={int(np.median(lens))} p95={int(np.percentile(lens, 95))} "
          f"max={max(lens)} (max_len={args.max_len}; truncated={np.mean(np.array(lens) >= args.max_len):.1%})")

    class_w = None
    if args.class_weights:
        counts = np.bincount(train["label"], minlength=3)
        class_w = torch.tensor(len(train) / (3 * counts), dtype=torch.float)
        print("class weights:", dict(zip(LABELS, class_w.tolist())))

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits, labels, weight=class_w.to(outputs.logits.device) if class_w is not None else None)
            return (loss, outputs) if return_outputs else loss

    # ---- train ----------------------------------------------------------------
    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        seed=args.seed,
        report_to="none",
        use_cpu=not torch.cuda.is_available(),
    )
    trainer_cls = WeightedTrainer if args.class_weights else Trainer
    trainer = trainer_cls(
        model=model, args=targs, train_dataset=ds_train, eval_dataset=ds_val,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0

    # ---- score the held-out test set ONCE 
    logits = trainer.predict(ds_test).predictions
    pred_ids = logits.argmax(-1)
    y_true = test_run["airline_sentiment"].values
    y_pred = np.array([ID2LABEL[i] for i in pred_ids])
    metrics = {
        "model": "smoke-test (random weights, ignore the numbers)" if args.smoke_test else args.model,
        "params_millions": round(n_params / 1e6, 1),
        "train_seconds": round(train_secs, 1),
        "hyperparameters": {k: getattr(args, k) for k in
                            ["epochs", "lr", "batch_size", "max_len", "weight_decay", "warmup_ratio", "class_weights", "seed"]},
        "best_val_macro_f1": round(max(l["eval_macro_f1"] for l in trainer.state.log_history if "eval_macro_f1" in l), 4),
        "test": {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
            **{f"f1_{l}": round(v, 4) for l, v in zip(LABELS, f1_score(y_true, y_pred, average=None, labels=LABELS))},
        },
    }
    print(json.dumps(metrics, indent=2))

    tag = "distilbert_smoke" if args.smoke_test else "distilbert"
    pd.DataFrame({"text": test_run["text"].values, "true": y_true, "pred": y_pred},
                 index=test_run.index).to_csv(ART / f"{tag}_test_predictions.csv", index_label="row_id")
    (ART / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=2))
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"saved model -> {out}")


if __name__ == "__main__":
    main()