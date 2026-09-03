"""Обучение NER с отбором чекпоинта по exact-span F1.

Отличия от baseline/train.py:
  * лучший чекпоинт выбирается по micro-F1 официального scorer, а не по dev_loss;
  * предсказания на dev проходят постобработку границ (solution/postprocess.py);
  * поддержаны bf16 и gradient checkpointing, чтобы обучение влезало в 6 ГБ VRAM.

Пример пробного прогона на ноутбуке:
    python -m solution.train --preset laptop --output-dir artifacts/run01
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

# comet_ml должен импортироваться раньше torch, поэтому трекинг идёт первым.
from solution.tracking import Tracker, check_mode, create_tracker, flatten_metrics, load_env_file

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    get_linear_schedule_with_warmup,
)

from baseline.common import (
    load_fast_tokenizer,
    read_records,
    resolve_device,
    validate_window,
)
from solution.inference import predict_records
from solution.metrics import evaluate
from solution.augment import Augmenter, build_gazetteer
from solution.tagging import NerDataset, tags_for
from solution.viterbi import build_log_transitions, estimate_transitions, load_transitions

JsonObject = dict[str, Any]

PRESETS = {
    # Дымовой прогон: проверяет пайплайн целиком за несколько минут.
    "smoke": dict(
        model_name="distilbert/distilbert-base-multilingual-cased",
        max_length=256, stride=64, batch_size=8, gradient_accumulation_steps=1,
        learning_rate=5e-5, epochs=1, limit=1500, gradient_checkpointing=False,
    ),
    # Первый настоящий прогон на 6 ГБ VRAM.
    "laptop": dict(
        model_name="FacebookAI/xlm-roberta-base",
        max_length=256, stride=64, batch_size=4, gradient_accumulation_steps=4,
        learning_rate=3e-5, epochs=4, limit=None, gradient_checkpointing=True,
    ),
    # A100: большая модель, широкое окно, без gradient checkpointing.
    # При 40 ГБ и OOM используйте --batch-size 8 --gradient-accumulation-steps 4.
    "a100": dict(
        model_name="FacebookAI/xlm-roberta-large",
        max_length=512, stride=128, batch_size=16, gradient_accumulation_steps=2,
        learning_rate=1e-5, epochs=8, limit=None, gradient_checkpointing=False,
    ),
}


def parse_args() -> argparse.Namespace:
    """Разбирает параметры обучения; --preset задаёт значения по умолчанию."""

    parser = argparse.ArgumentParser(description="Train exact-span NER for the Uzbek case.")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="laptop")
    parser.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--dev", type=Path, default=Path("data/dev.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/run01"))
    parser.add_argument("--model-name")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--limit", type=int, help="взять только первые N записей train")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=2, help="эпох без роста F1 до остановки")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=None)
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="выключить явно: пресет laptop включает его ради 6 ГБ, на A100 это"
             " лишние 30-40%% времени при незанятой памяти",
    )
    parser.add_argument("--tag-scheme", choices=("bio", "bilou"), default="bio")
    parser.add_argument(
        "--augment-case",
        type=float,
        default=0.0,
        help="доля записей с искажённым регистром (11%% ORG написаны капсом, 8,4%% со строчной)",
    )
    parser.add_argument(
        "--augment-swap",
        type=float,
        default=0.0,
        help="доля записей с подменой сущностей (recall на невиданных формах 73,2%% против 95,1%%)",
    )
    parser.add_argument(
        "--augment-swap-share",
        type=float,
        default=0.5,
        help="какая доля сущностей внутри выбранной записи подменяется",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="нужен моделям с собственным кодом (gte-multilingual). В офлайн-контейнере\n             этот код придётся вшивать в образ заранее",
    )
    parser.add_argument(
        "--viterbi",
        action="store_true",
        help="декодировать Витерби вместо argmax (и отбирать чекпоинт по этой же метрике)",
    )
    parser.add_argument(
        "--transition-weight",
        type=float,
        default=0.25,
        help=(
            "сила ограничений: 0 — переходы не влияют, 1 — как оценено, >1 — жёстче. "
            "Дефолт 0.25 подобран на run03: F1 на плато, но recall однотокенных "
            "сущностей 93%% против 79%% при весе 1.0"
        ),
    )
    parser.add_argument("--transitions", type=Path, help="готовый файл матриц переходов")
    parser.add_argument(
        "--transition-mode",
        choices=("conditional", "structural"),
        default="conditional",
        help="structural вычитает унарный приор тегов: выше recall, ниже precision",
    )
    parser.add_argument("--no-postprocess", action="store_true", help="отключить постобработку границ")
    parser.add_argument(
        "--comet",
        choices=("off", "online", "offline"),
        default="off",
        help="логирование в Comet ML: offline пишет .zip локально и не ходит в сеть",
    )
    parser.add_argument("--comet-project", default="uzbek-ner")
    parser.add_argument("--comet-workspace")
    parser.add_argument("--comet-offline-dir", type=Path, help="по умолчанию <output-dir>/comet")
    parser.add_argument("--comet-tag", action="append", dest="comet_tags", default=[])
    parser.add_argument("--log-every", type=int, default=25, help="шагов между точками кривой loss")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="файл с COMET_API_KEY")
    args = parser.parse_args()
    for key, value in PRESETS[args.preset].items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    if args.no_gradient_checkpointing:
        args.gradient_checkpointing = False
    return args


def set_seed(seed: int) -> None:
    """Фиксирует генераторы случайных чисел."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loader(
    dataset: TokenizedNerDataset,
    collator: DataCollatorForTokenClassification,
    batch_size: int,
    *,
    shuffle: bool,
) -> DataLoader:
    """Создаёт DataLoader с паддингом меток значением -100."""

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler: Any,
    device: torch.device,
    *,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    autocast_dtype: torch.dtype | None,
    tracker: Tracker,
    global_step: int,
    log_every: int,
) -> tuple[float, int]:
    """Проходит одну эпоху и возвращает средний loss и номер шага оптимизатора."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    steps = 0
    window_sum = 0.0
    window_count = 0
    for index, batch in enumerate(tqdm(loader, desc="Train", unit="batch", leave=False), start=1):
        batch = {key: value.to(device) for key, value in batch.items()}
        if autocast_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                loss = model(**batch).loss
        else:
            loss = model(**batch).loss
        (loss / gradient_accumulation_steps).backward()
        value = float(loss.item())
        total += value
        window_sum += value
        steps += 1
        window_count += 1
        if index % gradient_accumulation_steps == 0 or index == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if tracker.enabled and log_every > 0 and global_step % log_every == 0:
                tracker.log_metrics(
                    {
                        "train_loss_step": window_sum / max(window_count, 1),
                        "learning_rate": scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )
                window_sum = 0.0
                window_count = 0
    return total / max(steps, 1), global_step


def format_metrics(metrics: JsonObject) -> str:
    """Собирает однострочную сводку метрик."""

    parts = [
        f"{label} {metrics['by_label'][label]['f1']:.4f}" for label in ("ORG", "NAME", "GEO")
    ]
    return (
        f"micro F1 {metrics['micro']['f1']:.4f} "
        f"(P {metrics['micro']['precision']:.4f} / R {metrics['micro']['recall']:.4f})  "
        f"macro {metrics['macro']['f1']:.4f}  |  " + "  ".join(parts)
    )


def write_metrics_csv(path: Path, history: list[JsonObject]) -> None:
    """Сохраняет метрики по эпохам плоской таблицей — строится любым инструментом."""

    columns = [
        "epoch", "train_loss", "seconds",
        "micro_f1", "micro_precision", "micro_recall", "macro_f1",
        "ORG_f1", "NAME_f1", "GEO_f1",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for item in history:
            metrics = item["metrics"]
            writer.writerow({
                "epoch": item["epoch"],
                "train_loss": round(item["train_loss"], 6),
                "seconds": item["seconds"],
                "micro_f1": round(metrics["micro"]["f1"], 6),
                "micro_precision": round(metrics["micro"]["precision"], 6),
                "micro_recall": round(metrics["micro"]["recall"], 6),
                "macro_f1": round(metrics["macro"]["f1"], 6),
                **{
                    f"{label}_f1": round(metrics["by_label"][label]["f1"], 6)
                    for label in ("ORG", "NAME", "GEO")
                },
            })


def run(args: argparse.Namespace) -> int:
    """Обучает модель и сохраняет лучший по F1 чекпоинт."""

    loaded = load_env_file(args.env_file)
    if loaded:
        print(f"Из {args.env_file} загружено: {', '.join(loaded)}")
    check_mode(args.comet)  # падаем сразу, а не после токенизации
    set_seed(args.seed)
    device = resolve_device(args.device)
    autocast_dtype = None
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        autocast_dtype = torch.bfloat16

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_records = read_records(args.train, require_entities=True, limit=args.limit)
    dev_records = read_records(args.dev, require_entities=True)
    if args.trust_remote_code:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, use_fast=True, trust_remote_code=True
        )
        if not tokenizer.is_fast:
            raise ValueError("нужен fast tokenizer: без offset_mapping спаны не построить")
    else:
        tokenizer = load_fast_tokenizer(args.model_name)
    validate_window(tokenizer, args.max_length, args.stride)

    augmenter = Augmenter(
        build_gazetteer(train_records) if args.augment_swap > 0 else None,
        case_probability=args.augment_case,
        swap_probability=args.augment_swap,
        swap_share=args.augment_swap_share,
    )
    collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=-100)

    def make_train_loader(epoch: int) -> tuple[DataLoader, int]:
        """Собирает загрузчик; при аугментациях выборка меняется каждую эпоху."""

        records = train_records
        description = "Tokenize train"
        if augmenter.enabled:
            records = augmenter.apply_all(train_records, seed=args.seed * 1000 + epoch)
            description = f"Tokenize train (эпоха {epoch})"
        dataset = NerDataset(
            records, tokenizer, max_length=args.max_length, stride=args.stride,
            scheme=args.tag_scheme, description=description,
        )
        return build_loader(dataset, collator, args.batch_size, shuffle=True), len(dataset)

    train_loader, train_windows = make_train_loader(1)

    tags = tags_for(args.tag_scheme)
    id2label = dict(enumerate(tags))

    log_transitions = None
    if args.viterbi:
        transitions_path = args.transitions or (output_dir / "transitions.json")
        if transitions_path.exists():
            log_transitions, _ = load_transitions(transitions_path, mode=args.transition_mode)
            print(f"Матрицы переходов загружены: {transitions_path}")
        else:
            payload = estimate_transitions(
                train_records, tokenizer,
                max_length=args.max_length, stride=args.stride, scheme=args.tag_scheme,
            )
            transitions_path.parent.mkdir(parents=True, exist_ok=True)
            transitions_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            log_transitions = build_log_transitions(payload, mode=args.transition_mode)
            print(f"Матрицы переходов оценены и сохранены: {transitions_path}")
        # Множитель масштабирует логарифмы: 0 обнуляет влияние переходов,
        # значения больше единицы делают ограничения жёстче.
        log_transitions = log_transitions * args.transition_weight
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(tags),
        id2label=id2label,
        label2id={tag: index for index, tag in id2label.items()},
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader) // args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps
    )

    model_dir = output_dir / "model"
    print(
        f"Устройство: {device} | autocast: {autocast_dtype} | модель: {args.model_name}\n"
        f"Записей train: {len(train_records)} -> окон: {train_windows} | dev: {len(dev_records)}\n"
        f"Окно {args.max_length}/{args.stride} | батч {args.batch_size}"
        f" x {args.gradient_accumulation_steps} | lr {args.learning_rate} | эпох {args.epochs}\n"
    )

    config: JsonObject = {
        "preset": args.preset,
        "model_name": args.model_name,
        "max_length": args.max_length,
        "stride": args.stride,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "patience": args.patience,
        "seed": args.seed,
        "limit": args.limit,
        "postprocess": not args.no_postprocess,
        "tag_scheme": args.tag_scheme,
        "augment_case": args.augment_case,
        "augment_swap": args.augment_swap,
        "augment_swap_share": args.augment_swap_share if args.augment_swap else None,
        "trust_remote_code": bool(args.trust_remote_code),
        "viterbi": bool(args.viterbi),
        "transition_weight": args.transition_weight if args.viterbi else None,
        "transition_mode": args.transition_mode if args.viterbi else None,
        "num_tags": len(tags),
        "autocast": str(autocast_dtype),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "train_records": len(train_records),
        "train_windows": train_windows,
        "dev_records": len(dev_records),
        "train_path": str(args.train),
        "dev_path": str(args.dev),
    }

    tracker = create_tracker(
        args.comet,
        project=args.comet_project,
        workspace=args.comet_workspace,
        offline_dir=args.comet_offline_dir or (output_dir / "comet"),
        tags=[args.preset, args.model_name.split("/")[-1], *args.comet_tags],
    )
    if tracker.enabled:
        tracker.set_name(output_dir.name)
        tracker.log_parameters(config)
        print(f"Comet: режим {args.comet}\n")

    history: list[JsonObject] = []
    best_f1 = -1.0
    best_epoch = 0
    stale = 0
    started = time.time()
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        if augmenter.enabled and epoch > 1:
            train_loader, _ = make_train_loader(epoch)
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            autocast_dtype=autocast_dtype,
            tracker=tracker, global_step=global_step, log_every=args.log_every,
        )
        predictions = predict_records(
            model, tokenizer, dev_records,
            max_length=args.max_length, stride=args.stride,
            batch_size=args.eval_batch_size, device=device, id2label=id2label,
            scheme=args.tag_scheme, postprocess=not args.no_postprocess, progress=True,
            log_transitions=log_transitions,
        )
        metrics = evaluate(dev_records, predictions)
        elapsed = time.time() - epoch_started
        history.append({"epoch": epoch, "train_loss": train_loss, "metrics": metrics,
                        "seconds": round(elapsed, 1)})
        print(f"Эпоха {epoch}: loss {train_loss:.4f} | {format_metrics(metrics)} | {elapsed:.0f} c")

        if tracker.enabled:
            epoch_metrics = {"train_loss": train_loss, "epoch_seconds": elapsed}
            epoch_metrics.update(flatten_metrics(metrics))
            if device.type == "cuda":
                epoch_metrics["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
            tracker.log_metrics(epoch_metrics, step=global_step, epoch=epoch)

        if metrics["micro"]["f1"] > best_f1:
            best_f1 = metrics["micro"]["f1"]
            best_epoch = epoch
            stale = 0
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            (output_dir / "dev_metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            with (output_dir / "dev_predictions.jsonl").open("w", encoding="utf-8") as stream:
                for item in predictions:
                    stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"  ↑ новый лучший чекпоинт сохранён в {model_dir}")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"  ранняя остановка: F1 не растёт {stale} эпох(и)")
                break

    config.update({
        "best_epoch": best_epoch,
        "best_micro_f1": best_f1,
        "total_seconds": round(time.time() - started, 1),
    })
    (output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_metrics_csv(output_dir / "metrics.csv", history)
    if tracker.enabled:
        tracker.log_metrics({"best_micro_f1": best_f1, "best_epoch": best_epoch})
        for name in ("run_config.json", "history.json", "dev_metrics.json", "dev_predictions.jsonl"):
            tracker.log_asset(output_dir / name)
        destination = tracker.end()
        if destination:
            print(f"Comet: {destination}")

    print(f"\nЛучший результат: эпоха {best_epoch}, micro F1 {best_f1:.4f}")
    print(f"Артефакты: {output_dir}")
    return 0


def main() -> int:
    """Точка входа CLI."""

    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
