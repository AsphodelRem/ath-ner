"""Доменная адаптация энкодера маскированным языковым моделированием.

Первая стадия двухстадийной схемы: энкодер дообучается на большом узбекском
корпусе без разметки, затем solution/train.py дообучает его под NER.

Мотивация из разбора ошибок: recall на невиданных формах 73,2% у ORG и 76,2%
у GEO против 95% на знакомых. Пропущены `Moskvich`, `UConn`, `ТИНКОФ` —
названия, которых модель не встречала. MLM на сотнях миллионов токенов
узбекского текста даёт энкодеру представления для таких строк без разметки.

Обучение идёт через transformers.Trainer: цикл здесь совершенно обычный, и
Trainer бесплатно даёт возобновление после обрыва, чекпоинты, логирование и
смешанную точность. Своего кода остаётся ровно столько, сколько нужно на
упаковку блоков и маскирование целых слов.

    python -m solution.pretrain --corpus data/pretrain/corpus.jsonl \
        --eval-corpus data/pretrain/corpus.holdout.jsonl \
        --model-name /path/to/xlm-roberta-large --output-dir artifacts/mlm-large \
        --max-steps 18000

Результат в <output-dir>/model сразу пригоден как --model-name для
solution/train.py: голова MLM отбрасывается, энкодер и токенизатор берутся.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

JsonObject = dict[str, Any]


def parse_args() -> argparse.Namespace:
    """Разбирает параметры адаптации."""

    parser = argparse.ArgumentParser(description="Domain-adaptive MLM pretraining.")
    parser.add_argument("--corpus", type=Path, required=True, help="JSONL с полем text")
    parser.add_argument("--eval-corpus", type=Path, help="held-out JSONL для перплексии")
    parser.add_argument("--model-name", required=True, help="каталог или имя базовой модели")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    # Низкий lr принципиален: на большом модель переучится на домен и потеряет
    # мультиязычность, которая нужна для 5% нецелевых текстов в выборке.
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--no-whole-word-mask", action="store_true")
    parser.add_argument("--eval-blocks", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--resume", action="store_true", help="продолжить с последнего чекпоинта")
    parser.add_argument("--comet", choices=("off", "online", "offline"), default="off")
    parser.add_argument("--comet-project", default="uzbek-ner-mlm")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser.parse_args()


class PackedTextDataset(IterableDataset):
    """Поток блоков фиксированной длины из JSONL.

    Документы токенизируются и склеиваются в непрерывный поток, который режется
    на блоки. Так не теряется хвост коротких текстов и не тратится вычисление
    на паддинг: медианный пассаж заметно короче окна.
    """

    def __init__(self, path: Path, tokenizer: Any, *, block_size: int, single_pass: bool = False) -> None:
        self.path = path
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.single_pass = single_pass
        # В transformers 5 нет build_inputs_with_special_tokens, поэтому обёртку
        # выясняем у самого токенизатора — так код не привязан к архитектуре.
        wrapper = tokenizer("", add_special_tokens=True)["input_ids"]
        middle = len(wrapper) // 2
        self.prefix, self.suffix = wrapper[:middle], wrapper[middle:]

    def __iter__(self) -> Iterator[dict[str, list[int]]]:
        """Выдаёт блоки; воркеры читают непересекающиеся строки."""

        info = get_worker_info()
        shard, shards = (info.id, info.num_workers) if info else (0, 1)
        content = self.block_size - len(self.prefix) - len(self.suffix)
        buffer: list[int] = []
        heads: list[int] = []
        while True:  # корпус проходится по кругу, число шагов задаёт --max-steps
            with self.path.open(encoding="utf-8") as stream:
                for index, line in enumerate(stream):
                    if index % shards != shard or not line.strip():
                        continue
                    text = json.loads(line).get("text")
                    if not text:
                        continue
                    encoded = self.tokenizer(text, add_special_tokens=False)
                    ids = encoded["input_ids"]
                    words = encoded.word_ids()
                    buffer.extend(ids)
                    heads.extend(1 if i == 0 or words[i] != words[i - 1] else 0
                                 for i in range(len(ids)))
                    while len(buffer) >= content:
                        chunk, buffer = buffer[:content], buffer[content:]
                        chunk_heads, heads = heads[:content], heads[content:]
                        yield {
                            "input_ids": self.prefix + chunk + self.suffix,
                            "word_starts": [0] * len(self.prefix) + chunk_heads + [0] * len(self.suffix),
                        }
            if self.single_pass:
                return


class WholeWordMaskCollator:
    """Маскирует слова целиком, а не отдельные сабтокены.

    Штатный DataCollatorForLanguageModeling определяет границы слов по
    offset_mapping, которого у склеенных блоков быть не может. Здесь границы
    приходят из датасета признаком word_starts.

    Маскирование целых слов принципиально для нашего случая: кириллическое
    слово распадается на 2-4 сабтокена, и восстановить один из них по соседним
    заметно проще, чем слово целиком.

    Уже размеченные элементы (с готовым labels) пропускаются как есть — так
    отложенная выборка маскируется один раз и остаётся сравнимой между замерами.
    """

    def __init__(self, tokenizer: Any, *, probability: float = 0.15, whole_word: bool = True) -> None:
        self.tokenizer = tokenizer
        self.probability = probability
        self.whole_word = whole_word
        self.special = set(tokenizer.all_special_ids)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Собирает батч и расставляет маски."""

        if "labels" in features[0]:  # заранее замаскированная отложенная выборка
            return {key: torch.stack([torch.as_tensor(f[key]) for f in features])
                    for key in ("input_ids", "attention_mask", "labels")}

        width = max(len(item["input_ids"]) for item in features)
        pad = self.tokenizer.pad_token_id
        input_ids = torch.full((len(features), width), pad, dtype=torch.long)
        attention = torch.zeros((len(features), width), dtype=torch.long)
        labels = torch.full((len(features), width), -100, dtype=torch.long)

        for row, item in enumerate(features):
            ids = item["input_ids"]
            input_ids[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention[row, :len(ids)] = 1

            if self.whole_word:
                groups: list[list[int]] = []
                for position, head in enumerate(item["word_starts"]):
                    if ids[position] in self.special:
                        continue
                    if head or not groups:
                        groups.append([position])
                    else:
                        groups[-1].append(position)
            else:
                groups = [[p] for p, token in enumerate(ids) if token not in self.special]
            if not groups:
                continue

            for group in random.sample(groups, min(max(1, round(len(groups) * self.probability)),
                                                   len(groups))):
                for position in group:
                    labels[row, position] = ids[position]
                    draw = random.random()
                    if draw < 0.8:
                        input_ids[row, position] = self.tokenizer.mask_token_id
                    elif draw < 0.9:
                        input_ids[row, position] = random.randrange(len(self.tokenizer))
        return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}


class StatusCallback(TrainerCallback):
    """Печатает компактный статус, пригодный для чтения в файле лога.

    Штатный вывод Trainer — это словари вида {'loss': ..., 'epoch': ...} плюс
    прогресс-бар tqdm, который в перенаправленном выводе превращается в тысячи
    строк с возвратами каретки. Здесь одна строка на замер, с префиксом [MLM]
    для grep, реальным номером шага и перплексией вместо голого eval_loss.

    Поле epoch у Trainer здесь бессмысленно: корпус читается потоком по кругу.
    """

    def __init__(self, total_steps: int, tokens_per_step: int) -> None:
        self.total = total_steps
        self.tokens = tokens_per_step
        self.started = time.time()

    def on_log(self, args: Any, state: Any, control: Any, logs: JsonObject | None = None, **kwargs: Any) -> None:
        """Форматирует очередную запись лога."""

        if not logs or not state.is_world_process_zero:
            return
        step = int(state.global_step)
        if "eval_loss" in logs:
            loss = float(logs["eval_loss"])
            print(f"[MLM] held-out: loss {loss:.4f} | перплексия {math.exp(min(loss, 20)):.2f}",
                  flush=True)
            return
        if "loss" not in logs:
            return
        elapsed = time.time() - self.started
        remaining = elapsed / max(step, 1) * max(self.total - step, 0)
        print(
            f"[MLM] шаг {step:>6}/{self.total} ({100 * step / self.total:4.1f}%)"
            f" | loss {float(logs['loss']):.4f}"
            f" | lr {float(logs.get('learning_rate', 0)):.2e}"
            f" | токенов {step * self.tokens / 1e6:.1f} млн"
            f" | прошло {elapsed / 3600:.1f} ч | осталось ~{remaining / 3600:.1f} ч",
            flush=True,
        )


def build_eval_set(path: Path, tokenizer: Any, collator: WholeWordMaskCollator,
                   *, block_size: int, blocks: int) -> list[dict[str, torch.Tensor]]:
    """Готовит отложенную выборку, замаскированную один раз с фиксированным сидом."""

    state = random.getstate()
    random.seed(12345)
    dataset = PackedTextDataset(path, tokenizer, block_size=block_size, single_pass=True)
    collected = []
    for item in dataset:
        collected.append(item)
        if len(collected) >= blocks:
            break
    batch = collator(collected)
    random.setstate(state)
    return [{key: batch[key][i] for key in batch} for i in range(len(collected))]


def configure_comet(mode: str, project: str, output_dir: Path) -> list[str]:
    """Настраивает встроенную интеграцию Trainer с Comet через переменные среды."""

    if mode == "off":
        os.environ["COMET_MODE"] = "DISABLED"
        return []
    os.environ["COMET_PROJECT_NAME"] = project
    if mode == "offline":
        directory = output_dir / "comet"
        directory.mkdir(parents=True, exist_ok=True)
        os.environ["COMET_MODE"] = "OFFLINE"
        os.environ["COMET_OFFLINE_DIRECTORY"] = str(directory)
    return ["comet_ml"]


def run(args: argparse.Namespace) -> int:
    """Выполняет адаптацию и сохраняет энкодер."""

    from solution.tracking import check_mode, load_env_file

    load_env_file(args.env_file)
    check_mode(args.comet)
    report_to = configure_comet(args.comet, args.comet_project, args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.mask_token is None:
        raise ValueError(f"{args.model_name}: у токенизатора нет mask-токена, MLM невозможен")
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)

    collator = WholeWordMaskCollator(
        tokenizer, probability=args.mlm_probability, whole_word=not args.no_whole_word_mask
    )
    train_dataset = PackedTextDataset(args.corpus, tokenizer, block_size=args.max_length)
    eval_dataset = None
    if args.eval_corpus and args.eval_corpus.exists():
        eval_dataset = build_eval_set(
            args.eval_corpus, tokenizer, collator,
            block_size=args.max_length, blocks=args.eval_blocks,
        )
        print(f"Отложенная выборка: {len(eval_dataset)} блоков из {args.eval_corpus}")
    elif args.eval_corpus:
        print(f"ВНИМАНИЕ: {args.eval_corpus} не найден, перплексия считаться не будет")

    tokens_per_step = args.batch_size * args.gradient_accumulation_steps * args.max_length
    print(f"База: {args.model_name}")
    print(f"Шагов: {args.max_steps} | токенов за шаг: {tokens_per_step}"
          f" | всего ~{tokens_per_step * args.max_steps / 1e6:.0f} млн токенов\n")

    arguments = TrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=int(args.max_steps * args.warmup_ratio),
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type="linear",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        logging_steps=args.log_every,
        logging_first_step=True,
        save_steps=args.save_every,
        save_strategy="steps" if args.save_every else "no",
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.eval_every,
        dataloader_num_workers=args.num_workers,
        gradient_checkpointing=args.gradient_checkpointing,
        # Датасет отдаёт word_starts, а модель его не принимает: без этого
        # Trainer вырежет поле до коллатора, и маскирование целых слов отвалится.
        remove_unused_columns=False,
        report_to=report_to,
        seed=args.seed,
        # В перенаправленном выводе tqdm пишет тысячи строк с возвратами
        # каретки; в терминале он полезен, поэтому смотрим на TTY.
        disable_tqdm=not sys.stdout.isatty(),
    )

    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[StatusCallback(args.max_steps, tokens_per_step)],
    )
    # resume_from_checkpoint=True требует существующего чекпоинта и падает,
    # если его нет. Поэтому ищем последний сами: тогда --resume безопасен и
    # при первом запуске, и при перезапуске после обрыва.
    last = get_last_checkpoint(str(args.output_dir)) if args.resume and args.output_dir.exists() else None
    if args.resume:
        print(f"Возобновление с {last}" if last else "Чекпоинта нет, старт с нуля")
    trainer.train(resume_from_checkpoint=last)

    model_dir = args.output_dir / "model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(model_dir)

    info: JsonObject = {
        "stage": "domain-adaptive MLM",
        "base_model": args.model_name,
        "corpus": str(args.corpus),
        "max_length": args.max_length,
        "tokens_per_step": tokens_per_step,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "mlm_probability": args.mlm_probability,
        "whole_word_mask": not args.no_whole_word_mask,
        "seed": args.seed,
        "steps_done": int(trainer.state.global_step),
    }
    if eval_dataset:
        metrics = trainer.evaluate()
        info["eval_loss"] = metrics.get("eval_loss")
        info["eval_perplexity"] = math.exp(min(metrics.get("eval_loss", 20), 20))
        print(f"\nHeld-out перплексия: {info['eval_perplexity']:.2f}")
    (model_dir / "pretrain_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "history.json").write_text(
        json.dumps(trainer.state.log_history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\nЭнкодер: {model_dir}")
    print(f"Дальше: python -m solution.train --model-name {model_dir} ...")
    return 0


def main() -> int:
    """Точка входа CLI."""

    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
