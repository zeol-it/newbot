from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_dataset = script_dir / "antysmuty_contextual.jsonl"

    parser = argparse.ArgumentParser(
        description="Podziel dataset, wyślij pliki do OpenAI i utwórz job fine-tuningu."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_dataset,
        help="Wejściowy plik JSONL z przykładami treningowymi.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=None,
        help="Opcjonalny plik wyjściowy dla train JSONL.",
    )
    parser.add_argument(
        "--valid-output",
        type=Path,
        default=None,
        help="Opcjonalny plik wyjściowy dla validation JSONL.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Udział validation set, np. 0.1 dla 10%%.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed do tasowania przykładów przed splitem.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Bazowy model OpenAI obsługujący fine-tuning.",
    )
    parser.add_argument(
        "--suffix",
        default="antysmuty-style",
        help="Suffix dla tworzonego modelu fine-tuned.",
    )
    parser.add_argument(
        "--skip-split",
        action="store_true",
        help="Nie twórz nowych plików splitu; użyj istniejących train/valid.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Przygotuj split i wypisz plan, ale nie wysyłaj plików do OpenAI.",
    )
    return parser.parse_args()


def validate_api_key() -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return
    raise SystemExit("Brak OPENAI_API_KEY w środowisku.")


def load_jsonl(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"Plik datasetu nie istnieje: {path}")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise SystemExit("Dataset jest zbyt mały do sensownego splitu train/validation.")
    for line_number, line in enumerate(lines, start=1):
        try:
            json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Niepoprawny JSON w linii {line_number}: {exc}") from exc
    return lines


def split_paths(dataset_path: Path, train_output: Path | None, valid_output: Path | None) -> tuple[Path, Path]:
    train_path = train_output or dataset_path.with_name(f"{dataset_path.stem}.train.jsonl")
    valid_path = valid_output or dataset_path.with_name(f"{dataset_path.stem}.valid.jsonl")
    return train_path, valid_path


def split_dataset(lines: list[str], validation_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    if not 0 < validation_ratio < 1:
        raise SystemExit("--validation-ratio musi być w zakresie (0, 1).")

    shuffled = list(lines)
    random.Random(seed).shuffle(shuffled)
    validation_size = max(1, int(len(shuffled) * validation_ratio))
    training_size = len(shuffled) - validation_size
    if training_size < 1:
        raise SystemExit("Po splicie train set byłby pusty. Zmniejsz --validation-ratio.")
    train_lines = shuffled[:training_size]
    valid_lines = shuffled[training_size:]
    return train_lines, valid_lines


def write_jsonl(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def upload_file(client: OpenAI, path: Path):
    with path.open("rb") as handle:
        return client.files.create(file=handle, purpose="fine-tune")


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.resolve()
    train_path, valid_path = split_paths(dataset_path, args.train_output, args.valid_output)

    if args.skip_split:
        if not train_path.exists() or not valid_path.exists():
            raise SystemExit("Przy --skip-split muszą istnieć oba pliki train i valid.")
        train_count = sum(1 for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip())
        valid_count = sum(1 for line in valid_path.read_text(encoding="utf-8").splitlines() if line.strip())
    else:
        lines = load_jsonl(dataset_path)
        train_lines, valid_lines = split_dataset(lines, args.validation_ratio, args.seed)
        write_jsonl(train_path, train_lines)
        write_jsonl(valid_path, valid_lines)
        train_count = len(train_lines)
        valid_count = len(valid_lines)

    print(f"dataset={dataset_path}")
    print(f"train_file={train_path} ({train_count} examples)")
    print(f"valid_file={valid_path} ({valid_count} examples)")
    print(f"model={args.model}")
    print(f"suffix={args.suffix}")

    if args.dry_run:
        print("dry_run=true")
        return

    validate_api_key()
    client = OpenAI()

    train_upload = upload_file(client, train_path)
    valid_upload = upload_file(client, valid_path)

    print(f"train_upload_id={train_upload.id}")
    print(f"valid_upload_id={valid_upload.id}")

    job = client.fine_tuning.jobs.create(
        training_file=train_upload.id,
        validation_file=valid_upload.id,
        model=args.model,
        suffix=args.suffix,
    )

    print(f"job_id={job.id}")
    print(f"status={job.status}")
    print("Aby sprawdzić status później, uruchom:")
    print(
        "./envbrzydalek/bin/python - <<'PY'\n"
        "from openai import OpenAI\n"
        "client = OpenAI()\n"
        f"job = client.fine_tuning.jobs.retrieve('{job.id}')\n"
        "print(job.status)\n"
        "print(getattr(job, 'fine_tuned_model', None))\n"
        "PY"
    )


if __name__ == "__main__":
    main()