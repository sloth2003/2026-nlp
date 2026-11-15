"""
main.py - 로컬 실행 진입점 (Colab 은 notebook/train.ipynb 사용)

데이터 -> 모델 -> 학습 -> 방향별 test 평가 -> 제출용 .pt 6개 저장 -> 저장 파일 확인

실행 (project01-dev 폴더에서):
    python src/main.py                        # config.json 의 epochs 만큼 학습 (GPU 있으면 GPU)
    python src/main.py --limit 96 --epochs 2  # 일부 데이터로 빠르게 동작 확인
    python src/main.py --device cpu           # CPU 로 강제
    python src/main.py --fresh                # 저장된 지점을 버리고 처음부터

- 다시 실행하면 checkpoints/last.pt 에서 이어서 학습한다.
- 결과: project01-dev/checkpoints/last.pt, last.best.pt, final/<방향>.pt 6개
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch

import data
from checkpoint import load_model, load_training_state, save_all
from inference import evaluate, translate
from seq2seq import DIRECTIONS, build_model
from train import train

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = {"num": "12708", "kor": "만 이천칠백팔", "eng": "twelve thousand seven hundred eight"}


def parse_args():
    p = argparse.ArgumentParser(description="Seq2Seq Num2Text 로컬 학습")
    p.add_argument("--epochs", type=int, default=None, help="목표 총 epoch 수 (기본: config.json 의 epochs)")
    p.add_argument("--limit", type=int, default=None, help="분할마다 앞에서 N개만 사용 (빠른 동작 확인용)")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--fresh", action="store_true", help="저장된 last.pt 를 지우고 처음부터 학습")
    p.add_argument("--beam-width", type=int, default=5)
    p.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints"), help="학습 상태·제출 파일 저장 폴더")
    return p.parse_args()


def pick_device(name):
    """GPU 가 없으면 항상 CPU 로 내려간다."""
    if name == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if name == "cuda":
        print("[경고] CUDA 를 쓸 수 없어 CPU 로 실행합니다. (GPU 드라이버 / CUDA 빌드 torch 설치 확인)")
    return "cpu"


def main():
    args = parse_args()
    config = dict(data.config)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    model_kwargs = dict(config["model"])

    data.seed_everything(config["random_seed"])
    device = pick_device(args.device)
    if device == "cuda":
        print(f"device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("device: cpu")
        if args.limit is None:
            print("[안내] CPU 로 전체 데이터를 학습하면 매우 오래 걸립니다. 동작 확인은 --limit 96 --epochs 2 를 권장합니다.")

    # 1. 데이터
    loaders, src_vocabs, tgt_vocabs = data.build_loaders(config, DIRECTIONS, limit=args.limit)
    src_vocab = src_vocabs["num"]   # 세 입력 vocab 은 id 가 같다. 모델 크기·저장용 대표
    print("입력 vocab:", len(src_vocab), "| 출력 vocab:", {l: len(v) for l, v in tgt_vocabs.items()})
    for d in DIRECTIONS:
        print(f"  {d:8s}", {k: len(loaders[d][k].dataset) for k in ("train", "valid", "test")})

    # 2. 모델
    model = build_model(len(src_vocab), {l: len(v) for l, v in tgt_vocabs.items()}, **model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    print("파라미터:", f"{sum(p.numel() for p in model.parameters()):,}")

    # 3. 학습 (저장된 지점이 있으면 이어서)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last = ckpt_dir / "last.pt"
    best = ckpt_dir / "last.best.pt"
    if args.fresh:
        for path in (last, best):
            path.unlink(missing_ok=True)

    start_epoch = 0
    if last.is_file():
        try:
            last_epoch, _ = load_training_state(last, model, optimizer, device)
        except RuntimeError as e:
            # torch 의 오류는 수십 줄이라 첫 줄만 보여 주고, 해결 방법을 마지막에 둔다.
            raise SystemExit(
                f"{last} 를 불러올 수 없습니다: {str(e).splitlines()[0]}\n"
                "모델 크기나 vocab 이 저장할 때와 다릅니다. --fresh 로 처음부터 학습하세요."
            )
        start_epoch = last_epoch + 1
        if start_epoch >= config["epochs"]:
            print(f"[학습 완료 상태] {last}: 이미 {start_epoch} epoch 까지 학습했습니다. "
                  "더 학습하려면 --epochs 를 늘리세요. (평가와 저장만 다시 합니다)")
        else:
            print(f"[이어서 학습] {last} -> epoch {start_epoch + 1}부터")
    else:
        print("[처음부터 학습]")

    started = time.time()
    train(model, loaders, optimizer, config, device=device, start_epoch=start_epoch, ckpt_path=last)
    print(f"학습 시간: {(time.time() - started) / 60:.1f}분")

    # 4. 방향별 test 평가 (beam) - train() 이 valid loss 가 가장 낮았던 가중치로 되돌려 놓은 상태
    model.eval()
    for d in DIRECTIONS:
        s, t = d.split("2")
        r = evaluate(model.direction(d), loaders[d]["test"], tgt_vocabs[t],
                     method="beam", beam_width=args.beam_width, src_vocab=src_vocabs[s])
        print(f"  {d:8s} test acc {r['acc']:.4f} ({r['correct']}/{r['total']})  틀린 예: {r['errors'][:2]}")

    # 5. 제출용 .pt 6개 저장 -> 저장한 파일만으로 다시 로드해서 번역 확인
    final_dir = ckpt_dir / "final"
    paths = save_all(str(final_dir), model, src_vocab, tgt_vocabs, model_kwargs)
    for path in paths:
        d = Path(path).stem
        m, sv, tv = load_model(path, device=device)
        src_text = EXAMPLES[d.split("2")[0]]
        out = translate(m, src_text, sv, tv, method="beam", beam_width=args.beam_width)
        print(f"  {d:8s} {Path(path).stat().st_size / 1e6:.1f} MB  {src_text!r} -> {out!r}")
    print(f"제출용 파일 저장: {final_dir}")


if __name__ == "__main__":
    main()
