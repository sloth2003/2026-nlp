"""
checkpoint.py - 모델 저장·로드 (담당: 세원)

1. 제출용: 방향별 .pt 6개
   save_all(...)  -> project01/num2kor.pt, kor2num.pt, ... (6개)
   load_model(path) -> 그 방향의 Seq2Seq, src_vocab, tgt_vocab

   파일 하나에 들어가는 것
   - direction        : "kor2eng" 등
   - encoder, decoder : 공유 Encoder 와 출력 언어 Decoder 의 state_dict
   - src_vocab, tgt_vocab : Vocab 객체의 속성 (itos, stoi, col)
   - model_config     : vocab 크기 + Encoder/Decoder 생성 인자 (같은 크기로 다시 만들기 위해)

2. 학습 중단 대비 (Colab 세션이 끊길 때)
   save_training_state(...) / load_training_state(...)
"""

import os

import torch

from model import Encoder, Decoder
from seq2seq import Seq2Seq, DIRECTIONS
from data import Vocab


# ---------------------------------------------------------------------
# Vocab <-> dict
# ---------------------------------------------------------------------
def _vocab_state(vocab, lang=None):
    """Vocab 객체 -> 저장용 dict (itos, stoi, col 같은 기본 자료형만)

    통합 입력 vocab 은 방향마다 입력 언어가 다르므로,
    lang 을 주면 그 방향 파일 안에서는 col(토큰화 방식)을 입력 언어로 바꿔 저장한다.
    """
    state = dict(vars(vocab))
    if lang is not None and "col" in state:
        state["col"] = lang
    return state


def _vocab_from_state(state):
    """저장용 dict -> Vocab 객체 (__init__ 을 거치지 않고 속성만 복원)"""
    vocab = Vocab.__new__(Vocab)
    vocab.__dict__.update(state)
    return vocab


# ---------------------------------------------------------------------
# 1. 제출용 6개 파일
# ---------------------------------------------------------------------
def save_direction(path, model, direction, src_vocab, tgt_vocab, model_kwargs):
    """MultiSeq2Seq 에서 한 방향만 꺼내 저장"""
    src_lang, tgt_lang = direction.split("2")
    torch.save(
        {
            "direction": direction,
            "encoder": model.encoder.state_dict(),
            "decoder": model.decoders[tgt_lang].state_dict(),
            "src_vocab": _vocab_state(src_vocab, src_lang),
            "tgt_vocab": _vocab_state(tgt_vocab),
            "model_config": {
                "src_vocab_size": len(src_vocab),
                "tgt_vocab_size": len(tgt_vocab),
                **model_kwargs,
            },
        },
        path,
    )


def save_all(out_dir, model, src_vocab, tgt_vocabs, model_kwargs, directions=DIRECTIONS):
    """6방향 모두 저장 -> 저장한 경로 리스트

    tgt_vocabs: {"num": Vocab, "eng": Vocab, "kor": Vocab}
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for direction in directions:
        _, tgt_lang = direction.split("2")
        path = os.path.join(out_dir, f"{direction}.pt")
        save_direction(path, model, direction, src_vocab, tgt_vocabs[tgt_lang], model_kwargs)
        paths.append(path)
    return paths


def load_model(path, device="cpu"):
    """.pt 하나 -> (Seq2Seq, src_vocab, tgt_vocab)

    저장된 config 로 같은 크기의 Encoder / Decoder 를 먼저 만든 뒤 가중치를 넣는다.
    돌려주는 모델은 eval 모드.
    """
    ckpt = torch.load(path, map_location=device)   # GPU 에서 저장한 파일도 CPU 에서 로드 가능

    config = dict(ckpt["model_config"])
    src_vocab_size = config.pop("src_vocab_size")
    tgt_vocab_size = config.pop("tgt_vocab_size")

    encoder = Encoder(src_vocab_size, **config)
    decoder = Decoder(tgt_vocab_size, **config)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])

    model = Seq2Seq(encoder, decoder).to(device)
    model.eval()
    return model, _vocab_from_state(ckpt["src_vocab"]), _vocab_from_state(ckpt["tgt_vocab"])


# ---------------------------------------------------------------------
# 2. 학습 중단 대비
# ---------------------------------------------------------------------
def save_training_state(path, model, optimizer, epoch, best_score=None):
    """학습을 이어서 할 수 있게 모델 + optimizer + epoch 저장"""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
        },
        path,
    )


def load_training_state(path, model, optimizer=None, device="cpu"):
    """저장한 학습 상태를 model, optimizer 에 넣고 (epoch, best_score) 반환"""
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    return state["epoch"], state["best_score"]