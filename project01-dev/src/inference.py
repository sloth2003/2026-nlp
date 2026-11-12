"""
src/inference.py  (담당: 정상호)

추론 · 평가 모듈
  - greedy_decode : 빠른 평가용 배치 greedy 디코딩
  - beam_search   : 배치 beam search 디코딩 (최종 추론용)
  - translate     : 문자열(또는 문자열 리스트) → 번역 문자열
  - exact_match / evaluate : 문장 단위 정확 일치율

사용하는 모델 인터페이스 (attention.py):
  enc_output, hidden, x_mask = model.encode(x)                              # x: (B, Tv)
  logits, hidden, w = model.decode_step(token, hidden, enc_output, x_mask)  # token: (B, 1)
    - hidden: (num_layers, B, H)  → 배치 차원은 dim 1
    - enc_output, x_mask: 배치 차원은 dim 0
    - logits: (B, 1, V) 또는 (B, V)

모든 공개 함수는 내부에서 model.eval() + torch.no_grad()로 실행하고,
끝나면 원래 모드(train/eval)로 되돌린다. → train 루프 중 valid 평가에 그대로 써도 됨.
"""
import functools

import torch
import torch.nn.functional as F

PAD, BOS, EOS, UNK = 0, 1, 2, 3


# ---------------------------------------------------------------- 내부 유틸
def _eval_mode(fn):
    """eval 모드 + no_grad로 실행하고, 끝나면 원래 모드로 복구"""
    @functools.wraps(fn)
    def wrapper(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                return fn(model, *args, **kwargs)
        finally:
            model.train(was_training)
    return wrapper


def _device(model):
    return next(model.parameters()).device


def _step_logits(logits):
    """(B, 1, V) 또는 (B, V) → (B, V)"""
    return logits.reshape(logits.size(0), -1)


def _select_hidden(hidden, idx):
    """hidden (num_layers, B, H)을 배치 차원(dim 1)에서 idx 순서로 재배열"""
    if isinstance(hidden, tuple):  # LSTM으로 바뀌어도 동작하도록
        return tuple(h.index_select(1, idx) for h in hidden)
    return hidden.index_select(1, idx)


def strip_special(ids):
    """BOS·PAD 제거, EOS에서 자르기"""
    out = []
    for t in ids:
        if t == EOS:
            break
        if t in (BOS, PAD):
            continue
        out.append(t)
    return out


# ---------------------------------------------------------------- greedy
@_eval_mode
def greedy_decode(model, src, max_len=60):
    """
    src: (B, Tv) LongTensor (앞뒤 <bos>, <eos>, 패딩 포함 — 학습 배치와 같은 형식)
    반환: 길이 B 리스트, 각 원소는 토큰 id 리스트 (<bos>/<eos> 제외)
    """
    device = _device(model)
    src = src.to(device)
    B = src.size(0)

    enc_output, hidden, x_mask = model.encode(src)
    token = torch.full((B, 1), BOS, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    outputs = []

    for _ in range(max_len):
        logits, hidden, _ = model.decode_step(token, hidden, enc_output, x_mask)
        next_tok = _step_logits(logits).argmax(-1)          # (B,)
        next_tok = next_tok.masked_fill(finished, EOS)      # 끝난 문장은 EOS 유지
        outputs.append(next_tok)
        finished |= next_tok == EOS
        if finished.all():
            break
        token = next_tok.unsqueeze(1)

    seqs = torch.stack(outputs, dim=1).tolist()
    return [strip_special(s) for s in seqs]


# ---------------------------------------------------------------- beam search
@_eval_mode
def beam_search(model, src, beam_width=5, max_len=60, alpha=0.0, return_all=False):
    """
    배치 단위 beam search. 샘플마다 K개 빔을 (B*K) 배치로 펼쳐서 한 번에 디코딩한다.

    src        : (B, Tv) LongTensor
    beam_width : K
    alpha      : 길이 정규화 지수. 최종 점수 = logP / len**alpha (0이면 정규화 없음)
    return_all : True면 샘플마다 [(ids, score), ...] K개를 점수순으로 반환

    반환: 기본은 길이 B 리스트, 각 원소는 최고 점수 빔의 토큰 id 리스트 (<bos>/<eos> 제외)
    """
    device = _device(model)
    src = src.to(device)
    B, K = src.size(0), beam_width

    enc_output, hidden, x_mask = model.encode(src)

    # 샘플마다 K개로 복제: [s0, s0, ..., s1, s1, ...] → (B*K, ...)
    rep = torch.arange(B, device=device).repeat_interleave(K)
    enc_output = enc_output.index_select(0, rep)
    x_mask = x_mask.index_select(0, rep)
    hidden = _select_hidden(hidden, rep)

    # 처음엔 빔 0만 살아 있게 해서 K개 빔이 같은 토큰으로 중복 시작하는 것을 막는다
    scores = torch.full((B, K), float("-inf"), device=device)
    scores[:, 0] = 0.0

    token = torch.full((B * K, 1), BOS, dtype=torch.long, device=device)
    seqs = torch.empty((B * K, 0), dtype=torch.long, device=device)
    finished = torch.zeros(B * K, dtype=torch.bool, device=device)
    lengths = torch.zeros(B * K, device=device)            # EOS 포함 생성 길이
    batch_offset = (torch.arange(B, device=device) * K).unsqueeze(1)  # (B, 1)

    for _ in range(max_len):
        logits, hidden, _ = model.decode_step(token, hidden, enc_output, x_mask)
        logp = F.log_softmax(_step_logits(logits).float(), dim=-1)    # (B*K, V)
        V = logp.size(-1)

        # 끝난 빔은 EOS만 log 1 = 0 → 점수가 그대로 유지되고 계속 EOS만 붙는다
        if finished.any():
            logp[finished] = float("-inf")
            logp[finished, EOS] = 0.0

        cand = (scores.view(-1, 1) + logp).view(B, K * V)
        scores, top_idx = cand.topk(K, dim=-1)               # (B, K)
        beam_idx = top_idx // V                               # 어느 부모 빔에서 왔는지
        tok = top_idx % V                                     # 새로 붙는 토큰

        flat = (batch_offset + beam_idx).view(-1)             # (B*K,) 부모 빔의 펼친 인덱스
        seqs = torch.cat([seqs.index_select(0, flat), tok.view(-1, 1)], dim=1)
        hidden = _select_hidden(hidden, flat)

        prev_finished = finished.index_select(0, flat)
        lengths = lengths.index_select(0, flat) + (~prev_finished).float()
        finished = prev_finished | (tok.view(-1) == EOS) | torch.isinf(scores.view(-1))

        if finished.all():
            break
        token = tok.view(-1, 1)

    final = scores.view(-1)
    if alpha > 0:
        final = final / lengths.clamp(min=1).pow(alpha)
    final = final.view(B, K)
    order = final.argsort(dim=-1, descending=True)

    seqs = seqs.view(B, K, -1).tolist()
    final = final.tolist()
    order = order.tolist()

    if return_all:
        return [[(strip_special(seqs[b][k]), final[b][k]) for k in order[b]] for b in range(B)]
    return [strip_special(seqs[b][order[b][0]]) for b in range(B)]


def decode_batch(model, src, method="beam", beam_width=5, max_len=60, alpha=0.0):
    """method에 따라 greedy / beam 중 하나로 디코딩"""
    if method == "greedy":
        return greedy_decode(model, src, max_len=max_len)
    if method == "beam":
        return beam_search(model, src, beam_width=beam_width, max_len=max_len, alpha=alpha)
    raise ValueError(f"method는 'greedy' 또는 'beam': {method}")


# ---------------------------------------------------------------- 문자열 번역
def _texts_to_tensor(texts, src_vocab, device):
    """문자열 리스트 → 학습 배치와 같은 형식의 (B, Tv) 텐서 (앞뒤 <bos> <eos>, 오른쪽 패딩)"""
    rows = []
    for text in texts:
        ids = list(src_vocab.encode(text))
        # Vocab.encode가 <bos>/<eos>를 붙이는지와 무관하게 동작하도록
        if not ids or ids[0] != BOS:
            ids = [BOS] + ids
        if ids[-1] != EOS:
            ids = ids + [EOS]
        rows.append(ids)
    T = max(len(r) for r in rows)
    padded = [r + [PAD] * (T - len(r)) for r in rows]
    return torch.tensor(padded, dtype=torch.long, device=device)


def translate(model, texts, src_vocab, tgt_vocab, method="beam",
              beam_width=5, max_len=60, alpha=0.0):
    """
    texts: 문자열 하나 또는 문자열 리스트 (숫자도 문자열로! 예: "12345")
    반환: 입력이 문자열이면 문자열, 리스트면 리스트
    """
    single = isinstance(texts, str)
    if single:
        texts = [texts]
    src = _texts_to_tensor(texts, src_vocab, _device(model))
    preds = decode_batch(model, src, method, beam_width, max_len, alpha)
    outs = [tgt_vocab.decode(p) for p in preds]
    return outs[0] if single else outs


# ---------------------------------------------------------------- 평가
def exact_match(hyps, refs):
    """문장 단위 정확 일치율 (앞뒤 공백은 무시)"""
    assert len(hyps) == len(refs)
    if not refs:
        return 0.0
    return sum(h.strip() == r.strip() for h, r in zip(hyps, refs)) / len(refs)


def evaluate(model, loader, tgt_vocab, method="greedy", beam_width=5, max_len=60,
             alpha=0.0, max_batches=None, src_vocab=None, n_errors=5):
    """
    loader가 (src, tgt) 배치를 내보낸다고 가정 (data.py 인터페이스).
    train 중 valid에는 method="greedy"(+max_batches로 일부만), 최종 test에는 "beam" 권장.

    반환: dict(acc, correct, total, errors)
      errors: 틀린 예시 최대 n_errors개 [(src_text, ref, hyp)] — src_vocab을 넘기면 src_text 채움
    """
    correct = total = 0
    errors = []
    for i, (src, tgt) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        preds = decode_batch(model, src, method, beam_width, max_len, alpha)
        for j, (p, t) in enumerate(zip(preds, tgt.tolist())):
            hyp = tgt_vocab.decode(p)
            ref = tgt_vocab.decode(strip_special(t))
            if hyp.strip() == ref.strip():
                correct += 1
            elif len(errors) < n_errors:
                s = src_vocab.decode(strip_special(src[j].tolist())) if src_vocab else None
                errors.append((s, ref, hyp))
            total += 1
    return {"acc": correct / max(total, 1), "correct": correct, "total": total, "errors": errors}
