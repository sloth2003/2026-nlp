"""
seq2seq.py + checkpoint.py 테스트 (담당: 세원)

실행 (Colab):
    %cd /content/2026-nlp/project01-dev
    !python tests/test_seq2seq_checkpoint.py

확인하는 것
1. MultiSeq2Seq forward 출력 모양 (출력 언어 3개)
2. 역전파 시 Encoder 와 Decoder 3개 모두 gradient 가 생기는지
3. direction() 이 Encoder / Decoder 를 복사하지 않고 공유하는지
4. inference.py (상호) 의 greedy / beam / translate 와 연결되는지
5. save_all -> 6개 파일 -> load_model 로 같은 출력이 나오는지
6. 학습 상태 저장·로드
"""

import os
import sys
import tempfile

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import model as model_py
import seq2seq
import checkpoint
from data import Vocab
from inference import greedy_decode, beam_search, translate

# ---------------------------------------------------------------------
# model.py Decoder 에 fc 가 아직 없으면 테스트용으로 붙여서 사용
# (기수님이 fc 를 추가하면 이 부분은 자동으로 건너뜀)
# ---------------------------------------------------------------------
_probe = model_py.Decoder(10, embedding_dim=8, hidden_dim=8, num_layers=1)
_out, _ = _probe(torch.ones(1, 2, dtype=torch.long), torch.zeros(1, 1, 8))
if _out.size(-1) != 10:
    print("[알림] model.py Decoder 에 fc 가 없어 테스트용 fc 를 붙여서 진행")

    class _DecoderWithFC(model_py.Decoder):
        def __init__(self, vocab_size, hidden_dim=256, **kwargs):
            super().__init__(vocab_size, hidden_dim=hidden_dim, **kwargs)
            self.fc = nn.Linear(hidden_dim, vocab_size)

        def forward(self, y, hidden):
            dec_output, hidden = super().forward(y, hidden)
            return self.fc(dec_output), hidden

    seq2seq.Decoder = _DecoderWithFC
    checkpoint.Decoder = _DecoderWithFC


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 작은 vocab (실제 데이터 일부 형식)
    src_vocab = Vocab(["12708", "만 이천칠백팔", "twelve thousand seven hundred eight"], "num")
    tgt_vocabs = {
        "num": Vocab(["12708", "345"], "num"),
        "kor": Vocab(["만 이천칠백팔", "삼백사십오"], "kor"),
        "eng": Vocab(["twelve thousand seven hundred eight"], "eng"),
    }
    model_kwargs = dict(embedding_dim=16, hidden_dim=32, num_layers=2, dropout=0.1)

    model = seq2seq.build_model(
        len(src_vocab), {lang: len(v) for lang, v in tgt_vocabs.items()}, **model_kwargs
    ).to(device)

    B, Tv, Tq = 4, 7, 5
    x = torch.randint(4, len(src_vocab), (B, Tv), device=device)
    x[:, 0] = 1
    x[0, 5:] = 0                                    # PAD 포함

    # 1. forward 모양
    for lang, vocab in tgt_vocabs.items():
        y = torch.randint(4, len(vocab), (B, Tq), device=device)
        y[:, 0] = 1
        logits = model(x, y, lang)
        assert logits.shape == (B, Tq, len(vocab)), (lang, logits.shape)
    print("[OK] 1. forward 모양:", {l: len(v) for l, v in tgt_vocabs.items()})

    # 2. 역전파: 공유 Encoder + Decoder 3개 모두 gradient
    model.zero_grad()
    loss = 0
    for lang, vocab in tgt_vocabs.items():
        y = torch.randint(4, len(vocab), (B, Tq), device=device)
        loss = loss + model(x, y, lang).sum()
    loss.backward()
    assert all(p.grad is not None for p in model.encoder.parameters())
    for lang in tgt_vocabs:
        assert all(p.grad is not None for p in model.decoders[lang].parameters()), lang
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] 2. 역전파: Encoder + Decoder 3개 gradient 있음 (파라미터 {n_params:,}개)")

    # 3. direction() 은 공유
    d = model.direction("kor2eng")
    assert d.encoder is model.encoder and d.decoder is model.decoders["eng"]
    print("[OK] 3. direction(): Encoder / Decoder 공유")

    # 4. inference.py 연결
    model.eval()
    m = model.direction("num2kor")
    g = greedy_decode(m, x, max_len=10)
    b = beam_search(m, x, beam_width=3, max_len=10)
    t = translate(m, ["12708", "345"], src_vocab, tgt_vocabs["kor"], method="beam", beam_width=3, max_len=10)
    assert len(g) == B and len(b) == B and len(t) == 2
    print("[OK] 4. inference.py 연결 (학습 전이라 번역 결과는 의미 없음):", t)

    # 5. 저장 -> 로드 -> 같은 출력
    with tempfile.TemporaryDirectory() as out_dir:
        paths = checkpoint.save_all(out_dir, model, src_vocab, tgt_vocabs, model_kwargs)
        assert sorted(os.path.basename(p) for p in paths) == sorted(f"{d}.pt" for d in seq2seq.DIRECTIONS)

        for path in paths:
            direction = os.path.basename(path)[:-3]
            src_lang, tgt_lang = direction.split("2")
            loaded, sv, tv = checkpoint.load_model(path, device=device)

            y = torch.randint(4, len(tgt_vocabs[tgt_lang]), (B, Tq), device=device)
            y[:, 0] = 1
            with torch.no_grad():
                before = model.direction(direction)(x, y)
                after = loaded(x, y)
            assert torch.allclose(before, after, atol=1e-6), direction
            assert sv.itos == src_vocab.itos and tv.itos == tgt_vocabs[tgt_lang].itos
            assert sv.col == src_lang and tv.col == tgt_lang
        size_kb = os.path.getsize(paths[0]) / 1024
    print(f"[OK] 5. 6개 파일 저장·로드 후 출력 동일 (파일 하나 {size_kb:.0f} KB)")

    # 6. 학습 상태 저장·로드
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as out_dir:
        path = os.path.join(out_dir, "last.pt")
        checkpoint.save_training_state(path, model, optimizer, epoch=3, best_score=0.5)
        model2 = seq2seq.build_model(
            len(src_vocab), {lang: len(v) for lang, v in tgt_vocabs.items()}, **model_kwargs
        ).to(device)
        epoch, best = checkpoint.load_training_state(path, model2, torch.optim.Adam(model2.parameters()), device)
        assert epoch == 3 and best == 0.5
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)
    print("[OK] 6. 학습 상태 저장·로드")

    print("\nseq2seq.py + checkpoint.py 테스트 통과")


if __name__ == "__main__":
    main()