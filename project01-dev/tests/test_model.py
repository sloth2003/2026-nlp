"""
model.py 단독 / seq2seq.py 통합 테스트 (담당: 기수)

실행 (저장소 루트):
    python -m pytest project01-dev/tests/test_model.py -q

실행 (Colab):
    %cd /content/2026-nlp/project01-dev
    !python tests/test_model.py

확인하는 것 (CPU 필수, CUDA 사용 가능 시 함께 실행)
1. 양방향 Encoder 위치 정렬, 방향 합산, 출력 모양과 PAD 제외 hidden
2. 출력 언어 3개의 Decoder logits / hidden 모양과 한 토큰 디코딩
3. 공유 Encoder 1개, Decoder 별 독립 파라미터
4. 6방향 선택, hidden 전달, eval 모드 추론
5. 정답 시프트와 Cross-Entropy 역전파
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model import Encoder, Decoder
from seq2seq import build_model, DIRECTIONS
from data import Vocab, combined

# 실제 data.py 가 생성하는 출력 언어별 vocab 크기를 사용한다.
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
VOCABS = {lang: len(Vocab(combined[lang].dropna(), lang))
          for lang in ("num", "eng", "kor")}


@pytest.fixture(autouse=True)
def full_precision_comparisons():
    # CUDA 에서 배치 크기에 따른 TF32 오차를 피하고, 종료 후 원래 설정을 복원한다.
    previous = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = previous


# ---------------------------------------------------------------------
# 1. Encoder: 출력 모양 / padding
# ---------------------------------------------------------------------
@pytest.mark.parametrize("device", DEVICES)
def test_encoder_padding(device):
    encoder = Encoder(70).to(device).eval()
    assert encoder.gru.bidirectional
    x = torch.tensor([[1, 4, 2, 0, 0], [1, 5, 6, 2, 0]], device=device)
    with torch.no_grad():
        raw_results = []
        hook = encoder.gru.register_forward_hook(
            lambda module, args, result: raw_results.append(result))
        try:
            output, hidden = encoder(x)
        finally:
            hook.remove()
        raw_output, raw_hidden = raw_results[0]
        raw_output, _ = pad_packed_sequence(raw_output, batch_first=True, total_length=5)
        assert raw_output.shape == (2, 5, 512)
        assert raw_hidden.shape == (4, 2, 256)
        torch.testing.assert_close(output, raw_output[:, :, :256] + raw_output[:, :, 256:])
        torch.testing.assert_close(hidden, raw_hidden[0::2] + raw_hidden[1::2])
        assert output.shape == (2, 5, 256)
        assert hidden.shape == (2, 2, 256)
        assert torch.count_nonzero(output[:, 4:]) == 0
        for i, length in enumerate((3, 4)):
            # 최상위 층: 정방향 최종 상태는 마지막 실제 토큰 위치에 있다.
            torch.testing.assert_close(raw_output[i, length - 1, :256], raw_hidden[-2, i])
            # 역방향은 첫 토큰까지 읽고 끝나므로 최종 상태가 원래 첫 위치에 있다.
            torch.testing.assert_close(raw_output[i, 0, 256:], raw_hidden[-1, i])
            unpadded_output, unpadded_hidden = encoder(x[i:i+1, :length])
            torch.testing.assert_close(output[i:i+1, :length], unpadded_output)
            torch.testing.assert_close(hidden[:, i:i+1], unpadded_hidden)


# ---------------------------------------------------------------------
# 2. Decoder: 출력 언어 3개 / padding / 한 토큰 입력
# ---------------------------------------------------------------------
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("lang", VOCABS)
def test_decoder_logits_padding_and_step(device, lang):
    decoder = Decoder(VOCABS[lang]).to(device).eval()
    assert not decoder.gru.bidirectional
    hidden = torch.randn(2, 2, 256, device=device)
    y = torch.tensor([[1, 4, 2, 0, 0], [1, 2, 0, 0, 0]], device=device)
    with torch.no_grad():
        logits, final_hidden = decoder(y, hidden)
        assert logits.shape == (2, 5, VOCABS[lang])
        assert final_hidden.shape == hidden.shape
        # Unpacking 의 PAD 위치는 0 이므로 FC 이후에는 bias 와 같다.
        torch.testing.assert_close(logits[:, 3:], decoder.fc.bias.expand(2, 2, -1))
        step_hidden = hidden
        for t in range(2):
            step_logits, step_hidden = decoder(y[:, t:t+1], step_hidden)
            torch.testing.assert_close(step_logits, logits[:, t:t+1])
        for token in (torch.ones(2, 1, dtype=torch.long, device=device),
                      torch.zeros(2, 1, dtype=torch.long, device=device)):
            step_logits, step_hidden = decoder(token, hidden)
            assert step_logits.shape == (2, 1, VOCABS[lang])
            assert torch.isfinite(step_hidden).all()


# ---------------------------------------------------------------------
# 3. Seq2Seq 통합: 파라미터 공유 / 6방향 추론 / 역전파
# ---------------------------------------------------------------------
@pytest.mark.parametrize("device", DEVICES)
def test_shared_model_six_directions_and_cross_entropy(device):
    model = build_model(70, VOCABS).to(device)
    assert sum(isinstance(module, Encoder) for module in model.modules()) == 1
    parameter_sets = [{id(p) for p in decoder.parameters()}
                      for decoder in model.decoders.values()]
    for i in range(3):
        for j in range(i):
            assert parameter_sets[i].isdisjoint(parameter_sets[j])

    x = torch.tensor([[1, 4, 2, 0, 0], [1, 5, 6, 2, 0]], device=device)
    y = torch.tensor([[1, 4, 2, 0], [1, 5, 6, 2]], device=device)
    model.eval()
    with torch.no_grad():
        for direction in DIRECTIONS:
            lang = direction.split("2")[1]
            seq = model.direction(direction)
            assert seq.encoder is model.encoder
            assert seq.decoder is model.decoders[lang]
            assert not seq.training and not seq.decoder.training
            enc_output, hidden, mask = seq.encode(x)
            torch.testing.assert_close(mask, x != 0)
            received = []
            hook = seq.decoder.gru.register_forward_pre_hook(
                lambda module, args: received.append(args[1]))
            try:
                logits, next_hidden, attention = seq.decode_step(y[:, :1], hidden, enc_output, mask)
            finally:
                hook.remove()
            assert received[0] is hidden
            assert attention is None
            assert logits.shape == (2, 1, VOCABS[lang])
            assert next_hidden.shape == (2, 2, 256)
            torch.testing.assert_close(seq(x, y), model(x, y, lang))

    model.train()
    # 각 방향의 Loss 가 동일한 Encoder 와 선택한 Decoder 에 역전파되는지 확인한다.
    for direction in DIRECTIONS:
        lang = direction.split("2")[1]
        model.zero_grad(set_to_none=True)
        logits = model(x, y, lang)
        targets = y[:, 1:]
        predictions = logits[:, :-1, :]
        assert predictions.shape[:2] == targets.shape
        loss = F.cross_entropy(predictions.reshape(-1, VOCABS[lang]),
                               targets.reshape(-1), ignore_index=0)
        assert torch.isfinite(loss)
        loss.backward()
        for module in (model.encoder, model.decoders[lang]):
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in module.parameters())
            assert module.embedding.weight.grad.abs().sum() > 0
        # 역방향 GRU 파라미터에도 실제로 gradient 가 생기는지 확인한다.
        for name, parameter in model.encoder.gru.named_parameters():
            if "reverse" in name:
                assert parameter.grad.abs().sum() > 0
        for other_lang, decoder in model.decoders.items():
            if other_lang != lang:
                assert all(p.grad is None for p in decoder.parameters())


if __name__ == "__main__":
    # 직접 실행할 때도 같은 pytest 테스트와 CPU / CUDA 설정을 사용한다.
    raise SystemExit(pytest.main([__file__, "-q"]))
