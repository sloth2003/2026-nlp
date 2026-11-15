import random
from pathlib import Path

import torch
import torch.nn.functional as F

if __name__ != "__main__":
    from seq2seq import DIRECTIONS
    from inference import greedy_decode, strip_special
    from checkpoint import save_training_state

PAD_ID = 0


def batch_loss(target, logits):
    """이미 한 칸 맞춘 정답 (B,T)과 예측 점수 (B,T,V)의 평균 loss."""
    if logits.ndim != 3 or target.ndim != 2 or logits.shape[:2] != target.shape:
        raise ValueError("정답은 (B,T), 예측 점수는 (B,T,V)여야 합니다.")
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=PAD_ID,
    )


def _batch_result(model, x, y, tgt_lang):
    logits = model(x, y, tgt_lang)
    vocab_size = model.decoders[tgt_lang].embedding.num_embeddings
    if logits.ndim != 3 or logits.size(-1) != vocab_size:
        raise ValueError(
            f"{tgt_lang} Decoder 출력 {tuple(logits.shape)}: 마지막 차원은 출력 사전 "
            f"크기 {vocab_size}여야 합니다. model.py의 fc 출력층을 확인하세요."
        )
    target = y[:, 1:]
    loss = batch_loss(target, logits[:, :-1, :])
    if not torch.isfinite(loss):
        raise FloatingPointError("loss가 NaN/Inf입니다. 입력과 비패딩 정답을 확인하세요.")
    return loss, target.ne(PAD_ID).sum().item()


def mixed_batches(loaders, seed):
    """방향 이름만 섞고, 각 loader의 배치는 한 번씩 꺼낸다."""
    order = []
    streams = {}
    for direction in DIRECTIONS:
        loader = loaders[direction]["train"]
        if len(loader) == 0:
            raise ValueError(f"{direction} 학습 배치가 비어 있습니다.")
        order.extend([direction] * len(loader))
        streams[direction] = iter(loader)

    random.Random(seed).shuffle(order)
    for direction in order:
        x, y = next(streams[direction])
        yield direction, x, y


def train_epoch(model, loaders, optimizer, device, seed, grad_clip=1.0):
    """하나의 optimizer로 6방향을 한 바퀴 학습한다."""
    model.train()
    loss_sum, token_count = 0.0, 0
    for direction, x, y in mixed_batches(loaders, seed):
        tgt_lang = direction.split("2")[1]
        optimizer.zero_grad(set_to_none=True)
        loss, count = _batch_result(model, x.to(device), y.to(device), tgt_lang)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
        optimizer.step()
        loss_sum += loss.item() * count
        token_count += count
    return loss_sum / token_count


@torch.no_grad()
def validate(model, loaders, device, max_len=60):
    """방향별 valid loss와 greedy로 전체 토큰열이 일치한 비율을 반환한다."""
    model.eval()
    by_direction = {}
    for direction in DIRECTIONS:
        tgt_lang = direction.split("2")[1]
        one_model = model.direction(direction)
        loss_sum, token_count, correct, examples = 0.0, 0, 0, 0
        for x, y in loaders[direction]["valid"]:
            x, y = x.to(device), y.to(device)
            loss, count = _batch_result(model, x, y, tgt_lang)
            loss_sum += loss.item() * count
            token_count += count

            predictions = greedy_decode(one_model, x, max_len=max_len)
            for prediction, gold in zip(predictions, y.tolist()):
                correct += prediction == strip_special(gold)
            examples += y.size(0)
        if examples == 0:
            raise ValueError(f"{direction} 검증 배치가 비어 있습니다.")
        by_direction[direction] = {
            "loss": loss_sum / token_count, "greedy_exact": correct / examples,
            "correct": correct, "examples": examples,
        }
    return by_direction


def _save_epoch(path, model, optimizer, epoch, best_loss):
    # 저장 중 중단돼도 직전까지 완성된 파일을 읽을 수 있게 한다.
    tmp_path = path.with_name(path.name + ".tmp")
    save_training_state(tmp_path, model, optimizer, epoch, best_score=best_loss)
    tmp_path.replace(path)


def train(model, loaders, optimizer, config, device=None, start_epoch=0, ckpt_path=None):
    """노트북 진입점. 매 epoch의 방향별 검증 기록을 리스트로 반환한다.

    start_epoch는 0부터 세며, 재개 시 노트북이 모델과 optimizer를 먼저 복원한다.
    config의 epochs는 목표 총 횟수(기본 30). best 기준은 여섯 방향 valid loss의 평균.
    """
    epochs = config.get("epochs", 30)
    seed = config.get("random_seed", 42)
    grad_clip = config.get("grad_clip", 1.0)
    max_len = config.get("max_len", 60)
    if epochs < 1 or start_epoch < 0 or grad_clip <= 0 or max_len < 1:
        raise ValueError("epochs, grad_clip, max_len은 양수이고 start_epoch는 0 이상이어야 합니다.")
    device = next(model.parameters()).device if device is None else device
    history = []
    best_loss = float("inf")
    best_weights = None

    if ckpt_path is not None:
        ckpt_path = Path(ckpt_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        best_path = ckpt_path.with_name(ckpt_path.stem + ".best" + ckpt_path.suffix)
        if start_epoch > 0 and best_path.is_file():
            previous = torch.load(best_path, map_location="cpu", weights_only=True)
            best_loss, best_weights = previous["best_score"], previous["model"]
            del previous

    for epoch in range(start_epoch, epochs):
        train_loss = train_epoch(model, loaders, optimizer, device, seed + epoch, grad_clip)
        valid = validate(model, loaders, device, max_len)
        mean_valid_loss = sum(row["loss"] for row in valid.values()) / len(valid)
        history.append({"epoch": epoch + 1, "train_loss": train_loss,
                        "valid_loss": mean_valid_loss, "valid": valid})

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            best_weights = {key: value.detach().cpu().clone()
                            for key, value in model.state_dict().items()}
            if ckpt_path is not None:
                _save_epoch(best_path, model, optimizer, epoch, best_loss)
        if ckpt_path is not None:
            _save_epoch(ckpt_path, model, optimizer, epoch, best_loss)

        print(f"epoch {epoch + 1}/{epochs} | train loss {train_loss:.4f} | valid loss {mean_valid_loss:.4f}")
        for direction, row in valid.items():
            print(f"  {direction}: loss {row['loss']:.4f}, greedy exact {row['greedy_exact']:.1%}")

    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval()
    return history
