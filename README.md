# 2026-nlp

2026-2 자연어처리 팀 프로젝트

## 팀원

| 역할 | 이름 | 담당 |
|------|------|------|
| 팀장 | 홍세원 | `seq2seq.py`, `checkpoint.py`, `train.ipynb` |
| 팀원 | 박기수 | `model.py` |
| 팀원 | 이성현 | `train.py` |
| 팀원 | 이재후 | `data.py` |
| 팀원 | 정상호 | `inference.py` |

## 프로젝트

| 폴더 | 내용 | 마감 |
|------|------|------|
| [project01](./project01) | Seq2Seq Num2Text — 숫자·영어·한글 6방향 번역 (공유 Encoder + 언어별 Decoder, GRU, Beam Search) | 2026-11-24 23:59 |

## Project 01 모델

```
  num ─┐                           ┌─▶ Decoder[num] ──▶ num
  kor ─┼─▶ 공유 Encoder ── hidden ──┼─▶ Decoder[kor] ──▶ kor
  eng ─┘   (중간 표현)             └─▶ Decoder[eng] ──▶ eng
```

- Encoder: 통합 입력 vocab → 2층 양방향 GRU (정방향·역방향 합산)
- Decoder: 출력 언어별 3개, 2층 단방향 GRU + fc
- attention 미사용, 디코딩은 beam search
- 6방향을 섞어 한 번 학습하고, 방향별로 `.pt` 6개 저장 (공유 Encoder + 해당 Decoder)
- 데이터: num↔kor, num↔eng는 각 파일 전체, kor↔eng는 두 파일의 공통 숫자 12,738쌍
- 분할: 숫자 기준 8:1:1 (seed 42), 6방향 공통

## Project 01 구조

```
project01/            제출용 (최종 노트북 + 모델 파일 6개)
project01-dev/        작업용
├── data/             num2eng.txt, num2kor.txt (src·notebook 공용)
├── src/              로컬 실행용: main(진입점) / data / model / seq2seq / train / inference / checkpoint, config.json(설정)
├── notebook/         Colab 실행용: train.ipynb 하나에 src 코드를 모두 옮겨 놓음
├── tests/            파트별 테스트
└── checkpoints/      로컬 학습 결과 (last.pt·last.best.pt: 6 epoch 학습 상태, final/: 제출용 .pt 6개)
```

`src/`와 `notebook/`은 서로 의존하지 않는다. 한쪽 폴더만 있어도 실행된다.
`src/`의 코드를 고치면 `notebook/train.ipynb`의 해당 셀도 같이 고친다.

## 실행 (로컬)

환경 설치는 `requirements.txt` 위쪽 주석 참고 (torch는 CUDA 인덱스에서 먼저 설치).

```
cd project01-dev
python src/main.py                        # config.json의 epochs만큼 학습 (GPU가 있으면 GPU, 없으면 CPU)
python src/main.py --limit 96 --epochs 2  # 일부 데이터로 빠르게 동작 확인
python src/main.py --fresh                # 저장된 지점을 버리고 처음부터
```

- 다시 실행하면 `checkpoints/last.pt`에서 이어서 학습한다.
- 제출용 `.pt` 6개는 `checkpoints/final/`에 저장된다.

테스트:

```
python src/data.py
python tests/test_seq2seq_checkpoint.py
python -m pytest tests/test_model.py -q
```

Colab에서 테스트만 돌릴 때:

```python
!git clone https://github.com/sloth2003/2026-nlp.git /content/2026-nlp 2>/dev/null || git -C /content/2026-nlp pull
%cd /content/2026-nlp/project01-dev
!python tests/test_seq2seq_checkpoint.py
```

## 실행 (Colab)

1. `project01-dev/notebook/train.ipynb` 파일 하나만 Colab에 올려서 연다. (저장소 clone 불필요)
2. 런타임 유형을 T4 GPU로 바꾸고 위에서부터 순서대로 실행한다.
   - 데이터는 GitHub에서 자동으로 내려받는다.
   - 학습 셀을 실행할 때마다 Drive의 `last.pt`에서 이어서 `EPOCHS_PER_RUN`만큼 더 학습한다. (세션이 끊겨도 처음부터 다시 실행하면 이어진다)
   - 제출용 `.pt` 6개는 Drive `MyDrive/2026-nlp/checkpoints/final/`에 저장된다.
   - 마지막 셀은 저장한 `.pt`만으로 6방향 번역을 확인하고, `방향 문장` 형식으로 직접 입력해 볼 수 있다.