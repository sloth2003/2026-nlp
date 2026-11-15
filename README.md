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
├── config/           분할 비율, seed, batch size
├── data/             num2eng.txt, num2kor.txt
├── src/              data / model / seq2seq / train / inference / checkpoint
├── tests/            파트별 테스트
└── notebook/         Colab 실행용 (train.ipynb: 학습·평가·저장)
```

## 실행 (Colab)

1. 런타임 유형을 T4 GPU로 바꾼다.
2. `project01-dev/notebook/train.ipynb`를 열고 위에서부터 순서대로 실행한다.
   - 학습 셀을 다시 실행하면 Drive의 `last.pt`에서 이어서 학습한다.
   - 제출용 `.pt` 6개는 Drive `MyDrive/2026-nlp/checkpoints/final/`에 저장된다.

테스트:

```python
!git clone https://github.com/sloth2003/2026-nlp.git /content/2026-nlp 2>/dev/null || git -C /content/2026-nlp pull
%cd /content/2026-nlp/project01-dev
!python tests/test_seq2seq_checkpoint.py
```

로컬 환경은 `requirements.txt` 참고