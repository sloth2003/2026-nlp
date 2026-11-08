# 2026-nlp

2026-2 자연어처리 팀 프로젝트

## 팀원

| 역할 | 이름 | 담당 |
|------|------|------|
| 팀장 | 홍세원 | `attention.py` |
| 팀원 | 박기수 | `model.py` |
| 팀원 | 이성현 | `train.py` |
| 팀원 | 이재후 | `data.py` |
| 팀원 | 정상호 | `inference.py` |

## 프로젝트

| 폴더 | 내용 | 마감 |
|------|------|------|
| [project01](./project01) | Seq2Seq Num2Text — 숫자·영어·한글 6방향 번역 (GRU + Attention + Beam Search) | 2026-11-24 23:59 |

## Project 01 구조

```
project01/            제출용 (최종 노트북 + 모델 파일 6개)
project01-dev/        작업용
├── config/           분할 비율, seed, batch size
├── data/             num2eng.txt, num2kor.txt
├── src/              data / model / attention / train / inference
└── notebook/         Colab 실행용 노트북
```

## 실행 (Colab)

```python
!git clone https://github.com/sloth2003/2026-nlp.git /content/2026-nlp
import sys
sys.path.append('/content/2026-nlp/project01-dev/src')
```

로컬 환경은 `requirements.txt` 참고
