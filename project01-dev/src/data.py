# imports
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import json
from pathlib import Path
import random
from torch.nn.utils.rnn import pad_sequence

# set seed
def seed_everything(seed=42):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)  # CPU + GPU 모두 고정

# ship config
ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config" / "split.json", encoding="utf-8") as f:
    config = json.load(f)

# data 
num_kor = pd.read_csv(
    ROOT / 'data' / 'num2kor.txt',
    sep='\t',
    header=None,
    names=['num', 'kor']
)

num_eng = pd.read_csv(
    ROOT / 'data' / 'num2eng.txt',
    sep='\t',
    header=None,
    names=['num', 'eng']
)

# combined df
# num2kor / num2eng 는 서로 다른 숫자를 랜덤 샘플링한 것이라 숫자가 일부만 겹친다.
# outer merge 라서 한쪽에만 있는 숫자는 kor 또는 eng 가 NaN -> load_data 에서 쓰는 두 열 기준으로 걸러낸다.
combined = pd.merge(num_kor, num_eng, on='num', how='outer')
combined['num'] = combined['num'].astype('str')


# Vocab token <-> word
PAD, BOS, EOS, UNK = '<pad>', '<bos>', '<eos>', '<unk>'
SPECIALS = [PAD, BOS, EOS, UNK]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

def tokenize(text, col): 
    # 영어: 단어 단위 / 숫자·한국어: 글자 단위
    return text.split() if col == 'eng' else list(text)

class Vocab:
    def __init__(self, sentences, col):
        self.col = col
        tokens = sorted({t for s in sentences for t in tokenize(s, col)})
        self.itos = SPECIALS + tokens                        # id → 토큰 (index를 id로 삼음)
        self.stoi = {t: i for i, t in enumerate(self.itos)}  # 토큰 → id

    def __len__(self):
        return len(self.itos)

    def encode(self, text):
        ids = [self.stoi.get(t, UNK_ID) for t in tokenize(text, self.col)]  # 모르는 토큰 → <unk>
        return [BOS_ID] + ids + [EOS_ID]

    def decode(self, ids):
        toks = []
        for i in ids:
            if i == EOS_ID:              # <eos> 이후는 버림
                break
            if i in (PAD_ID, BOS_ID):
                continue
            toks.append(self.itos[i])    # <unk>는 그대로 보이게 둠
        return ' '.join(toks) if self.col == 'eng' else ''.join(toks)


# type for indexing data
class NumDataset(Dataset):
    def __init__(self, df, input_col, output_col, src_vocab, tgt_vocab):
        # df        : DataFrame
        # input_col : 입력 열 이름 (예: 'num')
        # output_col: 정답 열 이름 (예: 'kor')
        # src_vocab : 입력 언어 사전 / tgt_vocab: 출력 언어 사전
        self.src = [torch.tensor(src_vocab.encode(s)) for s in df[input_col]]
        self.tgt = [torch.tensor(tgt_vocab.encode(s)) for s in df[output_col]]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]


def collate_fn(batch):
    src, tgt = zip(*batch)
    src = pad_sequence(src, batch_first=True, padding_value=PAD_ID)
    tgt = pad_sequence(tgt, batch_first=True, padding_value=PAD_ID)
    return src, tgt


# data loader
def pair_df(df, input_col, output_col):
    # 입력·정답 열이 둘 다 있는 행만 남김 (한쪽 파일에만 있는 숫자 제외)
    return df.dropna(subset=[input_col, output_col]).reset_index(drop=True)


def load_data(input_col, output_col, df, config):
    df = pair_df(df, input_col, output_col)
    src_vocab = Vocab(df[input_col], input_col)
    tgt_vocab = Vocab(df[output_col], output_col)
    dataset = NumDataset(df, input_col, output_col, src_vocab, tgt_vocab)

    g = torch.Generator().manual_seed(config['random_seed'])
    train_set, valid_set, test_set = random_split(
        dataset,
        [config['train_ratio'], config['valid_ratio'], config['test_ratio']],
        generator=g
    )

    bs = config['batch_size']
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True,  collate_fn=collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=bs, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_set,  batch_size=bs, shuffle=False, collate_fn=collate_fn)
    return train_loader, valid_loader, test_loader, src_vocab, tgt_vocab

"""
***사용법***

seed_everything(config['random_seed'])
train_loader, valid_loader, test_loader, src_vocab, tgt_vocab = load_data('num', 'eng', combined, config)

src, tgt = next(iter(train_loader))
print(src.shape, tgt.shape)                 # (32, 12), (32, 최대길이) -> (batch, vocab)
print(tgt_vocab.decode(tgt[0].tolist()))    # 원래 문장으로 복원되는지 확인
"""

# test
if __name__ == '__main__':
    # 1. config
    for key in ['train_ratio', 'valid_ratio', 'test_ratio', 'random_seed', 'batch_size']:
        assert key in config, f'config에 {key} 없음'
    assert abs(config['train_ratio'] + config['valid_ratio'] + config['test_ratio'] - 1) < 1e-6, '비율 합이 1이 아님'
    print('[OK] config')

    # 2. 원본 데이터
    # num2kor / num2eng 의 숫자가 서로 다르므로 combined 의 NaN 은 정상. 파일별로 따로 검사한다.
    for name, df in [('num2kor', num_kor), ('num2eng', num_eng)]:
        assert df.isna().sum().sum() == 0, f'{name}에 빈 값 있음'
        assert df['num'].is_unique, f'{name}에 중복된 숫자 있음'
    has_kor, has_eng = combined['kor'].notna(), combined['eng'].notna()
    n_both = int((has_kor & has_eng).sum())
    assert combined['num'].is_unique, 'merge 후 중복된 숫자 있음'
    assert (has_kor | has_eng).all(), 'kor / eng 둘 다 없는 행 있음'
    assert has_kor.sum() == len(num_kor) and has_eng.sum() == len(num_eng), 'merge 중 행이 사라지거나 늘어남'
    assert len(combined) == len(num_kor) + len(num_eng) - n_both, 'merge 후 행 개수 이상'
    print(f'[OK] data: kor {len(num_kor)}, eng {len(num_eng)}, 둘 다 있는 숫자 {n_both}, combined {combined.shape}')

    # 3. tokenize
    assert tokenize('21546', 'num') == ['2', '1', '5', '4', '6']
    assert tokenize('twenty one', 'eng') == ['twenty', 'one']
    assert tokenize('이만 천', 'kor') == ['이', '만', ' ', '천']
    print('[OK] tokenize')

    for out_col in ['eng', 'kor']:
        print(f'\n===== num -> {out_col} =====')

        # 이 방향에서 실제로 쓰는 행 (num 과 out_col 이 둘 다 있는 행)
        pair = pair_df(combined, 'num', out_col)
        assert len(pair) == combined[out_col].notna().sum() and pair.isna()[['num', out_col]].sum().sum() == 0

        # 4. Vocab
        src_vocab = Vocab(pair['num'], 'num')
        tgt_vocab = Vocab(pair[out_col], out_col)
        assert src_vocab.itos[:4] == SPECIALS and tgt_vocab.itos[:4] == SPECIALS, '특수 토큰 id가 0~3이 아님'
        assert len(src_vocab) == 4 + 10, f'숫자 어휘 크기 이상: {len(src_vocab)}'
        assert src_vocab.encode('12') == [BOS_ID, src_vocab.stoi['1'], src_vocab.stoi['2'], EOS_ID]
        assert src_vocab.encode('1a')[2] == UNK_ID, '모르는 토큰이 <unk>로 안 바뀜'
        assert tgt_vocab.decode([BOS_ID, tgt_vocab.stoi[tgt_vocab.itos[4]], EOS_ID, 5, 6]) == tgt_vocab.itos[4], '<eos> 이후가 안 잘림'
        for s in pair[out_col]:   # 전체 문장 encode -> decode 복원 확인
            assert tgt_vocab.decode(tgt_vocab.encode(s)) == s, f'복원 실패: {s!r}'
        print(f'[OK] vocab: src {len(src_vocab)}, tgt {len(tgt_vocab)}')

        # 5. DataLoader
        seed_everything(config['random_seed'])
        train_loader, valid_loader, test_loader, src_vocab, tgt_vocab = load_data('num', out_col, combined, config)

        n = len(pair)   # combined 전체가 아니라 NaN 을 뺀 행 수
        sizes = [len(l.dataset) for l in (train_loader, valid_loader, test_loader)]
        assert sum(sizes) == n, '분할 후 개수 합이 안 맞음'
        idx = [set(l.dataset.indices) for l in (train_loader, valid_loader, test_loader)]
        assert not (idx[0] & idx[1] or idx[0] & idx[2] or idx[1] & idx[2]), 'train/valid/test가 겹침'
        print(f'[OK] split: train {sizes[0]}, valid {sizes[1]}, test {sizes[2]}')

        # 같은 시드면 같은 분할인지
        again = load_data('num', out_col, combined, config)[0]
        assert again.dataset.indices == train_loader.dataset.indices, '시드가 같은데 분할이 다름'
        print('[OK] split 재현성')

        src, tgt = next(iter(train_loader))
        assert src.dtype == torch.long and tgt.dtype == torch.long, 'dtype이 long이 아님 (Embedding 입력 불가)'
        assert src.shape[0] == config['batch_size'] and tgt.shape[0] == config['batch_size']
        assert (src[:, 0] == BOS_ID).all() and (tgt[:, 0] == BOS_ID).all(), '맨 앞이 <bos>가 아님'
        for row in tgt:   # <eos> 뒤는 전부 <pad>
            eos_pos = (row == EOS_ID).nonzero()[0].item()
            assert (row[eos_pos + 1:] == PAD_ID).all(), '<eos> 뒤에 pad가 아닌 값 있음'
        print(f'[OK] batch: src {tuple(src.shape)}, tgt {tuple(tgt.shape)}')

        # 배치 -> 원래 문장 복원
        orig = pair.iloc[train_loader.dataset.indices]
        lookup = dict(zip(orig['num'], orig[out_col]))
        for s_ids, t_ids in zip(src, tgt):
            num = src_vocab.decode(s_ids.tolist())
            assert lookup[num] == tgt_vocab.decode(t_ids.tolist()), f'입력-정답 짝이 안 맞음: {num}'
        print('[OK] batch 복원')
        print('예시:', src_vocab.decode(src[0].tolist()), '->', tgt_vocab.decode(tgt[0].tolist()))

    # 6. kor <-> eng: 두 파일에 모두 있는 숫자만 짝이 된다
    print('\n===== kor -> eng =====')
    pair = pair_df(combined, 'kor', 'eng')
    loaders = load_data('kor', 'eng', combined, config)
    assert sum(len(l.dataset) for l in loaders[:3]) == len(pair) == n_both, 'kor-eng 짝 개수가 겹치는 숫자 개수와 다름'
    src_vocab, tgt_vocab = loaders[3], loaders[4]
    lookup = dict(zip(pair['kor'], pair['eng']))
    src, tgt = next(iter(loaders[0]))
    for s_ids, t_ids in zip(src, tgt):
        kor = src_vocab.decode(s_ids.tolist())
        assert lookup[kor] == tgt_vocab.decode(t_ids.tolist()), f'입력-정답 짝이 안 맞음: {kor}'
    print(f'[OK] kor-eng 짝 {len(pair)}개, batch 복원')
    print('예시:', src_vocab.decode(src[0].tolist()), '->', tgt_vocab.decode(tgt[0].tolist()))

    print('\n모든 테스트 통과')