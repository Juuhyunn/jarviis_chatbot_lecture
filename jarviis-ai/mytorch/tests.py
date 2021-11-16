from django.test import TestCase

# Create your tests here.
import torch
import urllib.request
import pandas as pd
from torchtext import data # torchtext.data 임포트
# from torchtext. import TabularDataset, Iterator
from konlpy.tag import Mecab
from torchtext.legacy.data import TabularDataset, Iterator


class KoreanTorch:
    def __init__(self):
        pass

    def myMecab(self):
        # 1. 형태소 분석기 Mecab 설치
        # 2. 훈련 데이터와 테스트 데이터로 다운로드 하기
        urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt",
                                   filename="data/ratings_train.txt")
        urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt",
                                   filename="data/ratings_test.txt")
        train_df = pd.read_table('data/ratings_train.txt')
        test_df = pd.read_table('data/ratings_test.txt')
        print(f'메캅 상위 5개 : {train_df.head()}')
        print(f'훈련 데이터 샘플의 개수 : {len(train_df)}')
        print(f'테스트 데이터 샘플의 개수 : {len(test_df)}')

        # 3. 필드 정의하기(torchtext.data)
        # Mecab을 토크나이저로 사용
        tokenizer = Mecab()
        # 필드 정의
        ID = data.Field(sequential=False,
                        use_vocab=False)  # 실제 사용은 하지 않을 예정

        TEXT = data.Field(sequential=True,
                          use_vocab=True,
                          tokenize=tokenizer.morphs,  # 토크나이저로는 Mecab 사용.
                          lower=True,
                          batch_first=True,
                          fix_length=20)

        LABEL = data.Field(sequential=False,
                           use_vocab=False,
                           is_target=True)

        # 4. 데이터셋 만들기
        train_data, test_data = TabularDataset.splits(
            path='.', train='ratings_train.txt', test='ratings_test.txt', format='tsv',
            fields=[('id', ID), ('text', TEXT), ('label', LABEL)], skip_header=True)

        print(f'훈련 샘플의 개수 : {len(train_data)}')
        print(f'테스트 샘플의 개수 : {len(test_data)}')
        print(vars(train_data[0]))

        # 5. 단어 집합(Vocabulary) 만들기
        TEXT.build_vocab(train_data, min_freq=10, max_size=10000)
        print(f'단어 집합의 크기 : {len(TEXT.vocab)}')
        print(TEXT.vocab.stoi)

        # 6. 토치텍스트의 데이터로더 만들기
        batch_size = 5
        train_loader = Iterator(dataset=train_data, batch_size=batch_size)
        test_loader = Iterator(dataset=test_data, batch_size=batch_size)
        print(f'훈련 데이터의 미니 배치 수 : {len(train_loader)}')
        print(f'테스트 데이터의 미니 배치 수 : {len(test_loader)}')
        batch = next(iter(train_loader))  # 첫번째 미니배치
        print(batch.text)


if __name__ == '__main__':
    # print(f'Result : {torch.cuda.is_available()}')
    k = KoreanTorch()
    k.myMecab()