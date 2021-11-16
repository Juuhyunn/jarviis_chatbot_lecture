import pandas as pd
from django.db import models

# Create your models here.
from icecream import ic
from sklearn.model_selection import train_test_split

from common.models import ValueObject, Reader, Printer


class DiaryTest(object):
    def __init__(self):
        self.vo = ValueObject()
        self.vo.context = 'diary/data/'
        self.vo.fname = 'train.csv'
        self.reader = Reader()
        self.printer = Printer()

    def process(self):
        self.split_model()
        self.age_nominal()
        self.gender_nominal()
        self.situation_nominal()
        self.emotion_nominal()
        self.printer.dframe(self.vo.train)

    def new_model(self) -> object:
        return pd.read_csv(self.reader.new_file(self.vo))

    def split_model(self) -> []:
        self.vo.train, self.vo.test = train_test_split(self.new_model(), test_size=0.2, random_state=42)

    def age_nominal(self):
        # label = ['청소년', '청년', '중년', '노년']
        age_mapping = {'청소년': 0, '청년': 1, '중년': 2, '노년': 3}
        for i in self.vo.train, self.vo.test:
            i['연령'] = i['연령'].map(age_mapping)
        print(self.vo.train)

    def gender_nominal(self):
        # label = ['FEMALE', 'MALE', '남성', '여성']
        age_mapping = {'MALE': 0, '남성': 0, 'FEMALE': 1, '여성': 1}
        for i in self.vo.train, self.vo.test:
            i['성별'] = i['성별'].map(age_mapping)
        print(self.vo.train)

    def situation_nominal(self):
        # label = ['가족관계', '건강', '대인관계', '연애, 결혼, 출산', '재정', '직장, 업무 스트레스', '진로, 취업, 직장', '학교폭력/따돌림', '학업 및 진로']
        age_mapping = {'가족관계': 0, '건강': 1, '대인관계': 2, '연애, 결혼, 출산': 3, '재정': 4, '직장, 업무 스트레스': 5, '진로, 취업, 직장': 6, '학교폭력/따돌림': 7, '학업 및 진로': 8}
        for i in self.vo.train, self.vo.test:
            i['상황키워드'] = i['상황키워드'].map(age_mapping)
        print(self.vo.train)

    def emotion_nominal(self):
        # label = ['기쁨', '당황', '분노', '불안', '상처', '슬픔']
        age_mapping = {'기쁨': 0, '당황': 1, '분노': 2, '불안': 3, '상처': 4, '슬픔': 5}
        for i in self.vo.train, self.vo.test:
            i['감정'] = i['감정'].map(age_mapping)
        print(self.vo.train)


