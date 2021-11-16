import os
import django
import csv
import sys

from common.models import ValueObject, Reader, Printer
from diary.models import DiaryDataset


class DbUploader:
    def __init__(self):
        vo = ValueObject()
        reader = Reader()
        self.printer = Printer()
        vo.context = 'diary/data/'
        vo.fname = 'train.csv'
        self.csvfile = reader.new_file(vo)

    def process(self):
        self.insert_dataset()

    def insert_dataset(self):
        with open(self.csvfile, newline='', encoding='utf8') as f:
            data_reader = csv.DictReader(f)
            for row in data_reader:
                DiaryDataset.objects.create(age=row['연령'],
                                            gender=row['성별'],
                                            situation=row['상황키워드'],
                                            emotion=row['감정'],
                                            sentence=row['문장'])
        print('DIARY DATASET UPLOADED SUCCESSFULY!')

