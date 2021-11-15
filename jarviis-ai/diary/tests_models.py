import pandas as pd
from django.db import models

# Create your models here.
from icecream import ic

from common.models import ValueObject, Reader, Printer


class DiaryTest(object):
    def __init__(self):
        self.vo = ValueObject()
        self.vo.context = 'diary/data/'
        self.vo.fname = 'leisure'
        self.reader = Reader()
        self.printer = Printer()

    def process(self):
        pass

    def new_model(self, vo) -> object:
        return pd.read_csv(self.reader.new_file(vo))

    
