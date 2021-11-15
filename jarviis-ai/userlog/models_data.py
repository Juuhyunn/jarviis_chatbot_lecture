import os
import django
import csv
import sys

from common.models import ValueObject, Reader, Printer
from userlog.models import LogData


class DbUploader():
    def __init__(self):
        vo = ValueObject()
        reader = Reader()
        self.printer = Printer()
        vo.context = 'userlog/data/'
        vo.fname = 'logdata.csv'
        self.csvfile = reader.new_file(vo)

    def insert_data(self):
        self.insert_logdata()

    def insert_logdata(self):
        with open(self.csvfile, newline='', encoding='utf8') as f:
            data_reader = csv.DictReader(f)
            for row in data_reader:
                LogData.objects.create(location = row['location'],
                                       gps = row['gps'],
                                       log_type = row['log_type'],
                                       contents = row['contents'],
                                       item = row['item'])
        print('LOG DATA UPLOADED SUCCESSFULY!')

