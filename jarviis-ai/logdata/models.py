import csv
import random

from django.db import models

# Create your models here.
from common.models import ValueObject, Reader, Printer
from weather.models import Weather


class LogData(object):
    def __init__(self):
        vo = ValueObject()
        vo.context = 'location/data/'
        vo.fname = 'location_data.csv'
        reader = Reader()
        self.printer = Printer()
        self.csvfile = reader.new_file(vo)

    def process(self):
        self.random_log()

    def random_log(self):
        ls = self.dummy_csv()
        num = random.randint(0,len(ls)-1)
        result = self.create_visit(ls, num) if random.randint(0, 1) == 0 else self.create_payment(ls, num)
        # print(result)
        return result

    def create_visit(self, ls, num):
        return {'location': ls[num]['location'],
                'gps': ls[num]['address'],
                'log_type': 'visit',
                'contents': f"{ls[num]['category']}-{ls[num]['location']}을 방문함.",
                'item': f"{ls[num]['location']} 방문"}

    def create_payment(self, ls, num):
        return {'location': ls[num]['location'],
                'gps': ls[num]['address'],
                'log_type': 'payment',
                'contents': f"[{ls[num]['category']}] {ls[num]['location']}에서 결제함.",
                'item': f"{random.randint(0,1000)*100}"}

    def dummy_csv(self):
        ls = []
        with open(self.csvfile, newline='', encoding='utf8') as f:
            [ls.append(i) for i in csv.DictReader(f)]
        return ls


class UserLog(object):
    def __init__(self):
        self.weather = Weather()
        self.logdata = LogData()

    def process(self):
        print(self.create_log())

    def create_log(self):
        log = self.logdata.random_log()
        weather = self.weather.process()
        return {'location': log['location'],
                'gps': log['gps'],
                'weather': weather,
                'log_type': log['log_type'],
                'contents': log['contents'],
                'item': log['item']}
