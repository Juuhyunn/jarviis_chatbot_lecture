import csv
import random

from common.models import ValueObject, Reader, Printer
from location.models import LocationData
from location.models_process import Location
from location.serializers import LocationSerializer
from weather.models import Weather


class LogData(object):
    def __init__(self):
        vo = ValueObject()
        vo.context = 'location/data/'
        vo.fname = 'location_data.csv'
        reader = Reader()
        # self.printer = Printer()
        self.csvfile = reader.new_file(vo)
        self.weather = Weather()

    def process(self):
        return self.create_log()

    def create_log(self):
        log = self.random_log()
        weather = self.weather.process()
        latlng = Location().getLatLng(log['address'])
        return {'location': log['location'],
                'address': log['address'],
                'x': latlng[0],
                'y': latlng[1],
                'weather': weather,
                'log_type': log['log_type'],
                'contents': log['contents'],
                'item': log['item']}

    def random_log(self):
        ls = self.dummy_from_db()
        num = random.randint(0 ,len(ls ) -1)
        return self.create_visit(ls, num) if random.randint(0, 1) == 0 else self.create_payment(ls, num)

    def create_visit(self, ls, num):
        return {'location': ls[num]['location'],
                'address': ls[num]['address'],
                'log_type': 'visit',
                'contents': f"{ls[num]['category']}-{ls[num]['location']}을 방문함.",
                'item': f"{ls[num]['location']} 방문"}

    def create_payment(self, ls, num):
        return {'location': ls[num]['location'],
                'address': ls[num]['address'],
                'log_type': 'payment',
                'contents': f"[{ls[num]['category']}] {ls[num]['location']}에서 결제함.",
                'item': f"{random.randint(0 ,1000 ) *100}"}

    def dummy_from_csv(self):
        ls = []
        with open(self.csvfile, newline='', encoding='utf8') as f:
            [ls.append(i) for i in csv.DictReader(f)]
        return ls

    def dummy_from_db(self):
        ls = []
        data = LocationData.objects.all()
        serializer = LocationSerializer(data, many=True)
        [ls.append(dict(i)) for i in serializer.data]
        return ls