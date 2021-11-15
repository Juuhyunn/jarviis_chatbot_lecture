import csv
import time

# Create your models here.
from common.models import ValueObject
from bs4 import BeautifulSoup
from selenium import webdriver


class Location(object):
    def __init__(self):
        pass

    def process(self):
        vo = ValueObject()
        # self.crawling(vo)
        self.save_csv(vo)

    def crawling(self, vo):
        loc = '강남역'
        vo.url = f'https://m.map.naver.com/search2/search.naver?query={loc}'
        driver = webdriver.Chrome('common/data/chromedriver.exe')
        driver.get(vo.url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        driver.close()
        location = soup.find_all('div', {'class': 'item_info'})
        ls = []
        [ls.append([i.find('strong').text, i.find('em').text, i.find('p').text]) for i in location]
        return ls

    def save_csv(self, vo):
        vo.context = 'location/data/'
        ls = self.crawling(vo)
        # [print(i.find('strong').text) for i in location]
        with open(f'{vo.context}location_data.csv', 'w', encoding='UTF-8') as f:
            w = csv.writer(f)
            w.writerow(['location','category','address'])
            for i in ls:
                w.writerow(i)


