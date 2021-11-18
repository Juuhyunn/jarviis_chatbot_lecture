import json

import requests

from common.models import ValueObject
from bs4 import BeautifulSoup
from selenium import webdriver
import csv
import time


class Location(object):
    def __init__(self):
        pass

    def getLatLng_test(self):
        return self.kakao_api("서울특별시 서초구 강남대로 373 홍우빌딩 1층")

    def getAddress(self, keyword):
        url = 'https://dapi.kakao.com/v2/local/search/keyword.json?query=' + keyword
        headers = {"Authorization": "KakaoAK 851f4e6cf0cce36ebf456a4eb33b94d4"}
        # get 방식으로 주소를 포함한 링크를 헤더와 넘기면 result에 json형식의 주소와 위도경도 내용들이 출력된다.
        result = json.loads(str(requests.get(url, headers=headers).text))
        status_code = requests.get(url, headers=headers).status_code
        if (status_code != 200):
            # print(f"ERROR: Unable to call rest api, http_status_coe: {status_code}")
            return 0
        try:
            match_first = result['documents'][0]
            x = match_first['x']
            y = match_first['y']
            address = match_first['road_address_name']
            # print(x)
            # print(y)
            # print(address)
            return x, y, address
        except IndexError:  # match값이 없을때
            print('match값이 없을때')
            x, y, address = '0', '0', '0'
            return x, y, address
        except TypeError:  # match값이 2개이상일때
            print('match값이 2개이상일때')
            x, y, address = '1', '1', '1'
            return x, y, address

    def getLatLng(self, addr):
        url = 'https://dapi.kakao.com/v2/local/search/address.json?query=' + addr
        headers = {"Authorization": "KakaoAK 851f4e6cf0cce36ebf456a4eb33b94d4"}
        # get 방식으로 주소를 포함한 링크를 헤더와 넘기면 result에 json형식의 주소와 위도경도 내용들이 출력된다.
        result = json.loads(str(requests.get(url, headers=headers).text))
        status_code = requests.get(url, headers=headers).status_code
        if (status_code != 200):
            # print(f"ERROR: Unable to call rest api, http_status_coe: {status_code}")
            return 0
        # print(requests.get(url, headers=headers))
        # print(result)
        try:
            match_first = result['documents'][0]['address']
            x = match_first['x']
            y = match_first['y']
            # print(lon, lat)
            # print(match_first)
            # print(f'x : {x}, y : {y}')
            return x, y
        except IndexError:  # match값이 없을때
            print('match값이 없을때')
            x, y = '0', '0'
            return x, y

        except TypeError:  # match값이 2개이상일때
            print('match값이 2개이상일때')
            x, y = '1', '1'
            return x, y

    def save_csv(self):
        vo = ValueObject()
        vo.context = 'location/data/'
        ls = self.crawling(vo)
        # [print(i.find('strong').text) for i in location]
        with open(f'{vo.context}location_data.csv', 'w', encoding='UTF-8') as f:
            w = csv.writer(f)
            w.writerow(['location','category','address'])
            for i in ls:
                w.writerow(i)

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
