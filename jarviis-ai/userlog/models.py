from datetime import datetime

from django.db import models

# Create your models here.


class UserLog(models.Model):
    # use_in_migrations = True
    location = models.TextField()
    address = models.TextField()
    x = models.TextField()
    y = models.TextField()
    log_date = models.DateTimeField(default=datetime.now())
    weather = models.TextField()
    log_type = models.TextField()
    contents = models.TextField()
    item = models.TextField()
    user_id = models.IntegerField()

    class Meta:
        # managed = True
        db_table = 'userlog'

    def __str__(self):
        return f'[{self.pk}] 위치 : {self.location},' \
               f'주소 : {self.address},' \
               f'위도 : {self.x},' \
               f'경도 : {self.y},' \
               f'로그 생성 날짜 : {self.log_date},' \
               f'날씨 : {self.weather},' \
               f'로그 타입 : {self.log_type},' \
               f'내용 : {self.contents},' \
               f'주요 내용 : {self.item},' \
               f'사용자 : {self.user_id}'
