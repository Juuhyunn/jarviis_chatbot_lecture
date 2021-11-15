from django.db import models

# Create your models here.


class UserLog(models.Model):
    # use_in_migrations = True
    location = models.TextField()
    gps = models.TextField()
    log_date = models.DateTimeField(auto_now_add=True)
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
               f'GPS : {self.gps},' \
               f'로그 생성 날짜 : {self.log_date},' \
               f'날씨 : {self.weather},' \
               f'로그 타입 : {self.log_type},' \
               f'내용 : {self.contents},' \
               f'주요 내용 : {self.item},' \
               f'사용자 : {self.user_id}'


class LogData(models.Model):
    location = models.TextField()
    gps = models.TextField()
    log_type = models.TextField()
    contents = models.TextField()
    item = models.TextField()

    class Meta:
        db_table = 'logdata'

    def __str__(self):
        return f'[{self.pk}] 위치 : {self.location},' \
               f'GPS : {self.gps},' \
               f'로그 타입 : {self.log_type},' \
               f'내용 : {self.contents},' \
               f'주요 내용 : {self.item}'
