from django.db import models

# Create your models here.


from django.db import models

# Create your models here.

# 연령,성별,상황키워드,감정,문장


class DiaryDataset(models.Model):
    age = models.TextField()
    gender = models.TextField()
    situation = models.TextField()
    emotion = models.TextField()
    sentence = models.TextField()

    class Meta:
        db_table = 'diary_dataset'

    def __str__(self):
        return f'[{self.pk}] 나이 : {self.age},' \
               f'성별 : {self.gender},' \
               f'상황 키워드 : {self.situation},' \
               f'감정 : {self.emotion},' \
               f'문장 : {self.sentence}'
