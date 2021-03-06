from django.http import JsonResponse

# Create your views here.
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser

from diary.models_data import DbUploader
from diary.tests_models import DiaryTest


@api_view(['GET', 'POST'])
@parser_classes([JSONParser])
def process(request):
    DiaryTest().process()
    return JsonResponse({'DiaryTest': 'SUCCESS'})


@api_view(['GET', 'POST'])
@parser_classes([JSONParser])
def upload(request):
    DbUploader().process()
    return JsonResponse({'DiaryDbUploader': 'SUCCESS'})