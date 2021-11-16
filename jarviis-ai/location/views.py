from django.http import JsonResponse

# Create your views here.
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser

from location.models_data import DbUploader
from location.models_process import Location


@api_view(['GET', 'POST'])
@parser_classes([JSONParser])
def save_csv(request):
    Location().save_csv()
    return JsonResponse({'LocationCrawling': 'SUCCESS'})


@api_view(['GET', 'POST'])
@parser_classes([JSONParser])
def upload(request):
    DbUploader().insert_data()
    return JsonResponse({'LocationData': 'SUCCESS'})


@api_view(['GET', 'POST'])
@parser_classes([JSONParser])
def getLatLng(request):
    Location().getLatLng()
    return JsonResponse({'getlatlng': 'SUCCESS'})


