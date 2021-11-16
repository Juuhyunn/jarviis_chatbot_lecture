from django.shortcuts import render

# Create your views here.

from django.http import JsonResponse
from icecream import ic
from rest_framework import viewsets, permissions, generics, status
from rest_framework.parsers import JSONParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.decorators import api_view, parser_classes

from userlog.models import UserLog
from userlog.models_data import DbUploader
from userlog.models_process import LogData


@api_view(['GET'])
@parser_classes([JSONParser])
def process(request):
    LogData().process()
    return JsonResponse({'process Upload': 'SUCCESS'})


@api_view(['GET'])
@parser_classes([JSONParser])
def upload(request):
    DbUploader().insert_data()
    return JsonResponse({'LogData Upload': 'SUCCESS'})


@api_view(['GET'])
@parser_classes([JSONParser])
def test(request):
    LogData().dummy_from_db()
    return JsonResponse({'test': 'SUCCESS'})

# ===== react =====


@api_view(['GET', 'POST'])
@parser_classes([JSONParser])
def modify(request):
    return JsonResponse({'getlatlng': 'SUCCESS'})


@api_view(['GET', 'POST'])
@parser_classes([JSONParser])
def remove(request):
    return JsonResponse({'getlatlng': 'SUCCESS'})


@api_view(['GET', 'POST'])
@parser_classes([JSONParser])
def create(request):
    return JsonResponse({'getlatlng': 'SUCCESS'})


@api_view(['GET', 'POST'])
@parser_classes([JSONParser])
def find(request):
    return JsonResponse({'getlatlng': 'SUCCESS'})


@api_view(['GET', 'POST'])
@parser_classes([JSONParser])
def list(request):
    ic("********** list **********")
    userlog = UserLog.objects.all()

    return JsonResponse({'getlatlng': 'SUCCESS'})