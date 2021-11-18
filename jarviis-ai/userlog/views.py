from datetime import datetime

from django.shortcuts import render

# Create your views here.

from django.http import JsonResponse
from icecream import ic
from rest_framework import viewsets, permissions, generics, status
from rest_framework.parsers import JSONParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.decorators import api_view, parser_classes

from location.models_process import Location
from userlog.models import UserLog
from userlog.models_data import DbUploader
from userlog.models_process import LogData
from userlog.serializers import UserLogSerializer


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

# ===== react api =====


@api_view(['PUT'])
@parser_classes([JSONParser])
def modify(request):
    ic("********** modify **********")
    edit = request.data
    ic(edit)
    log = UserLog.objects.get(pk=edit['id'])
    db = UserLog.objects.all().filter(id=edit['id']).values()[0]
    # print(f' 변경 전 : {db}')
    db['location'] = edit['location']
    x, y, address = Location.getAddress(edit['location'])
    db['address'] = address
    db['x'] = x
    db['y'] = y
    db['log_date'] = edit['log_date']
    db['weather'] = edit['weather']
    db['log_type'] = edit['log_type']
    db['contents'] = edit['contents']
    db['item'] = edit['item']
    # print(f' 변경 후 : {db}')
    serializer = UserLogSerializer(data=db)
    # print(f'db type : {type(db)}  // serializer type : {type(serializer)}')
    if serializer.is_valid():
        serializer.update(log, db)
        return JsonResponse(data=serializer.data, safe=False)
    return JsonResponse(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['DELETE'])
@parser_classes([JSONParser])
def remove(request, pk):
    ic("********** remove **********")
    ic(f'pk : {pk}')
    db = UserLog.objects.get(pk=pk)
    db.delete()
    return JsonResponse({'User Log': 'DELETE SUCCESS'})


@api_view(['POST'])
@parser_classes([JSONParser])
def create(request):
    ic("********** create **********")
    new = request.data
    ic(f'new : {new}')
    if new['address'] == '':
        x, y, address = Location().getAddress(new['location'])
        new['address'] = address
        new['x'] = x
        new['y'] = y
    else:
        print(new['address'])
        x, y = Location().getLatLng(addr=new['address'])
        new['x'] = x
        new['y'] = y
    UserLog.objects.create(location=new['location'],
                           address=new['address'],
                           x=new['x'],
                           y=new['y'],
                           weather=new['weather'],
                           log_type=new['log_type'],
                           contents=new['contents'],
                           item=new['item'],
                           user_id=new['user_id'])
    return JsonResponse({'USER LOG': 'CREATE SUCCESS'})



@api_view(['GET', 'POST'])
@parser_classes([JSONParser])
def find(request):
    return JsonResponse({'getlatlng': 'SUCCESS'})


@api_view(['GET', 'POST'])
@parser_classes([JSONParser])
def list_by_date(request, year, month, day):
    ic("********** list by date **********")
    ic(f'date : {year}-{month}-{day}')
    userlog = UserLog.objects.filter(log_date__year= year, log_date__month=month, log_date__day=day)
    serializer = UserLogSerializer(userlog, many=True)
    ic(serializer.data)
    return JsonResponse(data = serializer.data, safe=False)