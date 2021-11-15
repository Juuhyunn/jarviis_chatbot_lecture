from django.http import JsonResponse

# Create your views here.
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser

from logdata.models import LogData, UserLog


@api_view(['GET', 'POST'])
@parser_classes([JSONParser])
def process(request):
    UserLog().process()
    return JsonResponse({'UserLog': 'SUCCESS'})