from django.http import JsonResponse

# Create your views here.
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser

from weather.models import Weather


@api_view(['GET', 'POST'])
@parser_classes([JSONParser])
def process(request):
    Weather().process()
    return JsonResponse({'Weather': 'SUCCESS'})