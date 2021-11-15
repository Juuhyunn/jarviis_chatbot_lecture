from django.http import JsonResponse

# Create your views here.
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser

from location.models import Location


@api_view(['GET', 'POST'])
@parser_classes([JSONParser])
def process(request):
    Location().process()
    return JsonResponse({'Location': 'SUCCESS'})