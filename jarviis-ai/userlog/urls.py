from django.conf.urls import url

from userlog import views

urlpatterns = [
    url(r'upload', views.upload),

]