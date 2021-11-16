from django.conf.urls import url

from userlog import views

urlpatterns = [
    url(r'process', views.process),
    url(r'upload', views.upload),
    url(r'test', views.test),

]