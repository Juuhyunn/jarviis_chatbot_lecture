from django.conf.urls import url

from logdata import views

urlpatterns = {
    url(r'process', views.process)
}