from django.conf.urls import url

from diary import views

urlpatterns = {
    url(r'process', views.process)
}