from django.conf.urls import url

from location import views

urlpatterns = {
    url(r'save_csv', views.save_csv),
    url(r'upload', views.upload),
    url(r'getLatLng', views.getLatLng),
}