
from django.urls import path, re_path


from . import views

urlpatterns = [
    path('', views.index, name='index'),
    re_path(r'^summarize/$', views.summarize_web_page, name='summarize')
]

