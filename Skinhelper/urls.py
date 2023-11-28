from django.contrib import admin
from django.urls import path
from django.views.generic import TemplateView
#from .views import upload_image
from .views import predict

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", TemplateView.as_view(template_name="index.html")),
    path('upload_image/', upload_image, name='upload_image'),
    path('predict/', predict, name='predict'),
    # path("main/", asdasda)
]
