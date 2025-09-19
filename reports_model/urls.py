from django.urls import path
from django.urls import path
from django.conf.urls.static import static
from django.conf import settings
from django.urls import re_path
from analyzer.views import (
    predict_sentiment,
)

# Base prefix for admin routes
add_base_url_admin="api/"

urlpatterns = [
    # SEO Management
    path(add_base_url_admin + 'predict_sentiment', predict_sentiment, name='predict_sentiment'),
   
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# Add any additional URLs here as needed    