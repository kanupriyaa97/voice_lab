from vfl import views
import django.contrib.auth.views
from django.urls import path

urlpatterns = [

        # The actions that django takes care of
        path('', views.home, name='home'),
        # These are the actions we have to define
        path('home', views.home),
        path('logout', django.contrib.auth.views.logout_then_login, name='logout'),
        path('register', views.register, name='register'),
        path('confirm-reg/str:<user_id>/str:<token>', views.confirm_registration, name='confirm-reg'),
        path('verify-org/str:<user_id>/str:<token>', views.verify_organization, name='verify-org'),
        path('login', views.log_in, name='login'),
        path('dashboard', views.load_dashboard, name='dashboard'),
        path('registeraudio', views.RegisterView.as_view(), name='registeraudio'),
        path('clear', views.clear_database, name='clear_database'),
        path('retrievesimilar', views.retrieve_similar, name='retrievesimilar'),
        path('retrieveaudio/<str:id>' , views.retrieve_audio, name='retrieveaudio'),
        path('waveformaudio/<str:id>' , views.waveform_audio, name='waveformaudio'),
        path('ser' , views.ser, name='ser'),
        path('texttospeech', views.texttospeech, name='texttospeech')
]