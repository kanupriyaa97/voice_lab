from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver
from jsonfield import JSONField
from django.contrib.postgres.fields import ArrayField
from passlib.hash import pbkdf2_sha256
from .storage import OverwriteStorage
from .validators import validate_file_extension

class Organization(models.Model):
    orgkey = models.CharField(max_length=12, blank=False)
    owner = models.OneToOneField(User, primary_key=True,on_delete=models.CASCADE)

class Audio(models.Model):
    file = models.FileField(storage=OverwriteStorage(), validators=[validate_file_extension])
    owner = models.ForeignKey(User,on_delete=models.CASCADE)
    updated_at = models.DateTimeField(auto_now=True)
    isAudio = models.BooleanField(default=False, blank=False)
    isTemporary = models.BooleanField(default=False, blank=False)
    title = models.CharField(max_length=255, blank=True)
    dvector = models.CharField(max_length=10000,blank=True)
    params = models.CharField(max_length=10000,blank=True)

    organization = models.ForeignKey(Organization, on_delete=models.CASCADE)

class Profile(models.Model):
    user = models.OneToOneField(User, primary_key=True,on_delete=models.CASCADE)
    org = models.ForeignKey(Organization, blank=False, on_delete=models.CASCADE)
    middle_name = models.CharField(max_length=1, blank=True)
    orgFlag = models.BooleanField(default=False, blank=False)
