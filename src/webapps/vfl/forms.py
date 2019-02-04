'''
Description: This file should be used
to create forms for registering the users,
logging in, registering audio recordings,
etc.

@author: Shahan A. Memon
'''

from django import forms
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
import re
from vfl.models import *
'''
Description: Defining the registration form
for the voice forensics lab
'''

class RegistrationForm(forms.Form):
	username = forms.CharField(max_length=20,
							   label = 'Username')
	firstname = forms.CharField(max_length=20,
				    label = 'First name')
	middlename = forms.CharField(max_length=1,
				     label = 'MI',
				     required=False)
	lastname = forms.CharField(max_length=20,
				   label = 'Last name')
	orgkey = forms.CharField(max_length=12,
				 label = 'Org-key')
	password1 = forms.CharField(max_length=256,
				    label = 'Password',
				    widget = forms.PasswordInput())
	password2 = forms.CharField(max_length=256,
				    label = 'Confirm Password',
				    widget = forms.PasswordInput())
	email = forms.CharField(max_length=50, label='Email Address')

		# Customizing our clean function
	def clean(self):
		cleaned_data = super(RegistrationForm, self).clean()
		# Confirms that the two password fields match
		password1 = cleaned_data.get('password1')
		password2 = cleaned_data.get('password2')
		if password1 and password2 and password1 != password2:
			raise forms.ValidationError("Passwords did not match.")

		return cleaned_data

	def clean_username(self):
		username = self.cleaned_data.get('username')
		if not username:
			raise forms.ValidationError("Username is Required")
		if(User.objects.filter(username__exact=username)):
			raise forms.ValidationError("Username is already taken.")

		return username

	def clean_firstname(self):
		firstname = self.cleaned_data.get('firstname')
		if not firstname:
			raise forms.ValidationError('First name is Required')

		return firstname

	def clean_lastname(self):
		lastname = self.cleaned_data.get('lastname')
		if not lastname:
			raise forms.ValidationError('Last name is Required')

		return lastname

	# The org-key needs to be exactly 12 characters long

	def clean_orgkey(self):
		orgkey = self.cleaned_data.get('orgkey')
		if not orgkey:
			raise forms.ValidationError('Org key is required')

		if(len(orgkey) != 12):
			raise forms.ValidationError('Org key needs to be exactly 12 characters')

		return orgkey

	def clean_email(self):
		email = self.cleaned_data.get('email')
		if not email:
			raise forms.ValidationError("Email is Required")
		if(User.objects.filter(email__exact=email)):
			raise forms.ValidationError("Email Address is already taken.")

   		# Now we validate if the email address is valid syntactically

		match = re.match('^[_a-z0-9-]+(\.[_a-z0-9-]+)'\
				'*@[a-z0-9-]+(\.[a-z0-9-]+)*(\.[a-z]{2,4})$', email)

		if(match == None):
			raise forms.ValidationError("Please Enter a Valid Email Address")

		return email

'''
Description: Defining the Login form
'''

class LoginForm(forms.Form):

	username = forms.CharField(max_length=20)
	password = forms.CharField(max_length=200,
			label = 'Password',
			widget = forms.PasswordInput())

	# Customizing our clean function
'''	def clean(self):
		cleaned_data = super(LoginForm, self).clean()
		
		try:
			user = authenticate(username=cleaned_data.get('username'),
				password=cleaned_data.get('password'))
		except:
			raise forms.ValidationError("Invalid Username or Password")
		if(user == None):
			raise forms.ValidationError("No user")

		return cleaned_data'''

'''
This is the form that we will have on the login page
for forgot password
'''

class ResetPasswordForm(forms.Form):
	username = forms.CharField(max_length=20)
	email = forms.CharField(max_length=50,
			label = 'Email Address')

	def clean(self):
		cleaned_data = super(ResetPasswordForm, self).clean()
		username = self.cleaned_data.get('username')
		email = self.cleaned_data.get('email')
		if(not(email == User.objects.get(username=username).email)):
			raise forms.ValidationError("Username or Email Invalid")
		return cleaned_data

	def clean_email(self):
		email = self.cleaned_data.get('email')
		if not email:
			raise forms.ValidationError("Email is Required")
		match = re.match('^[_a-z0-9-]+(\.[_a-z0-9-]+)*@[a-z0-9-]+(\.[a-z0-9-]+)*(\.[a-z]{2,4})$', email)

		if(match == None):
			raise forms.ValidationError("Please Enter a Valid Email Address")

		return email

	def clean_username(self):
		username = self.cleaned_data.get('username')
		if not username:
			raise forms.ValidationError("Username is Required")
		if(len(User.objects.filter(username=username)) < 1):
			raise forms.ValidationError("Invalid username")
		return username

'''
Now we need to create forms for registering,
bulk registering, recording, uploading audio.
'''

class AudioForm(forms.ModelForm):
    class Meta:
        model = Audio
        fields = ('file',)

class RetrieveSimilarForm(forms.Form):
	audio_file = forms.FileField()
	def clean(self):
		cleaned_data = super(RetrieveSimilarForm, self).clean()
		# Need to check the extension here.
		return cleaned_data

