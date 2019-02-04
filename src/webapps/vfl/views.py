'''
Created on Sun November 4 05:03:26 2018

#author(s): Shahan Ali Memon
<samemon@cs.cmu.edu>

Mentors: Charlie Garrod, Josh Sunshine

Description: This file describes the views
or actions of the vfl app for the webapps
course at CMU for the Fall of 2018.
'''

__author__ = "Shahan Ali Memon samemon@cs.cmu.edu"
__copyright__ = "Copyright (C) 2019 Shahan Ali Memon"
__license__ = "CMU"


import json
from os.path import join, dirname
from django.views.decorators.csrf import csrf_exempt
from watson_developer_cloud import WatsonApiException
from watson_developer_cloud import SpeechToTextV1
from watson_developer_cloud import PersonalityInsightsV3
from django import forms
from django.shortcuts import render, redirect, get_object_or_404
from django.core.exceptions import ObjectDoesNotExist
from django.urls import reverse
from django.contrib.auth.tokens import default_token_generator
from django.db import transaction
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth import login,authenticate
from django.http import HttpResponse, Http404
from django.core.mail import send_mail
from vfl.models import *
from vfl.forms import *
from mimetypes import guess_type
from django.views import View
from django.http import JsonResponse
import scipy.io.wavfile as wav
#from vfl.dvecs.voice_to_dvector import extract_dvector 
import os, sys
import numpy as np
import codecs, json
import vfl.compare_dvecs as dvec_compare
# Create your views here.
this_folder = os.path.dirname(os.path.abspath(__file__))
# Importing classifier files
sys.path.append(this_folder)
import jitter_shimmer.voice_to_jitshim as v2j
import gender.voice_to_gender as v2g

class RegisterView(View):

	def get(self, request):
		# We do not want to return all audios, but only ones which this user has access to
		# Let's first get the organization
		audio_list = []
		orgFlag = request.user.profile.orgFlag
		if(orgFlag):
			audio_list = Audio.objects.filter(organization=request.user.profile.org)
		else:
			audio_list = Audio.objects.filter(owner=request.user)
		return render(self.request, 'vfl/base_loggedin1.html', {'audios': audios_list})

	def post(self, request):
		form = AudioForm(self.request.POST, self.request.FILES)
		if(form.is_valid()):
            # We are not commiting yet - as we need to save the title
			audio = form.save(commit=False)
            # Saving the file name as the title 
			'''
            Note but we need to check if some other audio has the same title in which case
            the old audio needs to be deleted.
            '''
			# Let's check if the file is wav or txt
			if(".wav" in audio.file.name and not(audio.file.name.split(".wav")[1])):
				
				fname_user_ext = audio.file.name.split(".wav")[0] \
					+ "_" + request.user.username + ".wav"
				
				Audio.objects.filter(title__exact=fname_user_ext).delete()
				
				fname_user_annext = fname_user_ext.split(".wav")[0]+".txt"
				annot_present = Audio.objects.filter(title__exact=fname_user_annext)

				if(not(annot_present)):
					data = {'is_valid': False}
					return JsonResponse(data)
				# This means the anno is present
				fileLst = open(Audio.objects.get(title__exact=fname_user_annext).file.path).readlines()
				dictionary = {}
				for elem in fileLst:
					elemLst = str(elem.rstrip()).split(",")
					dictionary[elemLst[0]] = elemLst[1]
				audio.params=json.dumps(dictionary)
				audio.owner = request.user
				audio.organization = request.user.profile.org
				audio.save()
				audio_path = audio.file.path
				name = audio_path.split("/")[-1].split(".wav")[0]
				os.system("python vfl/dvec/extract_dvec.py "+\
					audio_path+" "+name+ " vfl/dvec/dvecs/")
				dvec = np.load("vfl/dvec/dvecs/"+name+".npy")
				dvec_json = json.dumps(dvec.tolist())
				audio.dvector=dvec_json
				audio.isAudio=True
				
				
				audio.save()
				os.remove("vfl/dvec/dvecs/"+name+".npy")
				data = {'is_valid': True, 'name': audio.file.name, 'url': audio.file.url}
				return JsonResponse(data)

			else:

				assert(".txt" in audio.file.name)
				# Now first we create a title for this
				fname_user_ext = audio.file.name.split(".txt")[0] \
					+ "_" + request.user.username + ".txt"
				# This is fine because I add _username so files of other users will not be deleted
				Audio.objects.filter(title__exact=fname_user_ext).delete()
				
				# Now that we have deleted duplicates
				audio.title = fname_user_ext
				audio.organization = request.user.profile.org
				audio.owner = request.user
				audio.save()
				data = {'is_valid': True, 'name': audio.file.name, 'url': audio.file.url}
				return JsonResponse(data)
		else:
			data = {'is_valid': False}
			print("not valid")

		


'''
Description: This function is meant to
be called when the user wants to upload
the recording to be analyzed.
'''

def home(request, context={}):
	if(not('regform' in context.keys())):
		context['regform'] = RegistrationForm()
	if(not('loginform' in context.keys())):
		context['loginform'] = LoginForm()

	return render(request, 'vfl/vfl_register1.html', context)

'''
Description: This function will be called upon
registration.
'''

def register(request):
	context = {}

	form = RegistrationForm(request.POST)
	context['regform'] = form

	if not form.is_valid():
		return home(request, context)

	username = form.cleaned_data['username']
	password = form.cleaned_data['password1']
	first_name = form.cleaned_data['firstname']
	middle_name = form.cleaned_data['middlename']
	last_name = form.cleaned_data['lastname']
	email = form.cleaned_data['email']
	orgkey = form.cleaned_data['orgkey']
	new_user = User.objects.create_user(username=username,
			password = password,
			last_name = last_name,
			first_name = first_name,
			email = email,
			is_active = False
			)

	new_user.save()

	new_user_is_admin = False

	# Now we need to check if the organization already exists
	try:
		org = Organization.objects.get(orgkey__exact=orgkey)
	except:
		org = []
	if(org):
		# That means we do not need to create a new organization
		new_profile = Profile(user=new_user, 
			org=org, middle_name=middle_name)
		new_profile.save()
	else:
		# That means we need to create a new organization
		# And the owner will be this new user
		new_user_is_admin = True
		new_org = Organization(orgkey=orgkey, owner=new_user)
		new_org.save()
		# OrgFlag will be true as this guy is the admin of organization
		new_profile = Profile(user=new_user, 
			org=new_org, 
			orgFlag=True, 
			middle_name=middle_name)
		new_profile.save()

	'''
	Now that we have taken care of the org-key we need to
	figure out two things: 
	1) We need to first send an email to the user 
	so that s/he can activate his/her account.
	2) We need to send an email to the admin of the organization
	if this user is not the admin, to give permission to this
	user to get access to the database.
	'''
	token = default_token_generator.make_token(new_user)

	email_body = """
	Thanks for registering to Voice Forensics Lab. 
	Please click the link below to
	verify your email address and complete the registration
	of your account:
	http://%s%s
	""" % (request.get_host(),
			reverse('confirm-reg', args=(username, token)))

	send_mail(subject="VFL: Verify your email address",
		message = email_body,
		from_email="samemon+devnull@cs.cmu.edu",
		recipient_list=[email])

	if(not(new_user_is_admin)):
		token = default_token_generator.make_token(new_user)
		email_body = """
		A new user """ + first_name + " " \
		+ last_name + """ has registered to your organization 
		on Voice Forensics Lab. 
		Please click the link below to
		grant data access rights to this user:
		http://%s%s
	    """ % (request.get_host(), reverse('verify-org', args=(username, token)))
	    # Now I need to get the admin of this organization
		organization = Organization.objects.get(orgkey__exact=orgkey)
		send_mail(subject="VFL: Verify a new member", 
			message = email_body, 
			from_email="samemon+devnull@cs.cmu.edu", 
			recipient_list=[organization.owner.email])

	return home(request)

'''
Description: This function will be called on
clicking the link sent to the user upon registration 
to confirm registration. Once confirmed, the user will
be directly redirected to the dashboard
'''

def confirm_registration(request,user_id,token):
	try:
		user = User.objects.get(username=user_id)
		user.is_active = True
		user.save()

		user = authenticate(username=user_id,
				password=user.password)
		login(request, user)
		return redirect(reverse('dashboard'))

	except:
		return redirect(reverse('home'))
'''
Description: This function will be called on
clicking the link sent to the admin of the organization
entered as org-key upon registration 
to confirm access rights. Once confirmed, 
the user who is the subject will be granted
user rights right away.
'''

def verify_organization(request,user_id, token):
	user = User.objects.get(username=user_id)
	user_profile = Profile.objects.get(user = user)
	user_profile.orgFlag = True
	user_profile.save()
	return redirect(reverse('home'))

'''
Description: This is the function to load the
dashboard. This will be called at intervals.
'''

def load_dashboard(request, context={}):
	'''
	First we need to check what kind of access the user has
	'''
	'''
	We need to check if the guy has been accepted to the organization
	or else s/he will be displayed only the recordings s/he
	has uploaded.
	'''
	orgFlag = request.user.profile.orgFlag
	if(orgFlag):
		context['audios'] = Audio.objects.filter(organization=request.user.profile.org).exclude(isTemporary=True)
	else:
		context['audios'] = Audio.objects.filter(owner=request.user).exclude(isTemporary=True)
	return render(request, 'vfl/base_loggedin1.html', context)

'''
Description: This is the function that will be 
called when the user tries to log into the system
to confirm the username and password. Once successful,
s/he will be authenticated and redirected to the dashboard.
'''

def log_in(request):
	if request.method == 'POST':
		form = LoginForm(request.POST)
		if form.is_valid():
			print('doing')
			username = request.POST['username']
			password = request.POST['password']
			user = authenticate(request,username=username,password=password)
			login(request, user)
			return redirect(reverse('dashboard'))


def ser(request):
	if request.method == "GET":
		return render(request, 'vfl/ser1.html')

	else:
		audio_file = request.POST
		new_entry = Audio(file=audio_file, isAudio=True,
						  isTemporary=False)
		new_entry.owner = request.user
		new_entry.organization = request.user.profile.org
		new_entry.save()
		return render(request, 'vfl/ser1.html')


def retrieve_similar(request):
	context = {}
	if request.method == "GET":
		form = RetrieveSimilarForm()
		context = {'retrievesimilarform': form}
		return load_dashboard(request,context)
	# If the request is POST
	form = RetrieveSimilarForm(request.POST, request.FILES)
	context['retrievesimilarform'] = form
	if not form.is_valid():
		return load_dashboard(request,context)

	audio_file = form.cleaned_data['audio_file']

	new_entry = Audio(file=audio_file, isTemporary=True)
	new_entry.owner = request.user
	new_entry.organization = request.user.profile.org
	new_entry.save()

	audio_path = new_entry.file.path
	name = str(audio_path.split("/")[-1].split(".wav")[0] + "_" + request.user.username + "_temp")
	os.system("python vfl/dvec/extract_dvec.py "+audio_path+\
			  " "+name+ " vfl/dvec/dvecs/")

	dvec = np.load("vfl/dvec/dvecs/"+name+".npy")
	similar_audios, top_scores = dvec_compare.retrieve_similar(request, dvec, new_entry.id)
	gender = v2g.extract_gender(dvec)

	jitter, shimmer = v2j.compute_jitter_shimmer(audio_path)

	new_entry.params = json.dumps({'gender':gender, 
		'jitter':jitter, 
		'shimmer':shimmer})

	new_entry.save()
	context['similar_audios'] = similar_audios
	context['target_audio'] = [new_entry]
	context['top_scores'] = top_scores

	# Here we should remove the dvector.
	os.remove("vfl/dvec/dvecs/"+name+".npy")
	return load_dashboard(request,context)

def waveform_audio(request, id):
	fname = this_folder + "/gender/temp_wavs/temp.wav"
	f = open(fname, "rb")
	response = HttpResponse()
	response.write(f.read())
	response['Content-Type'] = 'audio/wav'
	response['Content-Length'] = os.path.getsize(fname)
	return response


def retrieve_audio(request,id):

	audio = Audio.objects.get(id=id)
	return HttpResponse(audio.file)


def clear_database(request):
	# A user can delete only his own files
    for audio in Audio.objects.filter(owner=request.user):
        audio.file.delete()
        audio.delete()
    return redirect(request.POST.get('next'))

@csrf_exempt
def texttospeech(request):
	speech_to_text = SpeechToTextV1(
		iam_apikey='ivcz4sw1451NvNDgU_9Jfc9y4EqIpo4Qmy8iW4X8x-xX',
		url='https://stream.watsonplatform.net/speech-to-text/api'
	)
	try:
		speech_recognition_results = speech_to_text.recognize(
			audio=request.FILES["audio"],
			content_type='audio/wav').get_result()
		data = speech_recognition_results
		text = data.get("results")[0].get("alternatives")[0].get("transcript")

		personality_insights = PersonalityInsightsV3(
			version='2018-09-20',
			iam_apikey='JoSYNcMGd-pWBUQV289Fv8gh0kFpH5_SDCENobZTruqA',
			url='https://gateway.watsonplatform.net/personality-insights/api'
		)

		try:
			text_file = open("file.txt", "r")
			profile = personality_insights.profile(
				content=text_file.read(),
				accept='application/json',
				content_type='text/plain'
			).get_result()

			results = {}

			personality = profile.get("personality")
			openness = personality[0].get("children")
			conscientiousness = personality[1].get("children")
			extraversion = personality[2].get("children")
			agreeableness = personality[3].get("children")
			neuroticism = personality[4].get("children")
			traits = [openness, conscientiousness, extraversion,
					  agreeableness, neuroticism]

			for trait in traits:
				for x in range(5):
					name = trait[x].get("name")
					percentile = trait[x].get("percentile")
					results[name] = percentile

			return HttpResponse(json.dumps( results ))

		except WatsonApiException as ex:
			print ("Method failed: " + ex.message + ": " +str(ex.code))
			return render(request, 'vfl/ser1.html')

	except WatsonApiException as ex:
		print ("Method failed with status code " + ex.message);
		return render(request, 'vfl/ser1.html')
