'''
Description: This program is written 
to compute dvectors, and the distance
between dvectors for each audio to show 
top 3 similar audios.

@author: Shahan A. Memon
'''

from vfl.models import *
from sklearn.metrics.pairwise import cosine_similarity
import json
'''
This function goes through all the audio files
and compares their dvectors with the dvector of the current 
audio and sends back top 3 similar audios.
'''

def retrieve_similar(request,target_dvec, target_id, top=3):
	scores = []
	org_audios = Audio.objects.filter(organization=request.user.profile.org)
	all_audios = org_audios.exclude(id=target_id)
	all_audios = all_audios.exclude(isAudio=False)
	if (len(all_audios) <= top):
		return all_audios, []

	for audio in all_audios:
		curr_dvec = json.loads(audio.dvector)
		cos_sim = cosine_similarity(target_dvec,curr_dvec)
		scores.append(cos_sim)
	top_scores = [scores[x] for x in sorted(range(len(scores)), 
					key=lambda i: scores[i])[-top:]]
	top_audios = [all_audios[x] for x in sorted(range(len(scores)), 
					key=lambda i: scores[i])[-top:]]
	print(top_scores)
	return list(reversed(top_audios)), list(reversed(top_scores))