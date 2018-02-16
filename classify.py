'''
Author: Mohit Mangal
Email: mohit.mangal@innoplexus.com
Phone: +91-9530028338

Description: This script contains classifier code can be used for patch classification extracted from pdfs
'''
# -*- coding: utf-8 -*-
import os,csv
from nltk.tag import StanfordNERTagger
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
import pickle
import re
from operator import itemgetter
from pprint import pprint
import config

STANFORD_NER_TAGGER_PATH1 = config.PATH1_STANFORD_NER_TAGGER
STANFORD_NER_TAGGER_PATH2 = config.PATH2_STANFORD_NER_TAGGER
st=StanfordNERTagger(STANFORD_NER_TAGGER_PATH1,STANFORD_NER_TAGGER_PATH2)


'''
Following class contains functions for feature extraction, training and testing.
'''
class Classify():
	'''Following init function initializes variables required at the time of training and testing'''
	def __init__(self):

		self.classNumbers = 0
		#self.classes = {'Noise':0,'Date':1,'Title':2,'Author':3,'Affiliation':4,'Superviser':5,'Department':6,'AbsStart':7,'AbsEnd':8,'Keywords':9}
		#self.reverseClasses = {0:'Noise',1:'Date',2:'Title',3:'Author',4:'Affiliation',5:'Superviser',6:'Department',7:'AbsStart',8:'AbsEnd',9:'Keywords'}
		self.classes = {}
		self.reverseClasses = {}
		self.nerMap = {}
		self.firstWords = []

		self.trainHeaders = config.TRAIN_HEADERS

		self.fontSizeKey = config.FONT_SIZE_KEY
		self.fontSizeValue = config.FONT_SIZE

		self.pageNumberKey = config.PAGENUMBER_KEY
		self.pageNumber = config.PAGENUMBER

		self.patchPredictedTextKey = config.PATCH_PREDICTED_TEXT_KEY
		self.patchPredictedText = config.PATCH_PREDICTED_TEXT

		self.ClassKey = config.CLASS_KEY
		self.Class = config.CLASS

		self.modelFilename = config.MODEL_FILENAME
		self.mappingFilename = config.MAPPING_FILENAME
		self.firstWordsFilename = config.FIRSTWORDS_FILENAME


	'''Following function extracts features from given string'''
	def getPercentage(self,string):
		try:
			firstWordId = self.firstWords.index(string.lower().split()[0])
		except:
			firstWordId = -1
		upperCase = 0
		digits = 0
		commas = 0
		lineBreaks = 0
		newStr = ""
		for char in string:
			if char.isupper():
				upperCase += 1
			if char.isdigit():
				digits += 1
			if char==',':
				commas += 1
			if char=='\n':
				lineBreaks += 1
			if char.isalpha() or char==' ':
				newStr += char
			else:
				newStr += ' '

		percentageCapitalCase = (float(upperCase)/float(len(string)))*100
		percentageDigits = (float(digits)/float(len(string.split())))*100
		percentageCommas = (float(commas)/float(len(string.split())))*100
		lineBreaks = (float(lineBreaks)/float(len(string.split())))*100
		yearPresent = 0
		univPresent = 0
		schoolPresent = 0
		abstractPresent = 0
		keyWordPresent = 0
		monthPresent = 0
		months = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
		tags = []
		if len(newStr.split())>20:
			for word in newStr.split():
				tags += [(word,'O')]
		else:
			tags = st.tag(newStr.split())
		noWords = len(newStr.split())
		noTags = 0
		noPersonTag = 0
		for tag in tags:
			try:
				temp = int(tag[0])
				if temp>1950 and temp <2050:
					yearPresent = 1
			except:
				pass
			if 'universit' in tag[0].lower() or 'institu' in tag[0].lower() or 'colleg' in tag[0].lower():
				univPresent = 1
			if 'school' in tag[0].lower() or 'depart' in tag[0].lower():
				schoolPresent = 1
			if tag[0].lower().startswith('abstract'):
				abstractPresent = 1
			if tag[0].lower().startswith('keyword'):
				keyWordPresent = 1
			if tag[0].lower()=='may' or (tag[0].lower()[:4] in months):
				monthPresent = 1
			if tag[1]!='O':
				noTags += 1
			if tag[1]=='PERSON':
				noPersonTag += 1
		try:
			percentageTagged = (float(noTags)/float(noWords))*100
			percentageTaggedPerson = (float(noPersonTag)/float(noWords))*100
		except:
			percentageTagged = 0
			percentageTaggedPerson = 0

		return percentageCapitalCase, percentageDigits, percentageCommas, lineBreaks, percentageTagged, percentageTaggedPerson, yearPresent, univPresent, schoolPresent, abstractPresent, keyWordPresent,firstWordId,monthPresent
	'''Following function was meant to be used for optimizing NER tagging but not used currently'''
	def nerTag(self, strings):
		strings1 = []
		for string in strings:
			str1 = ""
			for char in string:
				if char.isalpha() or char==' ':
					str1 += char
			strings1 += [str1]
		finalString = ' @@@@@@@@@@@@@@@@@ '.join(strings1)
		tags =  st.tag(finalString.split())
		count = 0
		tempTags = []
		for tag in tags:
			if tag[0]=='@@@@@@@@@@@@@@@@@':
				self.nerMap[count] = tempTags[:]
				tempTags = []
				count += 1
				continue
			tempTags += [tag]
	'''Following function is used for training'''
	def train(self, trainDirName, saveModel=False, saveModelPath=None):

		trainingFiles = [f for f in listdir(trainDirName) if isfile(join(trainDirName, f))]

		trainingX = []
		trainingY = []
		fileNumber= 0
		count = 0

		for fileName in trainingFiles:

			print "Executing file %d: %s" %(count, fileName)
			count += 1
			fp = open(trainDirName+"/"+fileName,"r")
			reader = csv.DictReader(fp)
			patches = []

			self.nerMap = {}
			trainingXCurrent = []
			trainingYCurrent = []

			filepath = join(trainDirName,fileName)
			fp = open(filepath,"r")

			reader = csv.DictReader(fp)
			rowcount=1
			for row in reader:
				print "Executing patch %d" %(rowcount)
				Class = row.get(self.ClassKey, self.Class)

				if Class not in self.classes:
					self.classes[Class] = self.classNumbers
					self.reverseClasses[self.classNumbers] = Class
					self.classNumbers += 1
				y=self.classes[Class]

				patchPredictedText = row.get(self.patchPredictedTextKey,self.patchPredictedText)

				text = patchPredictedText
				text = text.split()

				if text and text[0] not in self.firstWords:
					self.firstWords += [text[0].lower()]

				percentageCapitalCase,percentageDigits,percentageCommas,\
				lineBreaks,percentageTagged,percentageTaggedPerson,\
				yearPresent, univPresent, schoolPresent, \
				abstractPresent, keyWordPresent,firstWordId,\
				monthPresent = self.getPercentage(patchPredictedText)

				fontSize = row.get(self.fontSizeKey,self.fontSizeValue)
				try:
					fontSize = int(fontSize)
				except:
					fontSize = 0

				pageNumber = row.get(self.pageNumberKey,self.pageNumber)
				try:
					pageNumber = int(pageNumber)
				except:
					pageNumber = -1

				x = [fileNumber,percentageCapitalCase,percentageDigits,\
					percentageCommas,lineBreaks,percentageTagged,\
					percentageTaggedPerson,yearPresent, univPresent,\
					schoolPresent, abstractPresent, keyWordPresent,\
					firstWordId,monthPresent,\
					len(row['patchPredictedText'].split()),fontSize,pageNumber]

				trainingXCurrent += [np.array(x)]
				trainingYCurrent += [y]
				rowcount+=1
			fp.close()

			'''
			generating 30D vector for each row in the csv file
			'''
			#Handling first row
			trainingXCurrent1 = []
			temp = np.concatenate((trainingXCurrent[0],np.zeros(len(trainingXCurrent[0]))))
			temp = np.concatenate((temp,trainingXCurrent[1]))
			trainingXCurrent1 += [temp]
			for i in xrange(1,len(trainingXCurrent)-1):
				temp = np.concatenate((trainingXCurrent[i],trainingXCurrent[i-1]))
				temp = np.concatenate((temp,trainingXCurrent[i+1]))
				trainingXCurrent1 += [temp]
			#Handling the last row
			temp = np.concatenate((trainingXCurrent[len(trainingXCurrent)-1],trainingXCurrent[len(trainingXCurrent)-2]))
			temp = np.concatenate((temp,np.zeros(len(trainingXCurrent[0]))))
			trainingXCurrent1 += [temp]


			trainingX += trainingXCurrent1
			trainingY += trainingYCurrent
			fileNumber = (fileNumber+1)%2
		self.clf1 = RandomForestClassifier(n_estimators=100)

		self.clf = self.clf1.fit(np.array(trainingX), np.array(trainingY))

		trainingAccuracy = self.clf.score(np.array(trainingX), np.array(trainingY))

		cv_scores = cross_val_score(self.clf1,np.array(trainingX),np.array(trainingY),scoring="accuracy",cv=10,n_jobs=-1)

		#give a path in here to dump the .pkl file (saveModel)
		if saveModel:
			print "saveModel: ", saveModel
			if saveModelPath:
				print "saveModelPath: ", saveModelPath
				print os.path.isdir(saveModelPath)
				raw_input("halt?")
				if os.path.isdir(saveModelPath):
					modelPath = saveModelPath
				else:
					print "path invalid"
					raw_input("halt?")
					modelPath = os.getcwd()
			else:
				modelPath=os.getcwd()

			savePath = os.path.join(modelPath,self.modelFilename)
			joblib.dump(self.clf, savePath)

			mappingPath = os.path.join(modelPath,self.mappingFilename)
			mapping = pickle.dumps(self.reverseClasses)

			with open(mappingPath,"w") as fp1:
				fp1.write(mapping)

			firstWordsPath = os.path.join(modelPath,self.firstWordsFilename)
			with open(firstWordsPath,"w") as fp1:
				fp1.write(str(self.firstWords))

		'''
		if --train and --test are given at the same time, return the model instance (the .pkl file) also
		'''

		print "completed !!!!"
		return str((float(sum(cv_scores))/len(cv_scores))*100),\
				str(trainingAccuracy*100), self.clf,\
				self.reverseClasses,self.firstWords

	'''Following function is used to get output on testData'''
	def getOutput(self,testData, modelFolderpath=None, trainedModel=None, mappingObject=None, firstWords=None):

		if trainedModel and mappingObject and firstWords:
			clf = trainedModel
			mapping = mappingObject
			self.firstWords = firstWords
		else:
			if modelFolderpath:
				if os.path.isdir(modelFolderpath):
					modelFiles = os.listdir(modelFolderpath)
					if self.modelFilename in modelFiles and self.mappingFilename in modelFiles and self.firstWordsFilename in modelFiles:
						clf = joblib.load(os.path.join(modelFolderpath, self.modelFilename))
						mapping = {}
						with open(os.path.join(modelFolderpath, self.mappingFilename),"r") as fp:
							mapping = pickle.loads(fp.read())
						self.firstWords = []
						with open(os.path.join(modelFolderpath, self.firstWordsFilename),"r") as fp:
							self.firstWords = eval(fp.read())

		testX = []
		testX1 = []
		testY = []
		patches = []
		fp = open(testData,"r")
		reader = csv.DictReader(fp)
		rowcount=1

		for row in reader:
			print "Executing patch %d" %(rowcount)
			rowcount+=1
			patchPredictedText = row.get(self.patchPredictedTextKey,self.patchPredictedText)
			percentageCapitalCase,percentageDigits,percentageCommas,\
			lineBreaks,percentageTagged,percentageTaggedPerson,\
			yearPresent, univPresent, schoolPresent, abstractPresent,\
			keyWordPresent,firstWordId,\
			monthPresent = self.getPercentage(patchPredictedText)
			fontSize = row.get(self.fontSizeKey,self.fontSizeValue)
			pageNumber = row.get(self.pageNumberKey,self.pageNumber)
			try:
				fontSize = int(fontSize)
			except:
				fontSize = 0
			try:
				pageNumber = int(pageNumber)
			except:
				pageNumber = -1

			x = [0,percentageCapitalCase,percentageDigits,percentageCommas,\
			lineBreaks,percentageTagged,percentageTaggedPerson,\
			yearPresent, univPresent, schoolPresent, abstractPresent,\
			keyWordPresent,firstWordId,monthPresent,\
			len(patchPredictedText.split()),fontSize,pageNumber]
			print x
			testX1 += [x]
			patches += [patchPredictedText]

		temp = np.concatenate((np.array(testX1[0]),np.zeros(len(testX1[0]))))
		temp = np.concatenate((temp,np.array(testX1[1])))

		testX += [temp]

		for i in xrange(1,len(testX1)-1):
			temp = np.concatenate((testX1[i],testX1[i-1]))
			temp = np.concatenate((temp,testX1[i+1]))
			testX += [temp]

		temp = np.concatenate((testX1[len(testX1)-1],testX1[len(testX1)-2]))
		temp = np.concatenate((temp,np.zeros(len(testX1[0]))))
		testX += [temp]
		testX = np.array(testX)
		for t in testX:
			t=t.astype(np.float)

		y = clf.predict(testX)

		return patches,y,mapping



	'''Following function was used to get articles from pdf thesis. Currently this is not in use'''
	def getArticles(self,testData):
		#os.system("rm testDataCleaned/*")
		clf = joblib.load('/home/niyati.tiwari/Documents/pdfExtractor/classifier_congress/trained.pkl')
		mapping = {}
		with open("/home/niyati.tiwari/Documents/pdfExtractor/classifier_congress/mapping.pkl","r") as fp:
			mapping = pickle.loads(fp.read())
		self.firstWords = []
		with open("/home/niyati.tiwari/Documents/pdfExtractor/classifier_congress/firstWords","r") as fp:
			self.firstWords = eval(fp.read())
		testHeaders = ['pageNumber','Patch Number','patchPredictedText','font','fontSize','Class']
		testX = []
		testX1 = []
		testY = []
		patches = []
		#testFile = [f for f in listdir("testDataCleaned") if isfile(join("testDataCleaned", f))]
		fp = open(testData,"r")
		reader = csv.DictReader(fp)
		for row in reader:
			percentageCapitalCase,percentageDigits,percentageCommas,lineBreaks,percentageTagged,percentageTaggedPerson,yearPresent, univPresent, schoolPresent, abstractPresent, keyWordPresent,firstWordId,monthPresent = self.getPercentage(row['patchPredictedText'])
			fontSize = row['fontSize']


			try:
				fontSize = int(fontSize)
			except:
				fontSize = 0
			pageNumber = row['pageNumber']
			try:
				pageNumber = int(pageNumber)
			except:
				pageNumber = -1
			x = [0,percentageCapitalCase,percentageDigits,percentageCommas,lineBreaks,percentageTagged,percentageTaggedPerson,yearPresent, univPresent, schoolPresent, abstractPresent, keyWordPresent,firstWordId,monthPresent, len(row['patchPredictedText'].split()),fontSize,pageNumber]
			print x
			testX1 += [x]
			patches += [row['patchPredictedText']]
		temp = np.concatenate((testX1[0],np.zeros(len(testX1[0]))))
		temp = np.concatenate((temp,testX1[1]))
		testX += [temp]
		for i in xrange(1,len(testX1)-1):
			temp = np.concatenate((testX1[i],testX1[i-1]))
			temp = np.concatenate((temp,testX1[i+1]))
			testX += [temp]
		temp = np.concatenate((testX1[len(testX1)-1],testX1[len(testX1)-2]))
		temp = np.concatenate((temp,np.zeros(len(testX1[0]))))
		testX += [temp]
		testX = np.array(testX)
		y = clf.predict(testX)
		absStarted = False
		absTmp = ""
		title = []
		author = []
		superviser = []
		date = []
		affiliation = []
		department = []
		abstract = []
		keyWords = []
		last = "None"
		print "-------------------------------------------------------------"
		for i in xrange(0,len(y)):
			#print patches[i]
			patch1 = ''.join([char for char in patches[i] if char.isalnum() or char==' '])
			#print "**************"
			result = y[i]
			'''if mapping[result]!='Noise':
				print patches[i]+"---------->"+mapping[result]'''
			if len(patches[i].strip())<5 and (' ' not in patches[i].strip()):
				continue
			if absStarted:
				absTmp += patches[i]
				if patches[i].split()<4:
					absStarted = False
					break
			elif mapping[result]=="AbsStart":
				absStarted = True
			elif mapping[result]=="AbsEnd" or (absStarted and len(patch1.split())<4):
				abstract += [absTmp]
				absStarted = False
				absTmp = ""
			elif mapping[result]=="Title":
				if last=="Title" and (patches[i] not in title[-1][0]):
					title[-1][0] += (" "+patches[i])
					title[-1][1] += 1
				else:
					title += [[patches[i],1]]
					last="Title"
			elif mapping[result]=="Author":
				author += [patches[i]]
				last="Author"
			elif mapping[result]=="Superviser":
				superviser += [patches[i]]
				last="Superviser"
			elif mapping[result]=="Department":
				department += [patches[i]]
				last="Department"
			elif mapping[result]=="Affiliation":
				affiliation += [patches[i]]
				last="Affiliation"
			elif mapping[result]=="Date":
				date += [patches[i]]
				last="Date"
			if mapping[result]=="Keywords":
				keyWords += [patches[i]]
				last="Keywords"
		if absTmp!="":
			abstract += [absTmp]
		if title == []:
			title += [[patches[0],1]]
		title1 = sorted(title,key=itemgetter(1),reverse=True)[0][0]
		try:
			if author==[]:
				author = superviser[:]
			author1 = author[0]
		except:
			author1 = ""
		try:
			affiliation1 = ""
			affiliation.sort(key = len)
			if any('university' in aff.lower() for aff in affiliation):
				for aff in affiliation[::-1]:
					if 'university' in aff.lower():
						affiliation1 = aff
						break
			elif any('institute' in aff.lower() for aff in affiliation):
				for aff in affiliation[::-1]:
					if 'institute' in aff.lower():
						affiliation1 = aff
						break
			else:
				affiliation1 = affiliation[-1]
		except:
			affiliation1 = ""
		try:
			department.sort(key = len)
			department1 = department[-1]
		except:
			department1 = ""
		try:
			superviser = list(set(superviser))
		except:
			superviser = []
		keyWords = ','.join(keyWords)
		try:
			date1 = date[0]
			months = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
			for d1 in date:
				if any(mon in d1.lower() for mon in months):
					date1 = d1
					break
		except:
			date1 = ""
		try:
			abstract = ' '.join(abstract[0].split()[:350])
		except:
			abstract = ""
		out = {'title':title,'title1':title1,'author':author,'author1':author1,'affiliation':affiliation,'affiliation1':affiliation1,'department':department,'department1':department1,'superviser':superviser,'date':date,'date1':date1,'abstract':abstract,'keyWords':keyWords,"fileName":sys.argv[1]}
		return out
