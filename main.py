from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords 
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import csv 
import re 

reader = csv.reader(
	open("KomentarSample.csv"), delimiter=";")

katabaku = csv.reader(
	open("kata_baku.csv"), delimiter=";") # ambil file csv kata baku menjadi array

kamus_katabaku = {} # empty dictionary untuk kamus kata baku

for row in katabaku : # membuat kamus kata baku dengan input kata tidak baku dan output kata bakunya
	kamus_katabaku[row[1]] = row[0]

komentar = [] # list berisikan semua komentar. 

listStopword = set(stopwords.words('indonesian')) #list kataa kata yang tidak bermakana dalam bahasa indonesia
factory = StemmerFactory()
stemmer = factory.create_stemmer()

for row in reader : #melakukan perulangan pada setiap baris komentar	
	
	print("SEBELUM : ")
	print(row[1])
	print()

	kom = re.sub('[^A-Za-z]+',' ', row[1]) # cleansing (regex) mengahpus tanda baca dan angka
	kom = kom.lower()# case folding (semua ke lower case)

	tokens = word_tokenize(kom) #tokenize, kalimat jadi array kata 

	removed = []
	for t in tokens:  #loop nyebutin setiap kata pada kalimat 
		
		try : 
			t = kamus_katabaku[t] # proses normalisasi, pemetaan kata non baku ke baku.
		except :
			pass  

		# negation handling (besok)
		
		if t not in listStopword: # jika kata itu gaada di listStopword berarti kata penting
			removed.append(t)

	removed = " ".join(removed)
	katadasar = stemmer.stem(removed)


	print("SESUDAH : ")
	print(removed) #kenapa print ditaro disini biar setiap proses dilangsung mengularkan output dan kebaris selanjutnya
	print("-----------------------------------")

	komentar.append(removed) 

#komentar.pop(0) # menghapus judul kolom pada file csv



