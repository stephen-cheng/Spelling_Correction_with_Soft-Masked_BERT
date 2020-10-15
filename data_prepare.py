import os
from os import listdir
from os.path import join
import re
import pandas as pd
import numpy as np


def load_data(path):
    input_file = os.path.join(path)
    with open(input_file) as f:
        book = f.read()
    return book


def clean_text(text):
    '''Remove unwanted characters and extra spaces from the text'''
    text = re.sub(r'\n', ' ', text) 
    text = re.sub(r'[{}@_*>()\\#%+=\[\]]','', text)
    text = re.sub('a0','', text)
    text = re.sub('\'92t','\'t', text)
    text = re.sub('\'92s','\'s', text)
    text = re.sub('\'92m','\'m', text)
    text = re.sub('\'92ll','\'ll', text)
    text = re.sub('\'91','', text)
    text = re.sub('\'92','', text)
    text = re.sub('\'93','', text)
    text = re.sub('\'94','', text)
    text = re.sub('\.','. ', text)
    text = re.sub('\!','! ', text)
    text = re.sub('\?','? ', text)
    text = re.sub(' +',' ', text)
    return text


def save_data(text_list, filename):
	with open(filename, 'w') as f:
		for line in text_list:
			f.write(line+'\n')
	print('Done')


if __name__ == "__main__":
	path = 'dataset/20books/'
	book_files = [f for f in listdir(path) if f.endswith('.rtf')]
	books = []

	# load data
	for book in book_files:
	    books.append(load_data(path+book))

	# clean data
	clean_books = []
	for book in books:
	    clean_books.append(clean_text(book))

	# split data
	sentences = []
	for book in clean_books:
	    for sentence in book.split('. '):
	        sentences.append(sentence + '.')
	print("There are {} sentences.".format(len(sentences)))

	# Limit the data we will use to train our model
	max_length = 200
	min_length = 4
	good_sentences = []
	for sentence in sentences:
	    if len(sentence) <= max_length and len(sentence) >= min_length:
	        good_sentences.append(sentence)
	print("{} sentences are used to our model.".format(len(good_sentences)))
	# save the data
	filename = 'dataset/sentences.txt'
	save_data(good_sentences, filename)

