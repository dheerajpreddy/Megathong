#!/usr/bin/env python
# coding: utf-8
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import spacy
nlp = spacy.load('en')

# Add neural coref to SpaCy's pipe
import neuralcoref
# # neuralcoref.add_to_pipe(nlp)
coref = neuralcoref.NeuralCoref(nlp.vocab)
nlp.add_pipe(coref, name='neuralcoref')

# doc = nlp('My sister has a dog. She loves him.')
# print (doc)

# print(doc._.has_coref)
# print(doc._.coref_clusters)

def read_article(file_name):
    stop_words = stopwords.words('english')

    file = open(file_name, "r")
    filedata = file.readlines()
    article = filedata[0].split(". ")
    sentences = []

    for sentence in article:
        # print(sentence)
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop() 

    # print (sentences)
    text = []
    for i in sentences:
        temp = ""
        for j in i:
            temp += str(j) + " "
        text.append(temp[:-1])
    # print (text)
    # print(" ".join(sentences))
    print (len(text))
    new = []
    for i in range(2,len(text)):
        doc = nlp(text[i-2] + text[i-1] + text[i])
        # doc = nlp('My sister has a dog. She loves him.')

        if doc._.has_coref:
            clust = doc._.coref_clusters
            doc = str(doc)

            for c in range(len(clust)):
                cc = clust[c].mentions
                imp = ""
                for j in cc:
                    j = str(j)
                    if j.lower() not in stop_words:
                        imp = j
                        continue
                    text[i] = (text[i].replace(j,imp))
        new.append(text[i])
    # print (new)
    return new

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def generate_summary(file_name, top_n=5):

    # global stopwords
    stop_words = stopwords.words('english')

    summarize_text = []

    # Step 1 - Read text anc split it
    sentences =  read_article(file_name)

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    # print("Indexes of top ranked_sentence order are ", ranked_sentence)    

    for i in range(top_n):
        temp = "".join(ranked_sentence[i][1])
        if (temp[-1] == '?') or temp.find('?') != -1:
            # print (temp)
            continue
        elif temp[-1] != '.':
            temp += '.' 

        summarize_text.append(temp)

    # Step 5 - Offcourse, output the summarize texr
    # print("Summarize Text: \n", ". ".join(summarize_text))
    print ("Summarized:")
    for i in summarize_text:
        print (i)

# let's begin
generate_summary( "/home/kripa/Documents/megathon/Output format/9_biology_6.txt", 30)