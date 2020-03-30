import os
import nltk
import copy
import inflect
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import OrderedDict
import numpy
import operator
import re
import sys

argumentList = sys.argv
p=inflect.engine()
dataset_path ='G:/IR/AASIGNMENTS/ASSIGNMENT_2/stories/stories'
#dataset_path='G:/IR/AASIGNMENTS/SRE'
stop_words = set(stopwords.words('english'))
directory = os.listdir(dataset_path)
document_list=[]
vocabs_list=[]
title_match_files=[]


def special_character(string):
    regex=re.compile('[@_!#$%^&*()<>?/\|}{~:3]')
    if(regex.search(string) == None):
        return 1
    else:
        return 0

def title_match(query):
    file_path='G:/IR/AASIGNMENTS/ASSIGNMENT_2/stories/stories/index.html'
    file=open(file_path,encoding='unicode_escape',mode='r')
    #query_1=''
    #query_1=' '.join(query)
    file_data = file.readlines()
    title_match_files=[]
    flag=0
    for line in file_data:
        c=0
        pos_1=[m.start() for m in re.finditer(r'\>', line)]
        #print(pos_1)
        if len(pos_1)!=0:
            line_1=line[pos_1[len(pos_1)-1]:]
        for tr in query:
            if tr in line_1.lower():
                flag=1
                c+=1
        if flag==1:
            #print(line)
            #if query_1.lower() in line.lower():
                #print(line)
            pos=[]
                #pos=re.findall(r'\"',line)
            pos=[m.start() for m in re.finditer(r'\"', line)    ]
            #print(pos)
            #print(line[pos[0]+1:pos[1]])
            if len(pos)>0 and c>0.4*len(query):
                title_match_files.append(line[pos[0]+1:pos[1]])
            flag=0
    return title_match_files

def preprocess(line):
    tokenizer = nltk.RegexpTokenizer('\w+')
    tokens_list = tokenizer.tokenize(line)
    for tokens in tokens_list:
        if tokens in stop_words:
            tokens_list.remove(tokens)
    lemmatizer = WordNetLemmatizer()
    words_list=[]
    for tokens in tokens_list:
        if tokens.isdigit() and len(tokens)<10 and special_character(tokens)==0:
            #print(tokens)
            word=p.number_to_words(int(tokens))
            digit_tokens=tokenizer.tokenize(word)
            for w in digit_tokens:
                if w not in stop_words:
                    words_list.append(lemmatizer.lemmatize(w))
        words_list.append(lemmatizer.lemmatize(tokens))

    words_list = [element.lower() for element in words_list]
    #print(words_list)
    return words_list


'''def title_match(query):
    file_path='G:/IR/AASIGNMENTS/ASSIGNMENT_2/stories/stories/index.html'
    file=open(file_path,encoding='unicode_escape',mode='r')
    #query_1=''
    query_1=' '.join(query)
    file_data = file.readlines()
    title_match_files=[]
    for line in file_data:
        if query_1.lower() in line.lower():
            #print(line)
            pos=[]
            #pos=re.findall(r'\"',line)
            pos=[m.start() for m in re.finditer(r'\"', line)]
            #print(pos)
            #print(line[pos[0]+1:pos[1]])
            title_match_files.append(line[pos[0]+1:pos[1]])
    return title_match_files

def preprocess(line):
    tokenizer = nltk.RegexpTokenizer('\w+')
    tokens_list = tokenizer.tokenize(line)
    for tokens in tokens_list:
        if tokens in stop_words:
            tokens_list.remove(tokens)
    lemmatizer = WordNetLemmatizer()
    words_list=[]
    for tokens in tokens_list:
        words_list.append(lemmatizer.lemmatize(tokens))

    words_list = [element.lower() for element in words_list]
    #print(len(words_list))
    return words_list'''

def calculate_query_vector(query):
    query_vector=[]
    for word in vocabs_list:
        idf_score=0
        if word in query:
            for doc in document_list:
                document= list(doc.keys())[0]
                vocabs=doc[document].keys()
                #if term in vocabs:
                if word in vocabs:
                    idf_score+=1
            query_vector.append(numpy.log(len(document_list)/(1+idf_score)))
        else:
            query_vector.append(0)
    return query_vector

def calculate_doc_vector(doc,query):
    doc_vector=[]
    for word in vocabs_list:
        tf_score=0
        if word in query:
            if word in doc.keys():
                doc_vector.append(doc[word])
            else:
                doc_vector.append(0)
        else:
            doc_vector.append(0)
    return doc_vector

def calculate_score(query_vector,doc_vector):
    #val =numpy.dot(query_vector,doc_vector)/(numpy.linalg.norm(query_vector)*numpy.linalg.norm(doc_vector))
    if (numpy.linalg.norm(query_vector)==0 or numpy.linalg.norm(doc_vector))==0 or numpy.dot(query_vector,doc_vector)==0 :
        return 0
    else:
        return numpy.dot(query_vector,doc_vector)/(numpy.linalg.norm(query_vector)*numpy.linalg.norm(doc_vector))
    #return numpy.dot(query_vector,doc_vector)/(numpy.linalg.norm(query_vector)*numpy.linalg.norm(doc_vector))

for subdir, dirs, files_list in os.walk(dataset_path):
    #files_list = os.listdir(dataset_path)
    for file in files_list:
        #document_list.append(file)
        file_path=os.path.join(subdir, file)
        file = open(file_path,encoding="unicode_escape",mode='r')
        file_data = file.readlines()
        fileName=file.name.split("/")[-2:]
        file_name=""
        file_name=file_name.join(fileName)
        doc_dict=dict()
        vocab_dict=dict()
        for line in file_data:
            procesed_word_list = preprocess(line)
            #print(procesed_word_list)
            for word in procesed_word_list:
                if word in vocab_dict.keys():
                    vocab_dict[word]+=1
                else:
                    if word not in vocabs_list:
                        vocabs_list.append(word)
                    vocab_dict[word]=1
        file.close()
        doc_dict[file_name]=vocab_dict
        document_list.append(doc_dict)

print("enter the query")
query=input().split()
print("enter the number of documents")
k=int(input())
query=preprocess(' '.join(query))
print(query)
final_doc_score_list=dict()
query_vector=calculate_query_vector(query)
for doc in document_list:
    doc_vector=calculate_doc_vector(doc[list(doc.keys())[0]],query)
    score=calculate_score(doc_vector,query_vector)
    final_doc_score_list[list(doc.keys())[0]]=score

final_doc_score_list=OrderedDict(sorted(final_doc_score_list.items(),key=operator.itemgetter(1),reverse=True))
title_match_files=title_match(query)
for i in title_match_files:
    print(i)
    k-=1
    if k==0:
        break;
if k>0:
    for i in final_doc_score_list.keys():
        if i.split("\\")[-1] not in title_match_files:
            print(i,final_doc_score_list[i])
            k-=1
            if k==0:
                break

