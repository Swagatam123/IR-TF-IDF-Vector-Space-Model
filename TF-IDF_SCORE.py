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
from num2words import num2words

argumentList = sys.argv
p=inflect.engine()
dataset_path ='G:/IR/AASIGNMENTS/ASSIGNMENT_2/stories/stories'
#dataset_path='G:/IR/AASIGNMENTS/SRE'
stop_words = set(stopwords.words('english'))
directory = os.listdir(dataset_path)
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
            pos=[m.start() for m in re.finditer(r'\"', line)]
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

document_list=[]
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
                    vocab_dict[word]=1
        file.close()
        doc_dict[file_name]=vocab_dict
        document_list.append(doc_dict)
#print(document_list)
print("enter the query")
#query=sys.argv[1]
query=input().split()
print("enter the number of documents")
k=int(input())
#k=sys.argv[2]
term_doc_freq=dict()
query=preprocess(' '.join(query))
print(query)
for term in query:
    doc_freq=0
    for doc in document_list:
        document= list(doc.keys())[0]
        #print(document)
        vocabs=doc[document].keys()
        if term in vocabs:
            doc_freq+=1
    #print(doc_freq)
    term_doc_freq[term]=numpy.log(len(document_list)/(1+doc_freq))
#print(term_doc_freq)
final_doc_score_list=dict()
pos=0
for term in query:
    for doc in document_list:
        document= list(doc.keys())[0]
        vocabs=doc[document].keys()
        if term in vocabs:
            if document in final_doc_score_list:
                #print(doc[document][term])
                final_doc_score_list[document]+=term_doc_freq[term]*doc[document][term]
            else:
                #print(doc[document][term])
                final_doc_score_list[document]=term_doc_freq[term]*doc[document][term]
        pos+=1
final_doc_score_list=OrderedDict(sorted(final_doc_score_list.items(),key=operator.itemgetter(1),reverse=True))
#final_doc_list=[x for _,x in sorted(zip(final_doc_score_list,document_list))]
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
#print(title_match(['The', 'Story', 'of', 'the', 'Three', 'Wishes']))


