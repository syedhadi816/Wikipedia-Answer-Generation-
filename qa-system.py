#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


'''
Name of the project :: QA System.

Problem statement :: This QA system should be able to answer Who, What, When and Where questions
(but not Why or How questions) and  provide the information only what is asked for by the question. below is the sample 
question and answers for that.

Sample input : 
 " python qa-system_lts.py  qa-log.txt "
 
 ** Note that log file path need to mention with space separted format.
 
Sample outputs : 

[You: ]
where is george mason university? 
[Output]:  george mason university is  in fairfax county
[You: ]
what is eliza?
[Output]:  Eliza is an early natural language processing computer program created from 1964 to 1966 at the mit artificial intelligence laboratory by joseph weizenbaum.
[You: ]
when is earth day?
[Output]:  Earth day is  in april 22
[You: ]
what is sirius?
[Output]:  Sirius is the brightest star in the night sky.

Algorithm : 
step 1 : User Input is taken, This query is checked for a number of if conditions to match the pattern. For example what is, what are, what did, where is, where was and so on for each of the who, where and when conditions. 
Step 2:  Once a pattern has been found the user query is pushed forward to a reformulate function. Now, the reformulate function takes the user query and it changes the structure of this query so that it resembles more of that it resembles the structure of an answer rather than a question. So the reformulate function would restructure the user statement.
Step 3: Reformulated query cleaned for better results.
Step 4: Wikiclean function
Step 5: Wikimatch function ( matches general information such as time and location). Outputs list of relavent results and their pages. Content of this page is matched with the user query that is reformulated.
Step4: Search through the sentences for relevant answers using object and regular expressions.
Step 7: Return the most relavent answer
step 8 : this system also picks the correct answer for a given question, depending on whether the answer is looking for a place or a date and so on. So if the answer is someone's birthday, for instance, then the code will be looking for a date within the answer and append that at the end of the reformulated sentence.

****************
Extra credit :: We have impelemented partial match of answers, partial match of patterns with Wikipedia information. System able to perform partial match with date and places.

sample output for extra credit :

[You]: when was apple founded?
[Output]:  Apple was founded in april 1, 1976

[You]: when was john adams born?
[Output]:  John adams was born in october 30, 1735

'''

# -*- coding: utf-8 -*-
# importing all required libraries
import re
import wikipedia
import random
import spacy
from nltk import pos_tag
import numpy as np
import sys

##Initialize the objects here
nlp = spacy.load("en_core_web_sm")
raw_wiki = []
raw_wiki.append('\n \n')
raw_wiki.append('\n \n')
q = []
q.append('\n \n')
q.append('\n  \n')
a = []
a.append('\n  \n')
a.append('\n \n')

## Function for converting to lower case
def take_inp():
    print('[You: ]')
    inp = input()
    inp = inp.lower()
    inp = re.sub(r'[?]', '', inp)
    return (inp)

## Function to clean and preprocess the wiki sentences
def clean_wiki(string):
    string = string.lower() #convert to lower case
    string = string.split('. ') #split into sentences

    def rem_bkt(string):
        #remove text within brackets
        return (re.sub("[\(\[].*?[\)\]]", "", string))

    def rem_spcl(string):
        #remove special characters
        #string = re.sub(r'[^\w\s]', '', string)
        return(string)

    def rem_spce(string):
        #remove double spaces
        string = re.sub(' +', ' ', string)
        return(string)

    string = [rem_spce(rem_spcl(rem_bkt(j))) for j in string]
    string = '. '.join(string)
    
    return(string)

## Function to match the wiki sentences with query    
def wiki_match(key):
    result = wikipedia.search(key)
    for i in range(len(result)):

        try:
            page = wikipedia.page(result[i], auto_suggest=False)
        except wikipedia.DisambiguationError as e:
            s = random.choice(e.options)
            page = wikipedia.page(s, auto_suggest=False)
        #page = wikipedia.page(result[i], auto_suggest=False)
        string = page.content
        string = clean_wiki(string)
        if string.find(key) != -1:
            text = string
            ind = string.find(key)
            raw_wiki.append(page)
            break
        else:
            return('Sorry, I could not find an answer to that.')
    ind_2 = min(text[ind:].find(', '), text[ind:].find('. '))
    string = (string[ind:ind+ind_2]+'.').capitalize()

    return(string)

## function to reformulating the the when question
def reformulate_when(keyword):

    if keyword[0][1].find('born') == -1:
        tokens_tag = pos_tag(keyword[0][1].split())

        tags = []
        found = []
        for i in range(len(tokens_tag)):
            k = tokens_tag[i][1]
            if k in ['NN' , 'NNS' , 'NNP' , 'NNPS']:
                if len(tokens_tag)>i+1:
                    if tokens_tag[i+1][1] not in ['NN' , 'NNS' , 'NNP' , 'NNPS']:
                        found.append(i)
                        prior = ' '.join(keyword[0][1].split()[:found[0]+1])
                        post = ' '.join(keyword[0][1].split()[found[0]+1:])
                        break
                else:
                    prior = keyword[0][1]
                    post = ''

    else:
        prior = keyword[0][1][:keyword[0][1].find('born')-1]
        post = keyword[0][1][keyword[0][1].find('born'):]

    
    conjugate = ' ' + keyword[0][0]
    if (conjugate == ' did'):
        prior+='ed'
        conjugate = ' '
        
    out = prior + conjugate + ' ' +post
    out = re.sub(' +', ' ', out)
    return(out)

## function to match with date related question
def wiki_match_time(key):
    posts = [' in', ' on', ' around']
    found = 0
    for post in posts:
        if found == 1:
            break
        result = wikipedia.search(key + post)
        for i in range(len(result)):

            try:
                page = wikipedia.page(result[i], auto_suggest=False)
            except wikipedia.DisambiguationError as e:
                s = random.choice(e.options)
                page = wikipedia.page(s, auto_suggest=False)
            #page = wikipedia.page(result[i], auto_suggest=False)
            string_ = page.content
            string = clean_wiki(string_)
            if string.find(key) != -1:
                text = string
                ind = string.find(key)
                raw_wiki.append(page)
                found = 1
                break
            else:
                return('Sorry, I could not find an answer to that.')
        if found == 0:
            return('Sorry, I could not find an answer to that.')
        ind_2 = text[ind:].find('. ')
        string = (string[ind:ind+ind_2]+'.').capitalize()
        ner = nlp(string)
        posterior = '-1'
        for word in ner.ents:
            if word.label_ == 'DATE':
                posterior = word.text
                break
        if posterior == '-1':
            return('Sorry, I could not find an answer to that.')
        output = (key.capitalize() + post +  ' ' +posterior)
        

    return(output)

## function to reformulating the the where question
def reformulate_where(keyword):

    if keyword[0][1].find('born') == -1:
        tokens_tag = pos_tag(keyword[0][1].split())

        tags = []
        found = []
        for i in range(len(tokens_tag)):
            k = tokens_tag[i][1]
            if k in ['NN' , 'NNS' , 'NNP' , 'NNPS']:
                if len(tokens_tag)>i+1:
                    if tokens_tag[i+1][1] not in ['NN' , 'NNS' , 'NNP' , 'NNPS']:
                        found.append(i)
                        prior = ' '.join(keyword[0][1].split()[:found[0]+1])
                        post = ' '.join(keyword[0][1].split()[found[0]+1:])
                        break
                else:
                    prior = keyword[0][1]
                    post = ''

    else:
        prior = keyword[0][1][:keyword[0][1].find('born')-1]
        post = keyword[0][1][keyword[0][1].find('born'):]
    
    priors_pat = ['is located', 'is found', 'can be found']
    priors_lst = []
    for p in priors_pat:
        s = prior + ' ' + p
        priors_lst.append(s)
    s = 'The address of ' + prior
    priors_lst.append(s)
    
    conjugate = ' ' + keyword[0][0]
    if (conjugate == ' did'):
        prior+='ed'
        conjugate = ' '
        
    out = prior + conjugate + ' ' +post
    out = re.sub(' +', ' ', out)
    
    priors_lst.append(out)
    priors_out = []
    for j in priors_lst:
        for k in [' is', ' in', ' on', ' around', ' at']:
            priors_out.append(re.sub(' +', ' ', (j+k)))
    
    return(priors_out, out)

## function to match the location realted question
def wiki_match_place(keys):
    found = 0
    for key in keys:
        if found == 1:
            break
        result = wikipedia.search(key)
        for i in range(len(result)):

            try:
                page = wikipedia.page(result[i], auto_suggest=False)
            except wikipedia.DisambiguationError as e:
                s = random.choice(e.options)
                page = wikipedia.page(s, auto_suggest=False)
            #page = wikipedia.page(result[i], auto_suggest=False)
            string = page.content
            string = clean_wiki(string)
            if string.find(key) != -1:
                text = string
                ind = string.find(key)
                raw_wiki.append(page)
                found = 1
                break
            else:
                return('Sorry, I could not find an answer to that.')
        if found == 0:
            return('Sorry, I could not find an answer to that.')
        ind_2 = text[ind:].find('. ')
        string = (string[ind:ind+ind_2]+'.').capitalize()
        ner = nlp(string)
        posterior = '-1'
        for word in ner.ents:
            if word.label_ == 'GPE':
                posterior = word.text
                break
        if posterior == '-1':
            return ('Sorry, I could not find an answer to that.')
        output = (key.capitalize() +  ' ' +posterior)
        

    return(output)

## function to match the location realted question
def partial_match_place(key):
    result = wikipedia.search(key)
    for i in range(len(result)):
        try:
            page = wikipedia.page(result[i], auto_suggest=False)
        except wikipedia.DisambiguationError as e:
            s = random.choice(e.options)
            page = wikipedia.page(s, auto_suggest=False)
        #page = wikipedia.page(result[i], auto_suggest=False)
        string = page.content
        string = clean_wiki(string)
        #string = string.split('.')
        ner = nlp(string)
        posterior = '-1'
        for word in ner.ents:
            if word.label_ == 'GPE':
                raw_wiki.append(page)
                posterior = word.text
                break
        if posterior != '-1':
            break
    if posterior == '-1':
        return ('Sorry, I could not find an answer to that.')
    
    out = key + ' in ' + posterior
    return (out)


#function to generate the what question answers
def what(inp):
    pattern = re.compile('what (is|was|were)\s([-\w\W]*)')
    keyword = pattern.findall(inp)
    response = keyword[0][1] + ' ' + keyword[0][0]
    response = wiki_match(response)
    return (response)

#function to generate the who question answers
def who(inp):
    pattern = re.compile('who (is|was|were)\s([-\w\W]*)')
    keyword = pattern.findall(inp)
    response = keyword[0][1] + ' ' + keyword[0][0]
    response = wiki_match(response)
    return (response)

#function to generate the when question answers
def when(inp):
    pattern = re.compile('when (is|was|were|did)\s([-\w\W]*)')
    keyword = pattern.findall(inp)
    response = reformulate_when(keyword)
    response = wiki_match_time(response)
    return (response)

#function to generate the where question answers
def where(inp):
    pattern = re.compile('where (is|was|were|did|are)\s([-\w\W]*)')
    keyword = pattern.findall(inp)
    response, prior = reformulate_where(keyword)
    response_1 = wiki_match_place(response)
    if response_1 == ('Sorry, I could not find an answer to that.'):
        response_2 = partial_match_place(prior)
        return (response_2)
    else:
        return (response_1)
    
## function to initiate the system 
def get_started(inp = 'start'):
    
    
    #a.append(response)

    while not re.findall(r'[Ee]xit', inp):

        inp = take_inp()
        q.append(inp)
        if re.findall(r'[Ee]xit', inp):
            break

        elif re.findall(r'what (is|was|were) (.*)', inp):
            response = what(inp)
            a.append(response)
            print('[Output]: ',response)

        elif re.findall(r'who (is|was|were) (.*)', inp):
            response = who(inp)
            a.append(response)
            print('[Output]: ',response)

        elif re.findall(r'when (is|was|were|did) (.*)', inp):
            response = when(inp)
            a.append(response)
            print('[Output]: ',response)

        elif re.findall(r'where (is|was|were|did|are) (.*)', inp):
            response = where(inp)
            a.append(response)
            print('[Output]: ',response)
        else:
            a.append('Sorry, I could not find an answer to that.')
            raw_wiki.append('no page found')
            print('Sorry, I could not find an answer to that.')


# In[ ]:

## main logging function goes here 
def main():
    '''
    This is the main function.
    '''
    #Fetch arguments in variables
    log_file_name = str(sys.argv[1]) #name of log file
    print(log_file_name)


    print("This is a QA system by Syed, Aiden, Akash and Vaish. It will try to answer questions that start with Who, What, When or Where. Enter exit to leave the program.")

    #initiate the QA function 
    get_started()
    
    grt_txt = " \n *** This is a QA system by Syed, Aiden, Akash and Vaish. It will try to answer questions that start with Who, What, When or Where. Enter exit to leave the program.*** \n"
    #Write the log file
    with open(log_file_name, 'w') as f:
        f.write(grt_txt)
#         f.write(log_txt)
        for k,j in zip(q,a):
            f.write('=?>'+ str(k)+'\n')
            f.write('=>'+ str(j)+'\n')
        f.write("\n Thank you! Goodbye.")

if __name__ == '__main__':
    main()


# In[ ]:




