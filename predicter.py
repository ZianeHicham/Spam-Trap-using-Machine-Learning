#Importing libraries
import pandas as pd 
from urllib.parse import urlparse
from tld import get_tld
import os.path
import re
import os




#First Directory Length
def fd_length(url):
    urlpath= urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0

#Length of Top Level Domain
def tld_length(tld):
    try:
        return len(tld)
    except:
        return -1
#number of digits in the url
def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits


#number of letters in the url
def letter_count(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
    return letters
#number of directory in the url
def no_of_dir(url):
    urldir = urlparse(url).path
    return urldir.count('/')

#Use of IP or not in domain
def having_ip_address(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
    if match:
        # print match.group()
        return -1
    else:
        # print 'No matching pattern found'
        return 1

#verify if the url has been shortened
def shortening_service(url):
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      'tr\.im|link\.zip\.net',
                      url)
    if match:
        return -1
    else:
        return 1



#this is the most important function that we use in the notebook named 'DetectSpams' to apply 
#all the features extraction process that we have seen on new urls
def make_prediction(url,model):

    df = pd.DataFrame({"url":url})
    df['url_length'] = df['url'].apply(lambda i: len(str(i))) #url length
    df['hostname_length'] = df['url'].apply(lambda i: len(urlparse(i).netloc)) # the length of hostname
    df['path_length'] = df['url'].apply(lambda i: len(urlparse(i).path)) # the length of path
    df['fd_length'] = df['url'].apply(lambda i: fd_length(i)) #first directory length
    df['tld'] = df['url'].apply(lambda i: get_tld(i,fail_silently=True))
    df['tld_length'] = df['tld'].apply(lambda i: tld_length(i))
    df = df.drop("tld",1)
    df['count-'] = df['url'].apply(lambda i: i.count('-')) #count of - in the url
    df['count@'] = df['url'].apply(lambda i: i.count('@')) #count of @ in the url
    df['count?'] = df['url'].apply(lambda i: i.count('?')) #count of ? in the url
    df['count%'] = df['url'].apply(lambda i: i.count('%')) #count of % in the url
    df['count.'] = df['url'].apply(lambda i: i.count('.')) #count of . in the url
    df['count='] = df['url'].apply(lambda i: i.count('=')) #count of = in the url
    df['count-http'] = df['url'].apply(lambda i : i.count('http')) #count of http in the url
    df['count-https'] = df['url'].apply(lambda i : i.count('https')) #count of https in the url
    df['count-www'] = df['url'].apply(lambda i: i.count('www')) #count of www in the url
    df['count-digits']= df['url'].apply(lambda i: digit_count(i)) #count of digits in the url
    df['count-letters']= df['url'].apply(lambda i: letter_count(i)) #count of letters in the url
    df['count_dir'] = df['url'].apply(lambda i: no_of_dir(i)) #count of directory in the url
    df['use_of_ip'] = df['url'].apply(lambda i: having_ip_address(i)) #ip is used in the domain or not
    df['short_url'] = df['url'].apply(lambda i: shortening_service(i))#a shortening service is used or not
    x = df[['hostname_length',
       'path_length', 'fd_length', 'tld_length', 'count-', 'count@', 'count?',
       'count%', 'count.', 'count=', 'count-http','count-https', 'count-www', 'count-digits',
       'count-letters', 'count_dir', 'use_of_ip']] #Select the features to feed into the classification model
    #for i in range(0,df.shape[0]):
    prediction = model.predict(x) #make predictions
    

    
    return prediction







