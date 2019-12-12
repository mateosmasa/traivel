# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""



#Extracción de los puntos de interes en Madrid 
#lectura del XML


import xml.etree.ElementTree as etree
import html
import pandas as pd 
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import unidecode
import numpy as np
import re
from fuzzywuzzy import process
import pickle
from collections import Counter
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from language_detector import detect_language
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import  LatentDirichletAllocation
from sklearn.metrics import jaccard_score



TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

# =============================================================================
# Creación dataframe de los POIS
# =============================================================================
#datos ayuntamiento madrid
    
def get_pois_madrid():
    with open("/home/cx02603/Descargas/turismo_v1_en.xml", 'r') as xml_file:
        xml_tree = etree.parse(xml_file)
    
    xroot = xml_tree.getroot()
    pois = pd.DataFrame(columns= ["name","description","lat","lon","category"])

    rows = []
    for elem in xroot.iter('service'):
        name = html.unescape(elem.find('basicData/name').text) if elem.find('basicData/name').text is not None else None
        des = remove_tags(elem.find("basicData/body").text) if elem.find("basicData/body").text is not None else None
        lat = elem.find("geoData/latitude").text if  elem.find("geoData/latitude").text is not None else None
        lon = elem.find("geoData/longitude").text if elem.find("geoData/longitude").text  is not None else None
        categoria = elem.find('extradata/categorias/categoria/item/[@name="Categoria"]').text \
                if elem.find('extradata/categorias/categoria/item/[@name="Categoria"]') is not None else None
        rows.append({"name": name, "description": des,'lat':lat,'lon':lon,'category':categoria})
        
    return pois.append(rows)

# =============================================================================
#  Extracción datos TripAdvisor
# =============================================================================
def create_trip_urls():
    return ["https://www.tripadvisor.es/Attractions-g187514-Activities-c47-oa{rango}-Madrid.html#FILTERED_LIST".format(rango=rango) 
            for rango in list(range(0,570,30))]
    

def get_trip_pois(url):
    #contenido de html
    html_content = requests.get(url).text
    soup = BeautifulSoup(html_content, "lxml")
    pois = pd.DataFrame(columns= ["url","name"])
    rows = []
    prefix_url = "https://www.tripadvisor.es" 
    for link in soup.find_all("div", class_="listing_title title_with_snippets"):
        rows.append({"url": prefix_url+link.find("a").attrs["href"], "name":  re.sub('\n','', link.get_text())})  
    return pois.append(rows)


def get_all_pois():
    urls = create_trip_urls()
    pois = pd.DataFrame()
    for url in tqdm(urls):
        pois = pois.append(get_trip_pois(url))
    return pois 


def request_html_trip(df):
    lista_req = []
    s = requests.Session()
    s.verify = True
    for i,row in tqdm(df.iterrows()):
        lista_req.append(s.get(row['url']).text)
    with open('list_req', 'wb') as fp:
        pickle.dump(lista_req, fp)
    return lista_req

def get_reviews(lista_req):
    df = pd.DataFrame()
    for req in tqdm(lista_req):
        soup = BeautifulSoup(req, "lxml") 
        try:
            reviews = [item for item in soup.find_all('div', class_="choices") if item.attrs["data-name"] == 
                       "ta_rating"][0].get_text()
            reviews = unidecode.unidecode(reviews.replace(".","").replace(" ", ""))
            reviews = re.findall(r'[A-Za-z]+|\d+', reviews)
            values = reviews[1::2]
            address = [item for item in soup.find_all('div', class_="detail_section address")][0].get_text()
            values = [address] + values
        except:
            print("no hay reviews")
            values = [0] * 5   
            address = [item for item in soup.find_all('div', class_="detail_section address")][0].get_text()
            values = [address] + values
        df = pd.concat([df,pd.DataFrame(values).T],axis=0)
    df.columns = ['address','Excelente', 'Muybueno', 'Normal', 'Malo', 'Pesimo']
    return df


def fuzzy_merge(df_1, df_2, key1, key2, threshold=90, limit=2):
    """
    df_1 is the left table to join
    df_2 is the right table to join
    key1 is the key column of the left table
    key2 is the key column of the right table
    threshold is how close the matches should be to return a match, based on Levenshtein distance
    limit is the amount of matches that will get returned, these are sorted high to low
    """
    s = df_2[key2].tolist()

    m = df_1[key1].apply(lambda x: process.extract(x, s, limit=limit))    
    df_1['matches'] = m

    m2 = df_1['matches'].apply(lambda x: ', '.join([i[0] for i in x if i[1] >= threshold]))
    df_1['matches'] = m2

    return df_1

def get_corpus(df):
    corpus = []
    for doc in df['tokens']:
        corpus.append(doc)
    return corpus


def split_to_words(corpus):
    seg_list = []
    for doc in corpus:
        string = " ".join(doc)
        seg_list.append(string)
    return seg_list


def display_topics(model, feature_names, no_top_words):
    '''Display topics generated from NMF and LDA mdoel'''
    lista_topics = {}
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        lista_topics[topic_idx] =[feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
    return lista_topics

def get_topics(lda_results,df):
    a = lda_results.argmax(1)
    df['topics'] = a
    b = Counter(a)
    print (b.most_common())
    return df

def jaccard_similarity(list1, list2):
    return jaccard_score(list1, list2 ,average='binary')

def best_results(user_selection,df,n_pois,relevant):
    kewyords = [col for col in df.columns if 'keyword' in col]
    df_keyword = df[kewyords]
    sim_ = pd.DataFrame(columns= ["index","sim"])
    rows = []
    for i,row in tqdm(df_keyword.iterrows()) :
        sim = jaccard_similarity(user_selection, row.values.flatten().tolist())
        rows.append({"index": i, "sim": sim})
    sim_ = sim_.append(rows).sort_values('sim',ascending=False) 
    if relevant == "Relevants":
        df = df.loc[sim_['index'].values.flatten().tolist(),['Tripadvisor','rate','Excelente']].drop_duplicates().reset_index(drop=False)
        df = df.merge(sim_,on='index',how='left')
        df['Excelente'] = df['Excelente']/df['Excelente'].sum()
        df['rate_sim'] = (df['rate']+df['sim']) * (df['Excelente'])
        #df['rate_sim'] = (df['rate']*df['sim']) + (df['Excelente'])
        print(df.sort_values('rate_sim',ascending=False).head(n_pois))
        return df.sort_values('rate_sim',ascending=False).head(n_pois)
    else:
        df = df.loc[sim_['index'].values.flatten().tolist(),['Tripadvisor','rate','Excelente']].drop_duplicates().reset_index(drop=False).head(n_pois)
        df = df.merge(sim_,on='index',how='left')
        df['Excelente'] = df['Excelente']/df['Excelente'].sum()
        df['rate_sim'] = (df['rate']*df['sim']) + (df['Excelente'])
        print(df.sort_values('rate_sim',ascending=False))
        return df.sort_values('rate_sim',ascending=False)


def review_rate(df):
    df["rate"] = df.apply(lambda x : (x.Excelente*5 + x.Muybueno*4 +  x.Normal*3 + x.Malo*2 + x.Pesimo*1) / 
      (x.Excelente + x.Muybueno + x.Normal + x.Malo + x.Pesimo) if (x.Excelente + x.Muybueno + x.Normal + x.Malo + x.Pesimo) != 0 else 0 ,axis=1)
    return df


def dict_user(df,user_selection):
    dict_user = {col:0 for col in df.columns if 'keyword' in col}
    for elem in user_selection:
        dict_user["keyword_{}".format(elem)] = 1
    return list(dict_user.values())

def user_selection(selection):
    if selection == 'Relevants':
        user_selection = ['architect','century','city','famous','royal']
    elif selection == 'Churches':
        user_selection = ['cathedral','catholic','convent']
    else: 
        user_selection = ['stock','trade','government','court','known']
    return user_selection
    
### Pois del ayuntamiento
#pois = get_pois_madrid()
###Listar POIS de tripAdvisor
#trip_pois = get_all_pois()
#
##Extraer las HTML de todos los POIS
#lista_req = request_html_trip(trip_pois)
#
##Extraer las puntuaciones y la direccion de los POIS
#df_reviews = get_reviews(lista_req)
#
##Match entre ayuntamiento y TripAdvisor
#df_reviews = fuzzy_merge(trip_pois,pois,'name','name',threshold=90)
#
#df_reviews = df_reviews.replace(r'^\s*$', np.nan, regex=True)
#
##Merge entre nombres POIS y sus métricas
#df_reviews_name = pd.concat([df_reviews.reset_index(),trip_pois.reset_index()],axis=1)
    

# =============================================================================
# APPLY NLP PREPROCESSING AND LDA MODEL TO OBTAIN TOPICS
# =============================================================================
    


def main(file,userselection,n_pois,output,desc):

    df_reviews = pd.read_csv(file).drop_duplicates(subset=['Tripadvisor'],keep='first')
        #Lower all words
    df_reviews['description']= df_reviews['description'].str.lower()
    #Numeric to strings
    df_reviews['description']= df_reviews['description'].apply(lambda x: re.sub(r'\d+', '', x))
    #remove html tags
    df_reviews['description']= df_reviews['description'].apply(lambda x: (html.unescape(x)))
    #remove punctuation
    df_reviews['description']= df_reviews['description'].apply(lambda x:  x.translate(str.maketrans('', '', 
              string.punctuation)))
    #remove accent
    df_reviews['description']= df_reviews['description'].apply(lambda x:  unidecode.unidecode(x))
    #remove specific characters and words
    df_reviews['description']= df_reviews['description'].apply(lambda x:  re.sub("description", '', x))
    df_reviews['description']= df_reviews['description'].apply(lambda x:  re.sub("wikipedia", '', x))
    df_reviews['description']= df_reviews['description'].apply(lambda x:  re.sub("'s", '', x))
    #stop words 
    stop_words= set(stopwords.words('english'))
    lemmatizer=WordNetLemmatizer()
    df_reviews['tokens'] = df_reviews['description'].apply(lambda x: [lemmatizer.lemmatize(word) for word in 
              word_tokenize(x) if not word in stop_words  and detect_language(word) =='English' ])
    
    #get corpus
    corpus = get_corpus(df_reviews)
    
    seg_list = split_to_words(corpus)
    vectorizer_model = CountVectorizer(stop_words=stop_words,
                               analyzer='word',
                               max_features=2000)
    vec_docs = vectorizer_model.fit_transform(seg_list)
    tf_feature_names = vectorizer_model.get_feature_names()
    
    no_topics = 10
    no_top_words = 5
    
    lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=40.,random_state=0).fit(vec_docs)
    display_topics(lda, tf_feature_names, no_top_words)
    lda_results = lda.fit_transform(vec_docs)
    
    df_reviews = get_topics(lda_results,df_reviews)
    
    topic_dict = display_topics(lda, tf_feature_names, no_top_words)
    
    h = pd.DataFrame.from_dict(topic_dict,orient='index').transpose().melt()
    
    df_reviews = df_reviews.merge(h,left_on='topics',right_on='variable',how='left')
    df_reviews = df_reviews.drop(columns=['topics','variable','tokens'])
    df_reviews =pd.get_dummies(df_reviews, prefix=['keyword'], columns=['value']).drop_duplicates()
    cols = [col for col in df_reviews.columns if 'keyword' not in col]
    
    df_reviews = df_reviews.groupby(cols).sum().reset_index()
    
    df_reviews = review_rate(df_reviews)
    
    selection = user_selection(userselection)
    
    results = best_results(dict_user(df_reviews,selection),df_reviews,n_pois,relevant=userselection)
    
    if output != "default":
       results.to_csv("{output}_{sufix}.csv".format(output=output,sufix=selection),index=False)
    desc_ = pd.read_csv(desc,sep="|")      
    results = results.merge(desc_,left_on='Tripadvisor',right_on='name',how='inner')
    return results


if __name__ == '__main__':
     main(file,userselection,n_pois,output,desc)





