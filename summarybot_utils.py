import re, itertools, time
from datetime import datetime,timedelta
import numpy as np
import pandas as pd
from collections import Counter
import spacy
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = stopwords.words('english')

# these are long loads, but they only happen once. Requires ~ 9 GB of RAM
print('loading NLP')
nlp = spacy.load('en_core_web_lg') 
print('loading Dictionary')
dictionary = corpora.Dictionary.load_from_text('models/_wordids.txt.bz2')
print('loading Corpus')
# this is a 7.5 GB file that is difficult to share online
corpus = corpora.MmCorpus('models/_tfidf.mm.bz2') 
print('loading LDA model')
lda = models.ldamodel.LdaModel.load('models/enwiki_ldamodel.model')
lda_topics_named = pd.read_csv('models/topic_list.csv', index_col=0, header=None, 
                                names=['index','topic'])
# This is very slow
#print('loading word vectors')
#word_vectors=models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)
# trick get the word_vector object cached for fast querying later
#_ = word_vectors.most_similar(positive=["apple"])

# entities
interesting_entity_types = ['EVENT','PERSON','PRODUCT',
                            'ORG','TIME',# v. relevant to a summary
                            'FAC','GPE','LOC', # locations
                            'LANGUAGE','WORK_OF_ART'] #others               
nonlocation_entities = ['EVENT','PERSON','PRODUCT','ORG','TIME','LANGUAGE','WORK_OF_ART']

def parse_time_command(command):
    """
    Get oldest response to filter the conversation down to.
    """
    if command.endswith('hour') or command.endswith('hours'):
        n_hour = float(command.split()[1]) 
        ts_oldest = time.time()-timedelta(hours = n_hour).total_seconds()
    elif command.endswith('minute') or command.endswith('minutes'):
        n_mins = float(command.split()[1]) 
        ts_oldest = time.time()-timedelta(minutes = n_mins).total_seconds()
    elif command.endswith('all'):
        ts_oldest = 0.
    else:
        raise ValueError
    return ts_oldest

def filter_history(slack_history):
    """
    Filtration of the channel's historical data.
    Input:
        'history' item from the API call chat.history (list of dicts)
    Returns:
        pandas DataFrame
    Options:
        return_asker_info: if True, return info on user call to bot.
    """
    df = pd.DataFrame(slack_history['messages'])
    df = df[df['subtype'].isna()] # filter bot-type messages
    emoji_pattern = re.compile(':.*?:') #slack emojis are bounded by colons
    df['text'] = df['text'].apply(lambda x: emoji_pattern.sub('',x)) # remove slack emojis
    tag_pattern = re.compile('<@.*?>')
    df['text'] = df['text'].apply(lambda x: tag_pattern.sub('',x)) # remove user tags
    return df

def get_conversants(df):
    """
    Given a DataFrame with a 'user' column, locate the major talkers.
    Returns a list of the user tags.
    """
    user_set = set(df['user'])
    user_tags = ['<@{}>'.format(id) for id in user_set]
    return user_tags
    
def extract_entities(conv_string):
    """
    Given a string as input, perform Named Entity Recognition.
    Extract interesting entities to report on in a summary.
    
    NOTE: Capitalization is important! 
          For example, "Amazon" will be recognized as an organization,
          but "amazon" will not be.
    """
    conv_nlp = nlp(conv_string)
    dct = {}
    # I feel like there's a way to get rid of/vectorize this for loop
    for ent in conv_nlp.ents:
        if ent.label_ in interesting_entity_types:
            try:
                dct[ent.label_].append(ent.text)
            except KeyError:
                dct[ent.label_] = [ent.text]
    dct_ = {k:set(dct[k]) for k in dct.keys()}
    return dct_

def extract_highlights(df,react_factor=4):
    """
    Perform outlier detection on emoji-count per-comment.
    Counts deviant by more than react_factor*sigma are classified
    as highlights of the discussion.
    """
    if 'reactions' in df.columns:
        df['reaction_count'] = df['reactions'].fillna("").apply(
                                lambda x: len(x))
        meanReact = df['reaction_count'].mean()
        stdReact = df['reaction_count'].std()
        highlight_df = df[df['reaction_count']>meanReact+react_factor*stdReact]
        if len(highlight_df)>0:
            highlights=''
            for i in range(highlight_df.shape[0]):
                user_h,text_h = highlight_df[['user','text']]
                highlights+='"raised_hands: *Highlight*: <@{0}>: {1}\n'.format(
                                                                 user_h,text_h)
            return highlights
        else:
            return None
    else:
        return None

def extract_sentiment(df):
    """
    Given a conversation DataFrame as input, return the sentiment
    based on a pretrained model.
    """
    return None

def extract_lemmatized_tokenized_nouns(df,min_length=4):
    """
    Given a conversation DataFrame as input, return a list of lemmatized
    nouns and proper nouns as a list of lists of tokens (stop words and
    strings of less than length min_length removed).
    """
    dlg = df['text'][::-1].tolist() # reverse order to maintain chronology.
    dlg_out = []
    for sentence in dlg:
        nlp_sentarr = nlp(' '.join(
        [w.lower() for w in sentence.split() if not w.lower() in stop_words and not len(w)<min_length]
        ))
        nouns_sentarr = [tk.lemma_ for tk in nlp_sentarr if tk.pos_ in ['NOUN','PROPN']]
        if len(nouns_sentarr)>0:
            dlg_out.append(nouns_sentarr)
    return dlg_out

def outlier_word_detection(df,sigma=1):
    """
    Given a DataFrame with a list of utterances, find terms that 
    appear erroneously often in the conversation, and return them.
    """
    dialog_list = df['text'][::-1].tolist()
    list_of_terms = list(itertools.chain.from_iterable(dialog_list))
    counts = Counter(list_of_terms)
    m = np.mean([counts[k] for k in counts.keys()])
    s = np.std([counts[k] for k in counts.keys()])
    outliers = [w for w in counts.keys() if counts[w]>m+s and w not in stop_words and len(w)>1]
    return outliers

def extract_topics(dialog_list, dictionary=dictionary, topic_model=lda, n_terms=3, return_broad_themes=True):
    """
    For a list of strings, create a Bag of Words from a gensim dictionary object. 
    Use a specified LDA topic_model to generate topics from the BoW, 
    and from this model, select the N highest-weighted terms.
    """
    dialog_list = list(itertools.chain.from_iterable(dialog_list)) # aggregate
    dlg_bow = dictionary.doc2bow(dialog_list)
    # get highest-weight topics
    topics = sorted(topic_model.get_document_topics(dlg_bow), key=lambda x: x[1])[::-1]
    topic_names = [lda_topics_named['topic'][i] for i,_ in topics]
    term_tuple_list = list(itertools.chain.from_iterable(
                               [topic_model.get_topic_terms(t[0]) for t in topics]
                               )
                            )
    top_n_terms = sorted(term_tuple_list, key=lambda x: x[1])[::-1][:n_terms]
    decode_top_n_terms = [dictionary[tp[0]] for tp in top_n_terms]
    if not return_broad_themes:
        return decode_top_n_terms
    else:
        return (topic_names,decode_top_n_terms)

def sum_topic_vectors(topic_list):
    """
    Given a list of strings, attempt to translate them into GoogleNews
    word vectors, aggregate them, and calculate the most similar vector
    in the vocabulary.
    """
    topic_list = [t for t in topic_list if t in word_vectors.vocab]
    if len(topic_list) < 2:
        return None
    synth = word_vectors.most_similar(positive=topic_list)[0][0]
    return synth

def create_firstTwoFromDict_string(dct,key):
    """
    Extract first two entries (if there are more than one)
    from dictionary value dct[key] and print prettily.
    """
    kstring = None
    if key in dct:
        if len(dct[key]) == 1:
            kstring = list(dct[key])[0]
        else:
            # take first two
            # XXX this could be improved
            kstring = '{0} and {1}'.format(*list(dct[key])[:2])
    return kstring

def create_locationFromEntityDict_string(dct):
    """
    Generate a string for locations, according to relevance
    priorities: GPE > FAC > LOC (country/city/state > named places > lakes etc.)
    """
    locstring = None
    if 'GPE' in dct:
        locstring = create_firstTwoFromDict_string(dct,'GPE')
    elif 'FAC' in dct:
        locstring = create_firstTwoFromDict_string(dct,'FAC')
    elif 'LOC' in dct:
        locstring = create_firstTwoFromDict_string(dct,'LOC')
    return locstring

def construct_payload(entities_dict, conversants, topic_list,
                      topic_names=None, highlights=None, enable_w2v=False):
    """
    Create the summary string:
        - create list of conversants
        - extract the most pertinent entities
        - frame summary in terms of mood
    """
    payload_dict = {}
    
    # conversants 
    # TODO: consider an "et al." condition for busy chats
    if len(conversants) == 1:
        conversant_string = '{0} wrote about'.format(conversants[0])
    elif len(conversants) == 2:
        conversant_string = '{0} and {1} wrote about'.format(*conversants)
    else:
        conversant_string = '{0} and {1} wrote about'.format(
            ', '.join(conversants[:len(conversants)-1]),
            conversants[-1]
            )
    
    # topics
    if len(topic_list)>0:
        if topic_names:
            payload = '{0}, broadly: {1}'.format(conversant_string,topic_names[0])
            payload += '\n in terms of: {0}\n'.format(topic_list)
        else:
            payload = '{0} topics: {1}\n'.format(conversant_string, topic_list)
        if enable_w2v:
            synthesized_topics = sum_topic_vectors(topic_list)
            if synthesized_topics:
                payload+= '(I think these topics sum to {0})\n'.format(synthesized_topics)
    else:
        payload = conversant_string
    
    # highlights
    if highlights:
        payload += highlights
    
    # get entities and log in dict
    for k in nonlocation_entities:
        payload_dict['{0}_string'.format(k)] = create_firstTwoFromDict_string(
                                                                entities_dict, k)
    
    # get location by priority
    payload_dict['LOC_string'] = create_locationFromEntityDict_string(entities_dict)
    
    # situation: no entities were extracted
    if all(v==None for v in payload_dict.values()):
        payload+="\n"
    else:
        payload+="The following things came up:"
        if payload_dict['EVENT_string']:
            payload+='\n - EVENT: {0}'.format(payload_dict['EVENT_string'])
        if payload_dict['LOC_string']:
            payload+='\n - LOCATION: {0}'.format(payload_dict['LOC_string'])
        if payload_dict['TIME_string']:
            payload+='\n - TIME: {0}'.format(payload_dict['TIME_string'])
        if payload_dict['PERSON_string']:
            payload+='\n - {0}'.format(payload_dict['PERSON_string'])
        if payload_dict['ORG_string']:
            payload+='\n - {0}'.format(payload_dict['ORG_string'])
        if payload_dict['LANGUAGE_string']:
            payload+='\n - {0}'.format(payload_dict['LANGUAGE_string'])
        if payload_dict['WORK_OF_ART_string']:
            payload+='\n - {0}'.format(payload_dict['WORK_OF_ART_string'])
    return payload