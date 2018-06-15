"""     
TODO: filter call to bot

TODO: filter dataframe to requested time interval
TODO: locate conversation intervals, filter to separate data frames
     TODO: "for conversation in conversation_dataframes:"
       TODO: extract entities, topics, sentiment <--- THE WHOLE PROJECT...
       TODO: compare discussion similarty to user_asker's RECENT history on the channel
       TODO: push extracted data into relevant template
"""
import re
import pandas as pd
import spacy

nlp = spacy.load('en_core_web_lg') # this is a long load, only happens once

interesting_entity_types = ['EVENT','PERSON','PRODUCT',
                            'ORG','TIME',# v. relevant to a summary
                            'FAC','GPE','LOC', # locations
                            'LANGUAGE','WORK_OF_ART' #others
                            ]
nonlocation_entities = ['EVENT','PERSON','PRODUCT','ORG','TIME','LANGUAGE','WORK_OF_ART']

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

def major_conversants(df):
    """
    Given a DataFrame with a 'user' column, locate the major talkers.
    Returns a list of the user tags.
    """
    # XXX TODO
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

def extract_sentiment(df, model):
    """
    Given a conversation DataFrame as input, return the sentiment
    based on a pretrained model.
    """
    return None

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

def create_location_string_from_entity_dict(dct):
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

def construct_payload(entities_dict, conversants):
    """
    Create the summary string:
        - create list of major conversants (flagging a dominant speaker)
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
    
    for k in nonlocation_entities:
        payload_dict['{0}_string'.format(k)] = create_firstTwoFromDict_string(entities_dict,k)
    payload_dict['LOC_string'] = create_location_string_from_entity_dict(entities_dict)
    
    # situation: no entities were extracted
    # XXX TODO: LOCATE SUBJECT NOUNS IN SPACY SPAN
    if all(v==None for v in payload_dict.values()):
        payload = conversant_string+'... something, but I cannot detect what.'
    else:
        payload = conversant_string+':'
        if payload_dict['EVENT_string']:
            payload+='\n - EVENT: {0}'.format(payload_dict['EVENT_string'])
        if payload_dict['LOC_string']:
            payload+='\n - LOCATION: {0}'.format(payload_dict['LOC_string'])
        if payload_dict['PERSON_string']:
            payload+='\n - {0}'.format(payload_dict['PERSON_string'])
        if payload_dict['ORG_string']:
            payload+='\n - {0}'.format(payload_dict['ORG_string'])
        if payload_dict['LANGUAGE_string']:
            payload+='\n - {0}'.format(payload_dict['LANGUAGE_string'])
        if payload_dict['WORK_OF_ART_string']:
            payload+='\n - {0}'.format(payload_dict['WORK_OF_ART_string'])
        if payload_dict['TIME_string']:
            payload+='\n - TIME: {0}'.format(payload_dict['TIME_string'])
    return payload