"""     
TODO: filter call to bot
TODO: filter emoticons and emojis
TODO: filter dataframe to requested time interval
TODO: locate conversation intervals, filter to separate data frames
    # TODO: "for conversation in conversation_dataframes:"
    #   TODO: get main conversationalists
    #   TODO: extract entities, topics, sentiment <--- THE WHOLE PROJECT...
    #   TODO: compare discussion similarty to user_asker's RECENT history on the channel
    #   TODO: push extracted data into relevant template
"""

def filter_history(slack_history,return_asker_info=False):
    """
    Filtration of the channel's historical data.
    Input:
        'history' item from the API call chat.history (list of dicts)
    Options:
        return_asker_info
    """
    df = pd.DataFrame(history['messages'])
    user_asker,time_asker = df.iloc[0,:][['user','ts']]
    df_nobots = df[df['subtype'].isna()]
