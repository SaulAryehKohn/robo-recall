import os,re,time
import pandas as pd
from flask import abort, Flask, jsonify, request
from slackclient import SlackClient
import summarybot_utils as sbut
import summarybot_ref as sref
#app = Flask(__name__)

#####################################################################
# constants
#####################################################################
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
INSIGHT_TESTING_TOKEN = os.environ["INSIGHT_TESTING_TOKEN"]
slack_client = SlackClient(SLACK_BOT_TOKEN)
starterbot_id = None
RTM_READ_DELAY = 1 # 1 second delay between reading from RTM
HELP_COMMAND = "help"
SUMMARIZE_COMMAND = "summarize"
MENTION_REGEX = "^<@(|[WU].+?)>(.*)"
TOO_FEW_RESPONSE = sref.too_few_response
HELP_RESPONSE = sref.help_response
VALUE_ERROR_RESPONSE = sref.value_error_response

#####################################################################
# helper methods
#####################################################################

def parse_bot_commands(slack_events):
    """
        Parses a list of events coming from the Slack RTM API to find bot commands.
        If a bot command is found, this function returns a tuple of command and channel.
        If its not found, then this function returns None, None.
    """
    for event in slack_events:
        if event["type"] == "message" and not "subtype" in event:
            user_id, message = parse_direct_mention(event["text"])
            if user_id == starterbot_id:
                return message, event["channel"]
    return None, None

def parse_direct_mention(message_text):
    """
        Finds a direct mention (a mention that is at the beginning) in message text
        and returns the user ID which was mentioned. 
        If there is no direct mention, returns None
    """
    matches = re.search(MENTION_REGEX, message_text)
    # the first group contains the username, 
    # the second group contains the remaining message
    return (matches.group(1), matches.group(2).strip()) if matches else (None, None)

#####################################################################
# bot internals
#####################################################################

def handle_command(command, channel):
    """
        Executes bot command if the command is known
    """
    # Default response is help text for the user
    default_response = "Hi! Not sure what you mean. Try *{0}*.\n {1}".format(SUMMARIZE_COMMAND,HELP_RESPONSE)
    
    # Finds and executes the given command, filling in response
    response = None
    highlights = None
    
    while command.startswith(SUMMARIZE_COMMAND):
        history = slack_client.api_call("channels.history", 
                    token=INSIGHT_TESTING_TOKEN,
                    channel=channel,
                    count=1000 #max
                    )
        # do an intial filtering on the history: get rid of emojis and user tags.
        df = sbut.filter_history(history)
        # get information on the poster
        user_asker= df.iloc[0,:]['user']
        
        # get the requested time span to filter by
        try:
            ts_oldest = sbut.parse_time_command(command)
        except ValueError:
            response = VALUE_ERROR_RESPONSE
            break
        
        # time-filter
        df = df[df['ts'].astype(float) > ts_oldest]
        
        # filter calls to bot XXX TODO: MAKE THIS MORE ELEGANT
        df = df[~df['text'].apply(lambda x: 'summarize' in x)]
        
        # should we continue?
        if len(df)<10:
            response = TOO_FEW_RESPONSE
            break
        
        # continuing -- NLP processing
        dialog_combined = df['text'][::-1].str.cat(sep=' ')
        
        #TODO: sentiment detect on CLEANED DataFrame
        # (we want VADER score per utterance)
        
        # Outlier detection: emoji count and words
        highlights = sbut.extract_highlights(df)
        outliers = sbut.outlier_word_detection(df)
        
        entity_dict = sbut.extract_entities(dialog_combined)
        lemmad_nouns = sbut.extract_lemmatized_tokenized_nouns(df)
        topic_names,topic_list = sbut.extract_topics(lemmad_nouns,n_terms=3)
        topic_list += outliers
        conversants = sbut.get_conversants(df)
        # create summary
        response = sbut.construct_payload(entity_dict, conversants, topic_list,
                                          highlights=highlights,topic_names=topic_names)
        break
        
    # In any case, we now have a response
    # Sends the response back to the channel
    slack_client.api_call(
        "chat.postEphemeral", #visible only to user
        channel=channel,
        user=user_asker,
        as_user=False, # sets subtype to "bot message" for easy cleaning
        text=response or default_response
        )

#####################################################################
# launch!
#####################################################################

if __name__ == "__main__":
    if slack_client.rtm_connect(with_team_state=False):
        print("Bot connected and running!")
        # Read bot's user ID by calling Web API method `auth.test`
        starterbot_id = slack_client.api_call("auth.test")["user_id"]
        while True:
            command, channel = parse_bot_commands(slack_client.rtm_read())
            if command:
                handle_command(command, channel)
            time.sleep(RTM_READ_DELAY)
    else:
        print("Connection failed. Exception traceback printed above.")

