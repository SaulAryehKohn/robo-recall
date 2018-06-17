import os,re,time,ast
import pandas as pd
from datetime import datetime,timedelta
from slackclient import SlackClient
import summarybot_utils as sbut

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
INSIGHT_TESTING_TOKEN = os.environ["INSIGHT_TESTING_TOKEN"]
slack_client = SlackClient(SLACK_BOT_TOKEN)
starterbot_id = None

RTM_READ_DELAY = 1 # 1 second delay between reading from RTM
HELP_COMMAND = "help"
SUMMARIZE_COMMAND = "summarize"
MENTION_REGEX = "^<@(|[WU].+?)>(.*)"

# TODO: move these into utils or somewhere else
TOO_FEW_RESPONSE = "_Sorry..._ there are too few messages for me to summarize. Go read 'em!"

HELP_RESPONSE = """I'm SumBot! I summarize stuff that you missed.
        You can get my help by asking me to:\n
        *summarize* all- I'll summarize the past 1000 messages on the channel (that's the maximum I'm allowed to see!).\n
        *summarize* _n_ hours - I'll summarize the past _n_ hours.\n
        *summarize* _m_ minutes - I'll summarize the past _m_ minutes.\n"""

VALUE_ERROR_RESPONSE = """_Sorry_, I don't understand the time frame you're asking for. \n {0}""".format(HELP_RESPONSE)

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
    # the first group contains the username, the second group contains the remaining message
    return (matches.group(1), matches.group(2).strip()) if matches else (None, None)

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
                    count=1000
                    )
        # do an intial filtering on the history: get rid of emojis and user tags.
        df = sbut.filter_history(history)
        # get information on the poster
        user_asker= df.iloc[0,:]['user']
        
        # XXX TODO: move this to summarybot_utils
        if command.endswith('hour') or command.endswith('hours'):
            try:
                n_hour = float(command.split()[1]) 
                ts_oldest = time.time()-timedelta(hours = n_hour).total_seconds()
            except ValueError:
                response = VALUE_ERROR_RESPONSE
                break
        elif command.endswith('minute') or command.endswith('minutes'):
            try:
                n_mins = float(command.split()[1]) 
                ts_oldest = time.time()-timedelta(minutes = n_mins).total_seconds()
            except ValueError:
                response = VALUE_ERROR_RESPONSE
                break
        elif command_endswidth('all'):
            ts_oldest = 0
        else:
            response = VALUE_ERROR_RESPONSE
            break
        
        # time-filter
        df = df[df['ts'] > ts_oldest]
        
        # filter calls to bot XXX TODO: MAKE THIS MORE ELEGANT
        df = df[~df['text'].apply(lambda x: 'summarize' in x)]
        
        # XXX TODO: move to summarybot_utils
        # count reactions: "highlights" have a >{react_factor}sigma deviation from average
        react_factor = 4
        df['reaction_count'] = df['reactions'].fillna("[]").apply(
                                lambda x: len(ast.literal_eval(x))
                                )
        meanReact,stdReact = df['reaction_count'].mean(),df['reaction_count'].std()
        highlight_df = df[df['reaction_count']>meanReact+react_factor*stdReact]
        if len(highlight_df)>0:
            for i in range(highlight_df.shape[0]):
                user_h,text_h = highlight_df[['user','text']]
                highlights+='"raised_hands: *Highlight*: <@{0}>: {1}\n'.format(
                                                                 user_h,text_h)
            
        
        # should we continue?
        if len(df)<10:
            response = TOO_FEW_RESPONSE
            break
        
        # continuing -- NLP processing
        dialog_combined = df['text'][::-1].str.cat(sep=' ')
        entity_dict = sbut.extract_entities(dialog_combined)
        conversants = sbut.get_conversants(df)
        # create summary
        response = sbut.construct_payload(entity_dict, conversants)
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

if __name__ == "__main__":
    if slack_client.rtm_connect(with_team_state=False):
        print("Starter Bot connected and running!")
        # Read bot's user ID by calling Web API method `auth.test`
        starterbot_id = slack_client.api_call("auth.test")["user_id"]
        while True:
            command, channel = parse_bot_commands(slack_client.rtm_read())
            if command:
                handle_command(command, channel)
            time.sleep(RTM_READ_DELAY)
    else:
        print("Connection failed. Exception traceback printed above.")

