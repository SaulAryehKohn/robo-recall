too_few_response = "_Sorry..._ there are just too few messages for me to summarize. Go read 'em!"

help_response = """I'm SumBot! I summarize stuff that you missed.
        You can get my help by asking me to:\n
        *summarize* all- I'll summarize the past 1000 messages on the channel (that's the maximum I'm allowed to see!).\n
        *summarize* _n_ hours - I'll summarize the past _n_ hours.\n
        *summarize* _m_ minutes - I'll summarize the past _m_ minutes.\n"""

value_error_response = """_Sorry_, I don't understand the time frame you're asking for. \n {0}""".format(help_response)