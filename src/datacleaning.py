from datetime import datetime as dt
def datetime_conversion(df):
    '''
    The created_at, launched_at, deadline, and state_changed_at columns are
    in unix timestamps, to work with them they are converted into datetime objects.
    '''
    df.created_at = df.created_at.apply(lambda x: dt.fromtimestamp(x))
    df.launched_at = df.launched_at.apply(lambda x: dt.fromtimestamp(x))
    df.deadline = df.deadline.apply(lambda x: dt.fromtimestamp(x))
    df.state_changed_at = df.state_changed_at.apply(lambda x: dt.fromtimestamp(x))
    return

def checkit(string):
    '''
    This function checks to see how long the ID number is in the string
    '''
    i = 0
    for letter in string:
        if letter.isdigit():
            i += 1
        else:
            break
    return i