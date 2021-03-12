import settings

def main() :

    # tell anyone using the settings module that it
    # is in the correct state
    # (all global variables have been updated)
    settings.STARTUP_CALLED = True
