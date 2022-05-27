import logging, sys

def _logger(debug=False, logfile='Data/log.txt', verbose=False):
    logFormatter = logging.Formatter(
        '%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] %(message)s'
    )
    rootLogger = logging.getLogger()
    # set debugging level
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    rootLogger.setLevel(level)

    # file logging to logfile
    fileHandler = logging.FileHandler(logfile)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    # logging to std out
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    # if verbose is set we print level else we print warnings only
    if verbose:
        consoleHandler.setLevel(level)
    else:
        consoleHandler.setLevel(logging.WARNING)
    rootLogger.addHandler(consoleHandler)
    return rootLogger