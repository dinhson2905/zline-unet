import logging

def setup_logger(name, log_file, level=logging.DEBUG):
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger