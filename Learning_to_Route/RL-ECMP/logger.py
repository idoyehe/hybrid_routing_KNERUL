import logging
level = logging.DEBUG
logging.basicConfig(format='%(levelname)s:%(message)s', level=level)
logger = logging.getLogger(__name__)
logger.setLevel(level)

