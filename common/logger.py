import logging

level = logging.INFO
logging.basicConfig(format='%(levelname)s:%(message)s', level=level)
logger = logging.getLogger(__name__)
logger.setLevel(level)
logger.disabled = 1
