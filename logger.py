# importing module
import logging
 
# Create and configure logger
logging.basicConfig(filename="newfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='a')
 
# Creating an object

logger = logging.getLogger()
 
# Setting the threshold of logger to DEBUG
logger.setLevel(logging.INFO)
 
# Test messages
# logger.debug("loss 0",5)
logger.info("Just an information %.4f %.4f",5,6)
