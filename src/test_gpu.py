import tensorflow as tf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List all available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
logger.info(f"Available GPUs: {gpus}")

if gpus:
    # Get GPU details
    for gpu in gpus:
        logger.info(f"GPU Device: {gpu}")
        try:
            memory_info = tf.config.experimental.get_memory_info(gpu)
            logger.info(f"Memory Info: {memory_info}")
        except:
            logger.info("Could not get memory info")
else:
    logger.warning("No GPU devices found!")

# Create a simple model to test GPU
x = tf.random.uniform((1000, 1000))
y = tf.random.uniform((1000, 1000))
z = tf.matmul(x, y)
logger.info("Matrix multiplication test completed") 