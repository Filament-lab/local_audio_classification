import os
import logging

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    datefmt="%Y-%m-%dT%H:%M",
    format="[%(asctime)s.%(msecs)03dZ]: %(levelname)s: [%(module)s - L%(lineno)d] %(message)s",
)
logger = logging.getLogger("audio-classification")
