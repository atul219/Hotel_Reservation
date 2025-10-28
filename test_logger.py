from src.logger import get_logger
from src.custom_exception import CustomException
import sys

logger = get_logger(__name__)

def test(a, b):
    try:
        result = a/b
    except Exception as e:
        logger.error("divide by 0")
        raise CustomException("Divede by 0", sys)


if __name__ == "__main__":
    test(10, 0)