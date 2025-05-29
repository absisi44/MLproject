from calendar import c
import sys

from matplotlib.backends.backend_gtk3 import err

def error_message_detail(error, error_detail:sys):
    """
    This function takes an error and its details, and returns a formatted error message.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message="Error occurred in python script name [{0}] line number [{1}] error message[{2}]".format(
     file_name,exc_tb.tb_lineno,str(error))
    
class CustomException(Exception):
    """
    Custom exception class that inherits from the built-in Exception class.
    It takes an error and its details, formats them, and raises a custom error message.
    """
    def __init__(self, error, error_detail:sys):
        super().__init__(error)
        self.error_message = error_message_detail(error, error_detail)

    def __str__(self):
        return self.error_message    