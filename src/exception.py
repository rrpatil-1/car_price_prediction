import sys
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def error_message_details(error,error_details:sys):
    _,_,exe_tb=error_details.exc_info()
    filename=exe_tb.tb_frame.f_code.co_filename
    error_message="error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        filename,exe_tb.tb_lineno,str(error)
    )
    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_details:sys):
        super().__init__(error_message)
        self.error_message=error_message_details(error_message,error_details=error_details)
        
    def __str__(self):
        return self.error_message