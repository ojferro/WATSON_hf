import sys                                                                                                                          
from io import StringIO 

class OutStreamCapture(object):
    """
    A context manager to replace stdout and stderr with StringIO objects and
    cache all output.
    """

    def __init__(self):
        self._stdout = None
        self._stderr = None
        self.stdout = None
        self.stderr = None

    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Restore original values of stderr and stdout.
        The captured contents are stored as strings in the stdout and stderr
        members.
        """
        self.stdout = sys.stdout.getvalue()
        self.stderr = sys.stderr.getvalue()
        sys.stdout = self._stdout
        sys.stderr = self._stderr