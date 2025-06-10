import os
from contextlib import contextmanager

@contextmanager
def cd(path):
    old_cwd = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old_cwd)
