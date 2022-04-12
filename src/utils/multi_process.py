from itertools import islice
import threading


class Worker(threading.Thread):
    def __init__(self, target, name=None, args=(), kwargs={}):
        threading.Thread.__init__(self, None, target, name, args, kwargs)
        self._return = None

    def run(self):
        self._return = self._target(*self._args, **self._kwargs)

    def get(self, timeout=None):
        self.join(timeout)
        return self._return


def chunk_list(input_list: list, chunk_size: int):
    """
    Chunk input list into the equal size
    :param input_list: Input list to be divided
    :param chunk_size: chunk size
    :return: Divided lists
    """
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]
