import time


class Timer:

    def __init__(self, name: str = None):
        self.start_time = 0
        self.end_time = 0

        self.name = name or "Timer"

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        execution_time = self.end_time - self.start_time
        print(f"`{self.name}` execution time: {execution_time:.3f} seconds")


def measure_time(func, name: str = None):
    def wrapper(*args, **kwargs):
        with Timer(name):
            return func(*args, **kwargs)

    return wrapper
