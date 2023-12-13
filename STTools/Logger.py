import datetime
import time
import logging

'''
A simple logger
'''
class STLogger():
    @staticmethod
    def datetime():
        return f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"

    @staticmethod
    def error(s):
        print("\033[0;31m", STLogger.datetime(), s, "\033[0m")

    @staticmethod
    def warning(s):
        print("\033[1;33m", STLogger.datetime(), s, "\033[0m")

    @staticmethod
    def success(s):
        print("\033[0;32m", STLogger.datetime(), s, "\033[0m")

    @staticmethod
    def info(s):
        print("\033[0;36m", STLogger.datetime(), s, "\033[0m")


def runtimeTest(f):
    def wrapper():
        startTime = time.time()
        STLogger.info(f"[{f.__name__}] start running.")
        ret = f()
        endTime = time.time()
        STLogger.info(f"[{f.__name__}] running time is: {(endTime - startTime)*1000} ms")
        return ret
    return wrapper


def dateLog(f):
    def wrapper():
        STLogger.info(f"[{f.__name__}] start running")
        return f()
    return wrapper

