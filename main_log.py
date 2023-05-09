from datetime import datetime
from multiprocessing import Process, Queue
import logging
import logging.handlers

datetime_str_format = "%m-%d-%Y_%H%M%S"
now_string = datetime.now().strftime(datetime_str_format)
root = logging.getLogger()
root.setLevel(logging.WARNING)


class OnlyAppModulesFilter(logging.Filter):
    def filter(self, record):
        return record.name in MainLogger.module_set


class MainLogger:
    logging_format = '[%(asctime)s][pid:%(process)-8s][tid:%(thread)-8d][%(levelname)-7s][%(name)-18s][%(funcName)-25s] %(message)s'
    default_level = logging.INFO
    model_log_file_name = '.'.join(["model_log_" + now_string, "log"])

    module_set = set()

    def __init__(self):
        self._shared_queue = Queue(-1)
        self._listener = Process(target=MainLogger.listener_process,
                                 args=(self._shared_queue, MainLogger.listener_configurer))
        self._listener.daemon = True

    def get_queue(self):
        return self._shared_queue

    def listen_on_queue(self):
        self._listener.start()


    @staticmethod
    def listener_configurer():
        import os
        app_logger = logging.getLogger()
        fh = logging.handlers.RotatingFileHandler(filename=MainLogger.model_log_file_name,
                                                  mode='a',
                                                  maxBytes=4e7,
                                                  backupCount=6)
        fmtr = logging.Formatter(MainLogger.logging_format)
        fh.setFormatter(fmtr)
        app_logger.setLevel(MainLogger.default_level)
        current_fh_names = [fh.__dict__.get(
            'baseFilename', '') for fh in app_logger.handlers]
        if not fh.__dict__['baseFilename'] in current_fh_names:  # This prevents multiple logs to the same file
            app_logger.addHandler(fh)

    @staticmethod
    def listener_process(queue, configurer):
        configurer()
        while True:
            try:
                record = queue.get()
                if record is None:  # We send this as a sentinel to tell the listener to quit.
                    break
                logger = logging.getLogger() 
                logger.addFilter(OnlyAppModulesFilter())
                logger.handle(record)  # No level or filter logic applied - just do it!
            except Exception as e:
                import sys, traceback
                print('Whoops! Problem:', file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

    @staticmethod
    def worker_configurer(queue):
        h = logging.handlers.QueueHandler(queue)
        app_logger = logging.getLogger()
        app_logger.addHandler(h)
        app_logger.setLevel(MainLogger.default_level)
