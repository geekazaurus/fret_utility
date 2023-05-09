from main_log import MainLogger
from fret_utility import FretUtility
from multiprocessing import freeze_support


if __name__ == '__main__':
    freeze_support()
    ml = MainLogger()
    ml.listen_on_queue()
    app = FretUtility(ml.get_queue())
    app.fire_up_gui()
