# from model import Model
# import os
# from main_log import logging, MainLogger
# from fret_utility import FretUtility
#
# MainLogger.module_set.add(__name__)
#
#
# class ModelTests:
#
#     def __init__(self, log_queue):
#         self._log_queue = log_queue
#         self._logger = logging.getLogger(__name__)
#         self._logger.debug('Model unit tests object is firing up')
#
#     @staticmethod
#     def load_data_sanity():
#         m = ModelTests.model
#         test_data_path = os.path.abspath('test_data.tif')
#         m.load_data(test_data_path)
#
#     @staticmethod
#     def median_filter_images():
#         m = ModelTests.model
#         test_data_path = os.path.abspath('test_data.tif')
#         m.load_data(test_data_path)
#         m.apply_median_filter((2, 2))
#         m.display_images()
#
#     @staticmethod
#     def gaussian_filter_images():
#         m = ModelTests.model()
#         test_data_path = os.path.abspath('test_data.tif')
#         m.load_data(test_data_path)
#         m.apply_gaussian_filter((3, 3))
#         m.display_images()
#
#     @staticmethod
#     def uniform_filter_images():
#         m = ModelTests.model()
#         test_data_path = os.path.abspath('test_data.tif')
#         m.load_data(test_data_path)
#         m.apply_uniform_filter((3, 3))
#         m.display_images()
#
#     @staticmethod
#     def full_analysis():
#         m = ModelTests.model
#         test_data_path = os.path.abspath('test_data.tif')
#         m.load_data(test_data_path)
#         m.apply_median_filter((3, 3))
#         m.display_images()
#         m.analyze_images()
#         m.dump_output_to_csv()
#
#     @staticmethod
#     def run_app():
#         app = FretUtility()
#         app.fire_up_gui()
