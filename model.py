from pathlib import Path  # File system non-native OOP abstraction
import imghdr
import imageio
from main_log import logging, MainLogger
import numpy as np
from data_preprocessor import DataPreprocessor
from matplotlib import pyplot as plt
from multiprocessing import Value, Queue
from model_worker import ModelWorker

MainLogger.module_set.add(__name__)


class Model:
    chi_str = f'chi'
    stderr_str = f'stderr'
    rmse_str = f'rmse'
    exponents_str = f'exponents'

    def __init__(self, log_queue: Queue, log_level: int):
        self._supported_file_types = ['tiff']
        self._is_file_loaded = False
        self._raw_images = None
        self._tensor_stack = []
        self._intensity_threshold = 0
        self._output_nan_replacement = {Model.stderr_str: 0,
                                        Model.rmse_str: 0,
                                        Model.exponents_str: 0,
                                        Model.chi_str: 0}
        self._log_queue = log_queue
        self.log_level = log_level
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.log_level)

    @staticmethod
    def analyze_images(worker_args: tuple, log_queue: Queue, pb_value: Value, verbose_analysis: Value):
        from os import getpid
        worker = ModelWorker(*worker_args, log_queue)
        worker.analyze_images(pb_value, verbose_analysis)

    @staticmethod
    def display_first_n_pixel_histograms(worker_args: tuple, log_queue: Queue, n: int):
        worker = ModelWorker(*worker_args, log_queue)
        worker.show_pixel_histogram(n)

    def set_nan_replacement(self, output_type: str, nan_replacement: int):
        if output_type in self._output_nan_replacement:
            self.logger.info(f'Setting {output_type} nan replacement value, '
                              f'old={str(self._output_nan_replacement[output_type])}'
                              f' new={nan_replacement}')
            self._output_nan_replacement[output_type] = nan_replacement

    def clear_model(self):
        self._is_file_loaded = False
        self._raw_images = None
        self._tensor_stack = []
        self._intensity_threshold = 0
        self._output_nan_replacement = {Model.rmse_str: 0,
                                        Model.exponents_str: 0,
                                        Model.chi_str: 0,
                                        Model.stderr_str: 0}

    def __set_file_loaded(self, is_file_loaded: bool):
        self.logger.debug(f'Setting file_loaded flag <old={self._is_file_loaded}, new={is_file_loaded}>')
        self._is_file_loaded = is_file_loaded

    def set_intensity_threshold(self, threshold: int):
        self.logger.info(f'Setting intensity threshold. old={self._intensity_threshold}, new={threshold}')
        self._intensity_threshold = threshold

    def load_data(self, input_file_path: str) -> bool:
        success: bool = False
        if not self.__is_input_file_valid(input_file_path):
            self.logger.warning('Attempt to load an invalid file, ignoring')
        else:
            try:
                self._raw_images = imageio.volread(uri=input_file_path)
                self.logger.info(f'Loaded {len(self._raw_images)} x '
                                  f'{self._raw_images[0].shape} {imghdr.what(input_file_path)} images')
                self._tensor_stack.append(self._raw_images.copy())
                self.__set_file_loaded(True)
                success = True
            except Exception as e:
                self.__set_file_loaded(False)
                self.logger.info(f'Caught exception when running:\n{e}')
        return success

    def __is_input_file_valid(self, input_file_path: str) -> bool:
        self.logger.info(f'Validating input file')
        valid = False
        try:
            input_file = Path(input_file_path)
        except Exception as e:
            self.logger.info(f'Caught exception when processing path:\n{e}')
            return valid

        if not input_file.exists:
            self.logger.warning(f'Input path leads to an non-existing file')
            return valid

        if not input_file.is_file():
            self.logger.warning(f'Input path does not lead to a file')
            return valid

        try:
            file_type = imghdr.what(input_file_path)
        except Exception as e:
            self.logger.info(f'Caught exception when reading input file binary header:\n{e}')
            return valid

        if file_type not in self._supported_file_types:
            self.logger.warning(f'{file_type} is an unsupported file type')
            return valid

        self.logger.info(f'Input file is valid')
        valid = True
        return valid

    def valid_for_conv_filter(self, kernel_size):
        if len(self._tensor_stack[0]) == 0:
            return False
        image_shape = self._tensor_stack[-1][0].shape
        max_dim = max(image_shape)
        if kernel_size > max_dim:
            return False
        else:
            return True

    def valid_for_block_reduce(self, block_size):
        if len(self._tensor_stack[0]) == 0:
            return False
        image_shape = self._tensor_stack[-1][0].shape
        for dim in image_shape:
            if not (dim % block_size) == 0:
                return False
        return True

    def apply_block_reduce(self, reduce_type, block_size):
        if len(self._tensor_stack) > 0:
            self.logger.info(f'Applying {reduce_type} pooling, block size: {str(block_size)}')
            dp = DataPreprocessor(self._tensor_stack[-1], self._log_queue, self.log_level)
            dp.apply_block_reduce(reduce_type, block_size)
            self._tensor_stack.append(dp.get_processed_images())
        else:
            self.logger.error(f'Tensor stack is empty')

    def apply_filter(self, filter_type: str, kernel_shape: tuple):
        if len(self._tensor_stack) > 0:
            dp = DataPreprocessor(self._tensor_stack[-1], self._log_queue, self.log_level)
            if filter_type == 'median':
                dp.apply_median_filter(kernel_shape)
            elif filter_type == 'uniform':
                dp.apply_uniform_filter(kernel_shape)
            elif filter_type == 'gaussian':
                dp.apply_gaussian_filter(kernel_shape)
            else:
                self.logger.error(f'Unknown filter type {filter_type}')
                return
            self._tensor_stack.append(dp.get_processed_images())
        else:
            self.logger.error(f'Tensor stack is empty')

    def display_images(self):
        if len(self._tensor_stack) > 1:
            images = self._tensor_stack[-1]
            figure_data = []
            for i in range(len(images)):
                f, figure_axis = plt.subplots(1, 2)

                figure_axis[0].imshow(self._raw_images[i])
                figure_axis[0].set_title('Raw')

                figure_axis[1].imshow(images[i])
                figure_axis[1].set_title('Filtered')
                figure_data.append((f, figure_axis))
                f.suptitle(f'Image {i + 1} Before/After')
            plt.show()

        elif len(self._tensor_stack) == 1:
            images = self._tensor_stack[-1]
            figure_data = []
            for i in range(len(images)):
                f, figure_axis = plt.subplots(1, 1)
                figure_axis.imshow(self._raw_images[i])
                figure_axis.set_title(f'Raw image {i+1}')
                figure_data.append((f, figure_axis))
            plt.show()

    def discard_top_tensor(self):
        prev_size = len(self._tensor_stack)
        self._tensor_stack.pop()
        cur_size = len(self._tensor_stack)
        self.logger.info(f'Discarded top tensor from the tensor stack, stack size: (prev={prev_size}, cur={cur_size})')

    def get_tensor_stack_size(self) -> int:
        return len(self._tensor_stack)

    def get_top_images(self) -> np.ndarray:
        if len(self._tensor_stack) > 0:
            return self._tensor_stack[-1]
        else:
            return np.zeros(shape=(1, 1, 1))

    def get_intensity_threshold(self) -> int:
        return self._intensity_threshold

    def get_nan_replacements(self) -> dict:
        return self._output_nan_replacement

    def save_top_images(self, name: str):
        images = self._tensor_stack[-1]
        imageio.mimwrite(name, images, bigtiff=False)

