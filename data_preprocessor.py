import numpy as np
from scipy import ndimage
from main_log import logging, MainLogger
from skimage.measure import block_reduce
from multiprocessing import Queue

MainLogger.module_set.add(__name__)


class DataPreprocessor:

    def __init__(self, images: np.ndarray, log_queue: Queue, log_level: int):
        self.__images = images
        self._dtype = images.dtype
        self._log_queue = log_queue
        self.log_level = log_level
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.log_level)
        self.__kernel_shape = None
        self.__processed_images = list()

    @staticmethod
    def __get_num_kernel_elements(kernel_shape) -> int:
        kernel_elements: int = 1
        for dim_size in kernel_shape:
            kernel_elements *= dim_size
        return kernel_elements

    def apply_block_reduce(self, reduce_type, block_size):
        orig_dtype = self._dtype
        reduce_type_to_func = {'max': np.max,
                               'mean': np.mean,
                               'median': np.median}
        reduce_func = reduce_type_to_func[reduce_type]
        self.logger.debug(f'Applying block reduce of type {reduce_type} with block size {block_size}')
        for i in range(len(self.__images)):
            self.__processed_images.append(block_reduce(image=self.__images[i],
                                                        block_size=block_size,
                                                        func=reduce_func).astype(orig_dtype,
                                                                                 copy=False))
            self.logger.debug(f'Done filtering image {i + 1}/{len(self.__images)}')

    def apply_median_filter(self, kernel_shape):
        orig_dtype = self.__images.dtype
        kernel_elements = DataPreprocessor.__get_num_kernel_elements(kernel_shape)
        self.logger.debug(f'Applying median filter with a {kernel_shape} kernel')
        for i in range(len(self.__images)):
            self.__processed_images.append(ndimage.filters.median_filter(input=self.__images[i],
                                                                         size=kernel_elements).astype(orig_dtype,
                                                                                                      copy=False))
            self.logger.debug(f'Done filtering {i+1}/{len(self.__images)}')

    def apply_gaussian_filter(self, kernel_size):
        """
        Sigma heuristic taken from:
        https://stackoverflow.com/questions/3149279/optimal-sigma-for-gaussian-filtering-of-an-image
        """
        orig_dtype = self.__images.dtype
        sigma = (DataPreprocessor.__get_num_kernel_elements(kernel_size)-1)/6
        self.logger.debug(f'Applying gaussian filter with sigma={sigma}')
        for i in range(len(self.__images)):
            self.__processed_images.append(ndimage.filters.gaussian_filter(input=self.__images[i],
                                                                           sigma=sigma).astype(orig_dtype,
                                                                                               copy=False))
            self.logger.debug(f'Done filtering {i + 1}/{len(self.__images)}')

    def apply_uniform_filter(self, kernel_size):
        orig_dtype = self.__images.dtype
        kernel_elements = DataPreprocessor.__get_num_kernel_elements(kernel_size)
        self.logger.debug(f'Applying uniform filter with {kernel_size} kernel')
        for i in range(len(self.__images)):
            self.__processed_images.append(ndimage.filters.uniform_filter(input=self.__images[i],
                                                                          size=kernel_elements).astype(orig_dtype,
                                                                                                       copy=False))
            self.logger.debug(f'Done filtering {i + 1}/{len(self.__images)}')

    def get_processed_images(self):
        return np.asarray(self.__processed_images, dtype=self._dtype)
