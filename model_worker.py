from main_log import logging, MainLogger, now_string
import numpy as np
from multiprocessing import Value, Queue
from curve_fit import CurveFitter
from matplotlib import pyplot as plt


MainLogger.module_set.add(__name__)


class ModelWorker:
    chi_str = f'chi'
    stderr_str = f'stderr'
    rmse_str = f'rmse'
    exponents_str = f'exponents'

    def __init__(self,
                 images,
                 intensity_threshold: int,
                 nan_replacement: dict,
                 log_level: int,
                 log_queue: Queue):
        self._images = images
        self._exponents = None
        self._stderr = None
        self._rmse = None
        self._chi_squared = None
        self._intensity_threshold = intensity_threshold
        self._output_nan_replacement = nan_replacement
        self._log_queue = log_queue
        self.log_level = log_level
        MainLogger.worker_configurer(self._log_queue)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.log_level)

    def _create_output_matrices(self, output_shape):
        self.logger.info(f'Creating output matrices')
        self._exponents = np.zeros(shape=output_shape)
        self._stderr = np.zeros(shape=output_shape)
        self._rmse = np.zeros(shape=output_shape)
        self._chi_squared = np.zeros(shape=output_shape)

    def _set_output_to_matrices(self, x, y, fit_output):
        self._exponents[y, x] = fit_output[0]
        self._stderr[y, x] = fit_output[1]
        self._chi_squared[y, x] = fit_output[2]
        self._rmse[y, x] = fit_output[3]

    def _replace_nans(self):
        self.logger.info(f'Replacing np.nan values')
        np.nan_to_num(x=self._exponents, copy=False, nan=self._output_nan_replacement.get(ModelWorker.exponents_str, 0))
        np.nan_to_num(x=self._stderr, copy=False, nan=self._output_nan_replacement.get(ModelWorker.stderr_str, 0))
        np.nan_to_num(x=self._chi_squared, copy=False, nan=self._output_nan_replacement.get(ModelWorker.chi_str, 0))
        np.nan_to_num(x=self._rmse, copy=False, nan=self._output_nan_replacement.get(ModelWorker.rmse_str, 0))
        self.logger.info(f'Done')

    def analyze_images(self, progress_value: Value, verbose_analysis: Value):
        self.logger.info(f'Starting image analysis')
        preprocessed_images = self._images
        output_shape = preprocessed_images[0].shape
        self._create_output_matrices(output_shape)

        num_pixels = output_shape[0] * output_shape[1]
        # For each pixel, analyze it and append data to relevant output matrix
        # then update the progress bar
        pixel_index = 0
        progress = 0
        for y in range(output_shape[1]):
            for x in range(output_shape[0]):
                pixel_index += 1
                fit_output = self._analyze_pixel(x, y, verbose_analysis)
                self._set_output_to_matrices(x, y, fit_output)
                new_progress = (pixel_index * 100) // num_pixels
                if new_progress > progress:
                    progress_value.value = new_progress
                    progress = new_progress
                    self.logger.info(f'Analysis status: {progress}%')

        # Quick and dirty fix for when pixel value != 100 here
        if progress_value.value < 100:
            progress_value.value = 100
            self.logger.info(f'Analysis status dirty patch promotion, progress: {progress}%')

        # Replace nans in all matrices
        nan_replacement = 0
        self._replace_nans()
        self.logger.info(f'Done analyzing images')
        self._dump_output_to_csv()

    def _analyze_pixel(self, x, y, verbose_analysis: Value) -> tuple:
        self.logger.debug(f'Start analysis for pixel: ({y}, {x})')
        # Prepare input series
        preprocessed_images = self._images
        orig_dtype = self._images.dtype
        x_data = np.array([val for val in range(len(preprocessed_images))], dtype=orig_dtype)
        y_data = np.asarray([preprocessed_images[i, y, x] for i in range(len(preprocessed_images))],
                            dtype=orig_dtype)
        fitter = CurveFitter(x_data, y_data, self._log_queue, self.log_level)

        threshold_test = [True if pixel_intensity > self._intensity_threshold else False for pixel_intensity in y_data]
        if not any(threshold_test):
            failing_image_indices = [i for i in threshold_test if i is False]
            failing_pixel_values = [val for val in y_data if val < self._intensity_threshold]
            self.logger.debug(f'Threshold test failed, Skipping pixel analysis')
        else:
            # Optimize curve parameters to fit data
            fitter.fit(verbose_analysis, (y, x))

        self.logger.debug(f'Finished analyzing pixel: ({y}, {x})')
        return fitter.get_output()

    @staticmethod
    def _get_output_csv_name(output_name):
        return '_'.join([output_name, now_string]) + '.csv'

    def _dump_output_to_csv(self):
        np.savetxt(fname=self._get_output_csv_name(ModelWorker.exponents_str), X=self._exponents, delimiter=',')
        np.savetxt(fname=self._get_output_csv_name(ModelWorker.stderr_str), X=self._stderr, delimiter=',')
        np.savetxt(fname=self._get_output_csv_name(ModelWorker.rmse_str), X=self._rmse, delimiter=',')
        np.savetxt(fname=self._get_output_csv_name(ModelWorker.chi_str), X=self._chi_squared, delimiter=',')

    def show_pixel_histogram(self, n):
        images = self._images

        # Create per-image pixel histograms
        image_pixel_data = list()
        for i in range(len(images)):
            image_pixel_data.append(images[i].flatten())

        if n > len(images):
            self.logger.warning(f'Input num images to analyze {n} is greater than number of images {len(images)},\n'
                                f' setting n=1')
            n = 1

        # Create figures depicting per-image pixel histogram and finally display
        figure_data = []
        for i in range(n):
            pixel_data = np.around(image_pixel_data[i]).astype(int)
            unique_pixel_values = set(pixel_data)
            f = plt.figure(i+1)
            plt.hist(x=pixel_data, bins=len(unique_pixel_values), width=1, align='mid')
            plt.title(f'Image {i+1} pixel value histogram')
            plt.xlabel('Pixel intensity')
            plt.ylabel('Count')
            plt.grid(visible=True)
            figure_data.append(f)
        plt.show()

