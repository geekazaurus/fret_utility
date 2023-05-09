from matplotlib import pyplot as plt
from lmfit import Model, fit_report
from multiprocessing import Value, Queue
from main_log import MainLogger, logging, datetime, datetime_str_format
import numpy as np

MainLogger.module_set.add(__name__)


class CurveFitter:

    def __init__(self, x, y, log_queue: Queue, log_level: int):
        self._x = x
        self._y = y
        self._output_dict = dict()
        self._log_queue = log_queue
        self.log_level = log_level
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.log_level)
        self._timestamp = datetime.now().strftime(datetime_str_format)

    @staticmethod
    def exponent_func(x, a, exp, b) -> float:
        return a * np.exp(-exp * x) + b

    def fit(self, verbose_analysis: Value, pixel_coords: tuple):
        self.logger.debug(f'Initiating curve fit for pixel={pixel_coords}, '
                          f'x_shape={self._x.shape}, y_shape={self._y.shape}')
        try:
            # Check whether we want to work on normalized image or not
            # and how to construct weights for the analysis.
            m = Model(self.exponent_func)
            y_max = np.iinfo(self._y.dtype).max
            y_min = np.iinfo(self._y.dtype).min
            orig_dynamic_range = y_max - y_min + 1
            weights = (1 / (np.ones(shape=self._y.shape) * np.sqrt(orig_dynamic_range)))
            results = m.fit(data=self._y,
                            x=self._x,
                            a=self._y[0] - self._y[len(self._y) - 1],
                            exp=0.5,
                            b=self._y[len(self._y) - 1],
                            weights=weights)
            if not results.success:
                self.logger.warning(f'lmfit.Model.fit() failed fitting a model to the input data')
                return

            if bool(verbose_analysis.value):
                try:
                    with open(file='_'.join([self._timestamp,
                                            'pixel',
                                            str(pixel_coords),
                                            'analysis' + '.txt']),
                              mode='w') as f:
                        f.write(fit_report(results))
                except Exception as e:
                    self.logger.error(f'Exception raised when attempting to write fit report to text file:\n{e}')
                    pass
                try:
                    f, figure_axis = plt.subplots(1, 2)
                    results.plot_residuals(figure_axis[0])
                    results.plot_fit(figure_axis[1])

                    f.suptitle(f'Pixel {pixel_coords}')
                    plt.grid(visible=True)
                    plt.show()
                except Exception as e:
                    self.logger.error(f'Exception raised when attempting to plot fit and residuals:\n{e}')
                    pass

            self._output_dict['exponent'] = results.params['exp'].value
            self._output_dict['reduced_chi_square'] = results.redchi  # reduced chi square statistic
            self._output_dict['stderr'] = results.params['exp'].stderr
            residuals = self._y - results.best_fit
            self._output_dict['rmse'] = self.__calc_root_mean_squared_error(residuals)

        except Exception as e:
            self.logger.warning(f'lmfit.Model threw an exception:\n<{e}>')

    def get_output(self) -> tuple:
        """
        :return: Returns fit output in the following tuple:
                 (predicted_exponent, standard error of exponent, reduced X^2, root mean squared error)
        """
        return self._output_dict.get('exponent', np.nan), \
               self._output_dict.get('stderr', np.nan), \
               self._output_dict.get('reduced_chi_square', np.nan), \
               self._output_dict.get('rmse', np.nan)

    def __calc_root_mean_squared_error(self, residuals) -> float:
        y_max = np.iinfo(self._y.dtype).max
        y_min = np.iinfo(self._y.dtype).min
        return np.sqrt(np.mean(residuals**2))

