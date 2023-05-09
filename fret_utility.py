import tkinter
from tkinter import filedialog as fd
from tkinter import messagebox
from tkinter import *
from tkinter.ttk import *
from main_log import logging, MainLogger, datetime_str_format
from model import Model
import os
from multiprocessing import Process, Value
from threading import Thread
from datetime import datetime
import time

MainLogger.module_set.add(__name__)


class FretUtility:
    root_geometry = f'700x500'
    log_severity_id_to_name = {0: "DEBUG",
                               1: "INFO",
                               2: "WARN",
                               3: "ERROR",
                               4: "CRITICAL"}

    log_filter_map = {"DEBUG": logging.DEBUG,
                      "INFO": logging.INFO,
                      "WARN": logging.WARN,
                      "ERROR": logging.ERROR,
                      "CRITICAL": logging.CRITICAL}

    def __init__(self, log_queue):
        self._root = Tk()
        self._root.title('Photo FRET Utility')
        self._root.geometry(FretUtility.root_geometry)
        self._log_queue = log_queue
        MainLogger.worker_configurer(self._log_queue)
        self.logger = logging.getLogger(__name__)
        self.log_level = MainLogger.default_level
        self.logger.setLevel(self.log_level)
        self._model = Model(self._log_queue, self.log_level)

        self._right_frame = Frame(self._root, height=480, width=340)
        self._right_frame.pack(side=RIGHT, fill=BOTH, expand=True)
        self._left_frame = Frame(self._root, height=480, width=340)
        self._left_frame.pack(side=LEFT, fill=BOTH, expand=True)

        self._general_frame = LabelFrame(self._left_frame, text="General", borderwidth=5, height=480, width=340)
        self._general_frame.pack(side=TOP, anchor='nw', expand=True, fill=BOTH)

        self._tensor_stack_size_label = Label(self._general_frame,
                                              text=f'Images stack size: {self._model.get_tensor_stack_size()}')
        self._tensor_stack_size_label.pack(anchor='nw', pady=2, padx=1)
        self._tensor_stack_size_label.configure(font=("Helvetica", 9, "bold"))

        self._load_images_btn = Button(self._general_frame, text="Load images", command=self._load_images)
        self._load_images_btn.pack(anchor='nw', pady=2, padx=1)

        self._pixel_value_histogram_frame = Frame(self._general_frame)
        self._pixel_value_histogram_frame.pack(anchor='nw')

        self._pixel_value_hist_btn = Button(self._pixel_value_histogram_frame,
                                            text="First N pixel value histograms",
                                            command=self._display_top_n_pixel_value_histograms)

        self._pixel_value_hist_btn.pack(side=tkinter.LEFT, pady=2, padx=1)
        self._pixel_value_hist_btn['state'] = 'disabled'

        self._pixel_value_hist_entry = Entry(self._pixel_value_histogram_frame, width=3)
        self._pixel_value_hist_entry.pack(side=tkinter.RIGHT, anchor='nw', pady=2, padx=1)
        self._pixel_value_hist_entry.insert(0, '1')

        self._display_top_images_btn = Button(self._general_frame, text="Show top images", command=self._display_images)
        self._display_top_images_btn.pack(anchor='nw', pady=2, padx=1)
        self._display_top_images_btn['state'] = 'disabled'

        self._discard_top_tensor_btn = Button(self._general_frame,
                                              text="Discard top images",
                                              command=self._discard_top_images)
        self._discard_top_tensor_btn.pack(anchor='nw', pady=2, padx=1)
        self._discard_top_tensor_btn['state'] = 'disabled'

        self._save_top_images_btn = Button(self._general_frame,
                                           text="Save top images",
                                           command=self._save_top_images)
        self._save_top_images_btn.pack(anchor='nw', pady=2, padx=1)

        self._logging_severity_frame = LabelFrame(self._general_frame, text="Logger level",
                                                  borderwidth=5,
                                                  height=40,
                                                  width=40)
        self._logging_severity_frame.pack(side=TOP, anchor='nw', expand=True, pady=5)
        self._logging_severity = IntVar()
        self._logging_severity.set(1)
        self._debug_log_severity_radio_button = Radiobutton(self._logging_severity_frame,
                                                            text=f'Debug',
                                                            variable=self._logging_severity,
                                                            value=0)
        self._debug_log_severity_radio_button.pack(anchor='nw')

        self._info_log_severity_radio_button = Radiobutton(self._logging_severity_frame,
                                                            text=f'Info',
                                                            variable=self._logging_severity,
                                                            value=1)
        self._info_log_severity_radio_button.pack(anchor='nw')

        self._warn_log_severity_radio_button = Radiobutton(self._logging_severity_frame,
                                                            text=f'Warning',
                                                            variable=self._logging_severity,
                                                            value=2)
        self._warn_log_severity_radio_button.pack(anchor='nw')

        self._err_log_severity_radio_button = Radiobutton(self._logging_severity_frame,
                                                            text=f'Error',
                                                            variable=self._logging_severity,
                                                            value=3)
        self._err_log_severity_radio_button.pack(anchor='nw')

        self._crit_log_severity_radio_button = Radiobutton(self._logging_severity_frame,
                                                           text=f'Critical',
                                                           variable=self._logging_severity,
                                                           value=4)
        self._crit_log_severity_radio_button.pack(anchor='nw')

        self._set_log_severity_button = Button(self._logging_severity_frame,
                                               text=f'Set logging severity',
                                               command=self._set_logging_severity)
        self._set_log_severity_button.pack()

        self._quit_button = Button(self._general_frame, text="Quit", command=self._root.quit)
        self._quit_button.pack(side='left', anchor='s', pady=2, padx=1)

        self._prepro_frame = LabelFrame(self._right_frame, text="Image pre-processing",
                                        borderwidth=5, height=240, width=340)
        self._prepro_frame.pack(side=TOP, expand=True, fill=BOTH)

        self._filter_types = {1: 'uniform',
                              2: 'median',
                              3: 'gaussian',
                              4: 'max',
                              5: 'mean',
                              6: 'median'}

        self._filter_type = IntVar(value=None)
        # -------------------------------------------------------------------------------------------------------
        prepro_frame_row = 0
        self._uniform_filter_radio_button = Radiobutton(self._prepro_frame, text=f'{self._filter_types[1]} filter',
                                                        variable=self._filter_type, value=1)
        self._uniform_filter_radio_button.grid(row=prepro_frame_row, column=0, sticky="W")
        prepro_frame_row += 1
        # -------------------------------------------------------------------------------------------------------
        self._median_filter_radio_button = Radiobutton(self._prepro_frame, text=f'{self._filter_types[2]} filter',
                                                       variable=self._filter_type, value=2)
        self._median_filter_radio_button.grid(row=prepro_frame_row, column=0, sticky="W")
        prepro_frame_row += 1
        # -------------------------------------------------------------------------------------------------------
        self._gaussian_filter_radio_button = Radiobutton(self._prepro_frame, text=f'{self._filter_types[3]} filter',
                                                         variable=self._filter_type, value=3)
        self._gaussian_filter_radio_button.grid(row=prepro_frame_row, column=0, sticky="W")
        prepro_frame_row += 1
        # -------------------------------------------------------------------------------------------------------
        self._block_reduce_max_rb = Radiobutton(self._prepro_frame,
                                                text=f'{self._filter_types[4]} pooling',
                                                variable=self._filter_type,
                                                value=4)
        self._block_reduce_max_rb.grid(row=prepro_frame_row, column=0, sticky="W")
        prepro_frame_row += 1
        # -------------------------------------------------------------------------------------------------------
        self._block_reduce_mean_rb = Radiobutton(self._prepro_frame,
                                                 text=f'{self._filter_types[5]} pooling',
                                                 variable=self._filter_type,
                                                 value=5)
        self._block_reduce_mean_rb.grid(row=prepro_frame_row, column=0, sticky="W")
        prepro_frame_row += 1
        # -------------------------------------------------------------------------------------------------------
        self._block_reduce_median_rb = Radiobutton(self._prepro_frame,
                                                   text=f'{self._filter_types[6]} pooling',
                                                   variable=self._filter_type,
                                                   value=6)

        self._block_reduce_median_rb.grid(row=prepro_frame_row, column=0, sticky="W")
        prepro_frame_row += 1
        # -------------------------------------------------------------------------------------------------------
        self._kernel_size_note_label = Label(self._prepro_frame,
                                             text="Kernel size, shape=(ksize, ksize):")
        self._kernel_size_note_label.grid(row=prepro_frame_row, column=0, sticky="W")
        self._kernel_size_entry = Entry(self._prepro_frame, width=4)
        self._kernel_size_entry.grid(row=prepro_frame_row, column=1, sticky="W")
        self._kernel_size_entry.insert(0, '2')
        prepro_frame_row += 1
        # -------------------------------------------------------------------------------------------------------
        self._apply_filter_btn = Button(self._prepro_frame,
                                        text="Apply filter",
                                        state=DISABLED,
                                        command=self._apply_filter)

        self._apply_filter_btn.grid(row=prepro_frame_row, column=0, sticky="W")
        prepro_frame_row += 1
        # -------------------------------------------------------------------------------------------------------
        self._analysis_frame = LabelFrame(self._right_frame, text="Analysis", borderwidth=5, height=240, width=340)
        self._analysis_frame.pack(side=BOTTOM, expand=True, fill=BOTH)
        analysis_frame_row = 0
        self._pixel_intensity_threshold_label = Label(self._analysis_frame,
                                                      text="Pixel intensity threshold:")
        self._pixel_intensity_threshold_label.grid(row=analysis_frame_row, column=0, sticky="W")
        self._pixel_intensity_threshold_entry = Entry(self._analysis_frame, width=6)
        self._pixel_intensity_threshold_entry.grid(row=analysis_frame_row, column=1, sticky="W")
        self._pixel_intensity_threshold_entry.insert(0, '0')
        analysis_frame_row += 1
        # -------------------------------------------------------------------------------------------------------
        self._std_err_nan_rep_label = Label(self._analysis_frame,
                                              text="Standard error nan replacement:")
        self._std_err_nan_rep_label.grid(row=analysis_frame_row, column=0, sticky="W")
        self._std_err_nan_rep_entry = Entry(self._analysis_frame, width=4)
        self._std_err_nan_rep_entry.grid(row=analysis_frame_row, column=1, sticky="W")
        self._std_err_nan_rep_entry.insert(0, '0')
        analysis_frame_row += 1
        # -------------------------------------------------------------------------------------------------------
        self._rmse_nan_rep_label = Label(self._analysis_frame,
                                              text="RMSE nan replacement:")
        self._rmse_nan_rep_label.grid(row=analysis_frame_row, column=0, sticky="W")
        self._rmse_nan_rep_entry = Entry(self._analysis_frame, width=4)
        self._rmse_nan_rep_entry.grid(row=analysis_frame_row, column=1, sticky="W")
        self._rmse_nan_rep_entry.insert(0, '0')
        analysis_frame_row += 1
        # -------------------------------------------------------------------------------------------------------
        self._exponents_nan_rep_label = Label(self._analysis_frame,
                                              text="Decay exponent nan replacement:")
        self._exponents_nan_rep_label.grid(row=analysis_frame_row, column=0, sticky="W")
        self._exponents_nan_rep_entry = Entry(self._analysis_frame, width=4)
        self._exponents_nan_rep_entry.grid(row=analysis_frame_row, column=1, sticky="W")
        self._exponents_nan_rep_entry.insert(0, '0')
        analysis_frame_row += 1
        # -------------------------------------------------------------------------------------------------------
        self._chi_nan_rep_label = Label(self._analysis_frame,
                                        text="Reduced Chi^2 nan replacement:")
        self._chi_nan_rep_label.grid(row=analysis_frame_row, column=0, sticky="W")
        self._chi_nan_rep_entry = Entry(self._analysis_frame, width=4)
        self._chi_nan_rep_entry.grid(row=analysis_frame_row, column=1, sticky="W")
        self._chi_nan_rep_entry.insert(0, '0')
        analysis_frame_row += 1
        # -------------------------------------------------------------------------------------------------------
        self._verbose_analysis_shared_state = Value('i', 0, lock=False)
        self._verbose_analysis_var = IntVar()
        self._verbose_analysis_var.set(0)
        self._verbose_analysis_checkbox = Checkbutton(self._analysis_frame,
                                                      text="Verbose analysis",
                                                      variable=self._verbose_analysis_var,
                                                      command=self._toggle_verbose_analysis_shared_state)
        self._verbose_analysis_checkbox.grid(row=analysis_frame_row, column=0, sticky="W")

        analysis_frame_row += 1

        # -------------------------------------------------------------------------------------------------------
        self._run_btn = Button(self._analysis_frame, text="Run", command=self._run_analysis)
        self._run_btn.grid(row=analysis_frame_row, column=0, sticky="w", padx=1, pady=2)
        self._run_btn['state'] = 'disabled'

        self._run_progress_var = IntVar()
        self._run_pb = Progressbar(self._analysis_frame,
                                   orient=HORIZONTAL,
                                   length=100,
                                   mode='determinate',
                                   variable=self._run_progress_var)
        self._run_pb.grid(row=analysis_frame_row, column=1, columnspan=2, sticky="w", padx=1, pady=2)
        self._run_progress_value = Value('i', 0, lock=False)

        self._run_pb_percent_label = Label(self._analysis_frame,
                                           text=f'0%'.rjust(5))
        self._run_pb_percent_label.grid(row=analysis_frame_row, column=3, sticky="W", padx=1, pady=2)
        analysis_frame_row += 1
        # -------------------------------------------------------------------------------------------------------
        self._reset_model_btn = Button(self._analysis_frame, text="Reset model", command=self._reset_model)
        self._reset_model_btn['state'] = 'disabled'
        self._reset_model_btn.grid(row=analysis_frame_row, column=0, sticky="w", padx=1, pady=2)
        analysis_frame_row += 1

        self._root.grid_rowconfigure(1, weight=1)
        self._root.grid_columnconfigure(0, weight=1)

        self._images_stack_buttons = [self._discard_top_tensor_btn,
                                      self._display_top_images_btn,
                                      self._apply_filter_btn]

        self._analysis_buttons = [self._run_btn,
                                  self._pixel_value_hist_btn] + self._images_stack_buttons

        self._final_state_enable = [self._reset_model_btn]
        self._final_state_disable = [self._run_btn]

    def _toggle_verbose_analysis_shared_state(self):
        state = bool(self._verbose_analysis_var.get())
        self.logger.info(f'Changing verbose analysis state from {bool(self._verbose_analysis_shared_state.value)} to '
                         f'{state}')
        self._verbose_analysis_shared_state.value = state

    def _display_top_n_pixel_value_histograms(self):
        n_top_images_str = self._pixel_value_hist_entry.get()

        if not n_top_images_str.isdigit() or int(n_top_images_str) <= 0:
            tkinter.messagebox.showerror(title='Illegal input',
                                         message=f'Invalid value of n first images')
            return
        else:
            self.logger.info(f'Displaying pixel histogram for n={n_top_images_str} first images')
            worker_proc_args = (self._model.get_top_images(),
                                self._model.get_intensity_threshold(),
                                self._model.get_nan_replacements(),
                                self.log_level)

            worker = Process(target=self._model.display_first_n_pixel_histograms,
                             args=(worker_proc_args, self._log_queue, int(n_top_images_str)),
                             daemon=True)
            worker.start()

    def _disable_images_stack_related_buttons(self):
        for button in self._images_stack_buttons:
            button['state'] = 'disabled'

    def _enable_images_stack_related_buttons(self):
        for button in self._images_stack_buttons:
            button['state'] = 'normal'

    def _disable_analysis_buttons(self):
        for button in self._analysis_buttons:
            button['state'] = 'disabled'

    def _enable_analysis_buttons(self):
        for button in self._analysis_buttons:
            button['state'] = 'normal'

    def _post_apply_filter(self):
        self._update_tensor_stack_size_label()
        self._reset_run_analysis()
        messagebox.showinfo(title='Preprocessing status', message='Images processed successfully.')

    def _load_images(self):
        filepath = fd.askopenfilename(filetypes=[('Tif files', '*.tif')], initialdir=os.getcwd())
        self.logger.info(f'Loading images from path:{filepath}')
        if self._model.load_data(filepath):
            self._enable_analysis_buttons()
            self._update_tensor_stack_size_label()
            self._load_images_btn['state'] = 'disabled'
            self._update_tensor_stack_size_label()

    def _update_tensor_stack_size_label(self):
        self._tensor_stack_size_label['text'] = f'Images stack size: {self._model.get_tensor_stack_size()}'

    def _set_logging_severity(self):
        new_log_level_str = FretUtility.log_severity_id_to_name.get(self._logging_severity.get(), None)
        if new_log_level_str is not None:
            new_log_level = FretUtility.log_filter_map[new_log_level_str]
            if self.logger.getEffectiveLevel() != new_log_level:
                self.logger.info(f'Log level set from {logging.getLevelName(self.logger.getEffectiveLevel())} to'
                                 f' {logging.getLevelName(new_log_level)}')
                self.log_level = new_log_level
                self.logger.setLevel(self.log_level)         # FretUtility
                # Model, CurveFitter, DataPreprocessor and ModelWorker upon creation
                self._model.logger.setLevel(self.log_level)
                self._model.log_level = self.log_level
                messagebox.showinfo(title='Set logging level', message='Logging level updated successfully')

    def _reset_model(self) -> None:
        """
        Displays an askyesno message box.
        If user chooses yes, Calls models clean_model() API method
        and greys out all analysis related buttons.
        If user chooses no, does nothing.
        """
        if messagebox.askyesno(title="Hold your horses",
                               message="Are you sure you want to reset the model?"):
            self.logger.info(f'Clearing model')
            self._model.clear_model()
            self._disable_analysis_buttons()
            self._update_tensor_stack_size_label()
            self._reset_apply_filter_progress_bar()
            self._run_progress_var.set(0)
            self._run_progress_value.value = 0
            self._filter_type.set(None)
            self._run_pb_percent_label['text'] = f'0%'.rjust(5)

            self._reset_model_btn['state'] = 'disabled'
            self._load_images_btn['state'] = 'normal'
            self._disable_images_stack_related_buttons()
            self._kernel_size_entry.delete(0, tkinter.END)
            self._kernel_size_entry.insert(0, '2')
            self._pixel_value_hist_entry.delete(0, tkinter.END)
            self._pixel_value_hist_entry.insert(0, '1')
            self._pixel_intensity_threshold_entry.delete(0, tkinter.END)
            self._pixel_intensity_threshold_entry.insert(0, '0')
            self._std_err_nan_rep_entry.delete(0, tkinter.END)
            self._std_err_nan_rep_entry.insert(0, '0')
            self._rmse_nan_rep_entry.delete(0, tkinter.END)
            self._rmse_nan_rep_entry.insert(0, '0')
            self._exponents_nan_rep_entry.delete(0, tkinter.END)
            self._exponents_nan_rep_entry.insert(0, '0')
            self._chi_nan_rep_entry.delete(0, tkinter.END)
            self._chi_nan_rep_entry.insert(0, '0')
            self._p_value_nan_rep_entry.delete(0, tkinter.END)
            self._p_value_nan_rep_entry.insert(0, '0')
            self._verbose_analysis_var.set(0)

    @staticmethod
    def _set_final_analysis_state(enable_btns, disable_btns):
        for button in enable_btns:
            button['state'] = 'normal'
        for button in disable_btns:
            button['state'] = 'disabled'
        messagebox.showinfo(title='Analysis status',
                            message='Analysis complete\n'
                                    '\nNote:\n'
                                    'Increased logging verbosity may overwhelm\n'
                                    'the logging queue facilities.\n'
                                    'In such cases, patience is key.')

    @staticmethod
    def _update_pb_progress(worker, pb_var, pb_value, pb_percent_lbl, callback=None, *callback_args):
        while worker.is_alive() or pb_value.value < 99:
            pb_var.set(pb_value.value)
            pb_percent_lbl['text'] = f'{str(pb_value.value)}%'.rjust(5)
        if callback is not None:
            callback(*callback_args)

    def _validate_nan_replacements(self):
        try:
            self._model.set_nan_replacement(self._model.stderr_str, int(self._std_err_nan_rep_entry.get()))
        except Exception:
            messagebox.showerror(title='Oops...', message='Standard error nan replacement value must be an integer')
            return False

        try:
            self._model.set_nan_replacement(self._model.rmse_str, int(self._rmse_nan_rep_entry.get()))
        except Exception:
            messagebox.showerror(title='Oops...', message='RMSE nan replacement value must be an integer')
            return False

        try:
            self._model.set_nan_replacement(self._model.exponents_str, int(self._exponents_nan_rep_entry.get()))
        except Exception:
            messagebox.showerror(title='Oops...', message='Exponents nan replacement value must be an integer')
            return False

        try:
            self._model.set_nan_replacement(self._model.chi_str, int(self._chi_nan_rep_entry.get()))
        except Exception:
            messagebox.showerror(title='Oops...', message='Reduced Chi^2 nan replacement value must be an integer')
            return False

        return True

    def _validate_intensity_threshold(self):
        try:
            self._model.set_intensity_threshold(int(self._pixel_intensity_threshold_entry.get()))
        except Exception:
            messagebox.showerror(title='Oops...',
                                 message='Pixel intensity threshold value must be an integer in [0,65535].')
            return False
        return True

    def _run_analysis(self):
        self.logger.info(f'Running analysis')
        if not self._validate_intensity_threshold() or \
           not self._validate_nan_replacements():
            return

        worker_proc_args = (self._model.get_top_images(),
                            self._model.get_intensity_threshold(),
                            self._model.get_nan_replacements(),
                            self.log_level)

        analysis_process = Process(target=self._model.analyze_images,
                                   args=(worker_proc_args,
                                         self._log_queue,
                                         self._run_progress_value,
                                         self._verbose_analysis_shared_state),
                                   daemon=True)

        pb_update_thread = Thread(target=self._update_pb_progress,
                                  args=(analysis_process,
                                        self._run_progress_var,
                                        self._run_progress_value,
                                        self._run_pb_percent_label,
                                        self._set_final_analysis_state,
                                        self._final_state_enable,
                                        self._final_state_disable),
                                  daemon=True)
        self.logger.debug(f'Dispatching progress bar update thread, tid={pb_update_thread.ident}')
        self.logger.debug(f'Dispatching worker <thread/process> id={analysis_process.ident}')
        pb_update_thread.start()  # daemon
        analysis_process.start()  # daemon

        self._run_btn['state'] = 'disabled'

    def _set_intensity_threshold(self, val: int):
        self.logger.info(f'Setting intensity threshold')
        self._model.set_intensity_threshold(val)

    def _display_images(self):
        self.logger.info(f'Displaying images')
        self._model.display_images()

    def _reset_apply_filter_progress_bar(self):
        self._apply_filter_progress_var.set(0)
        self._apply_filter_progress_value.value = 0

    def _reset_run_analysis(self):
        self._run_progress_var.set(0)
        self._run_progress_value.value = 0
        self._filter_type.set(None)
        self._run_pb_percent_label['text'] = f'0%'.rjust(5)
        self._apply_filter_btn['state'] = 'normal'
        self._run_btn['state'] = 'normal'

    def _discard_top_images(self):
        self.logger.info(f'Discarding top tensor')
        self._model.discard_top_tensor()
        if self._model.get_tensor_stack_size() == 0:
            self._disable_analysis_buttons()
            self._disable_images_stack_related_buttons()
            self._load_images_btn['state'] = 'normal'
        else:
            self._enable_analysis_buttons()
            self._reset_run_analysis()
        self._update_tensor_stack_size_label()

    @staticmethod
    def _exec_after_worker_dies(worker, callback=None, *args):
        grace_iter = 0
        grace_iterations = 4 * 2  # 2 seconds worth of sleep
        while grace_iter < grace_iterations:
            time.sleep(0.25)
            grace_iter += 1

        while worker.is_alive():
            time.sleep(0.25)

        if callback is not None:
            callback(*args)

    def _apply_filter(self):
        self.logger.info('Applying filter')
        kernel_size_input_str = self._kernel_size_entry.get()
        if not kernel_size_input_str.isdigit():
            tkinter.messagebox.showerror(title='Illegal kernel size input',
                                         message=f'Input must be an integer value.')
            return
        kernel_size = int(kernel_size_input_str)

        kernel_shape = (kernel_size, kernel_size)
        self.logger.info(f'Using {kernel_shape} kernel')
        self._apply_filter_btn['state'] = 'disabled'
        filter_type = int(self._filter_type.get())
        if 1 <= filter_type <= 3:
            if not self._model.valid_for_conv_filter(kernel_size):
                messagebox.showerror(title='Oops..', message='Invalid kernel size.\n'
                                                             'Kernel size should be smaller than\n'
                                                             'the spatial dimensions of the images.')
                self._apply_filter_btn['state'] = 'normal'
                return
            filter_worker = Thread(target=self._model.apply_filter,
                                   args=(self._filter_types.get(filter_type, 'n/a'), kernel_shape),
                                   daemon=True)
            post_filter_actions_thread = Thread(target=self._exec_after_worker_dies,
                                                args=(filter_worker,
                                                      self._post_apply_filter,),
                                                daemon=True)
            filter_worker.start()
            post_filter_actions_thread.start()
        elif 4 <= filter_type <= 6:
            if not self._model.valid_for_block_reduce(kernel_size):
                messagebox.showerror(title='Oops..', message='Invalid block size.\n'
                                                             'Image spatial dimensions should be\n'
                                                             'integer divisible by block size')
                self._apply_filter_btn['state'] = 'normal'
                return
            block_reduce_worker = Thread(target=self._model.apply_block_reduce,
                                         args=(self._filter_types.get(filter_type, 'n/a'), kernel_size),
                                         daemon=True)
            post_filter_actions_thread = Thread(target=self._exec_after_worker_dies,
                                                args=(block_reduce_worker,
                                                      self._post_apply_filter,),
                                                daemon=True)
            block_reduce_worker.start()
            post_filter_actions_thread.start()
        else:
            self.logger.warning(f'Received invalid preprocessing filter type: {filter_type}')
            return

    def _save_top_images(self):
        if self._model.get_tensor_stack_size() == 0:
            messagebox.showerror(title='Oops...', message='Images stack is empty, no images to save.')
        else:
            up_to_date_now_str = datetime.now().strftime(datetime_str_format)
            output_file_name = '_'.join(['fretutil_saved_stack', up_to_date_now_str]) + '.tif'
            self.logger.info(f'Saving images from the top of the stack to file: {output_file_name}')
            self._model.save_top_images(output_file_name)
            messagebox.showinfo(title='Save images status', message=f'Successfully saved images to {output_file_name}')

    def fire_up_gui(self):
        self.logger.info(f'Firing up GUI')
        self._root.mainloop()
