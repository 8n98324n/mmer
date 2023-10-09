        
import numpy as np
import os
import pandas as pd

"""
This module contains classes and methods for transforming a BVP signal in a BPM signal.
"""



class Video:
    """
    Manage (multi-channel, row-wise) BVP signals, and transforms them in BPMs.
    """


    default_landmarks_list = [3, 4, 5, 6, 8, 9, 10, 18, 32, 36, 43, 48, 50, 67, 69, 92, 101, 103, 104, 108, 109, 123, 134, 135, 142, 151, 164, 167, 182, 187, 192, 197, 201, 205, 206, 207, 210, 211, 212, 216, 248, 262, 266, 273, 278, 280, 297, 299, 322, 330, 332, 333, 337, 338, 352,  363, 364, 371, 393, 406, 411, 416, 421, 425, 426, 427, 430,431, 432, 436]

    def __init__(self):
        pass

    def check_path_or_create(self, path):
        # Check if the path exists
        import os
        if not os.path.exists(path):
            # If it does not exist, create the directory
            os.makedirs(path)
        return path

    def convert_sig_to_bpm(self, source_folder_path, target_folder_path,ldmks_list = default_landmarks_list, same_frame_rate= True):
        
        import researchtoolbox.remote_ppg.vhr.signal_process as prc
        v = prc.Video(source_folder_path, target_folder_path, ldmks_list, same_frame_rate)
        return v.convert_files_of_raw_signal_to_bpm()
    
    # def update_global_list(self, output_list, MAD_list, average_MAD, HR_change_error,  file_name, input_file_path, total_rows, output_sequence, n_errors):
    #     file_name_with_sequence = file_name.replace(".csv", "") + "_" + str(output_sequence)
    #     output_file_name = os.path.join(input_file_path,"output", file_name_with_sequence + ".csv")
    #     pd.DataFrame(output_list).to_csv(output_file_name, index=False, header=False)
    #     output_sequence += 1
    #     this_average_MAD = np.mean(MAD_list)
    #     average_MAD = np.vstack((average_MAD, [file_name_with_sequence, this_average_MAD]))
    #     HR_change_error = np.vstack((HR_change_error, [file_name_with_sequence, n_errors, total_rows]))
    #     current_time = 0
    #     n_errors = 0
    #     output_list = np.array([0])
    #     MAD_list = np.array([])
    #     return output_sequence, current_time,n_errors,output_list,MAD_list, average_MAD, HR_change_error
    
    # # def reset_global_list(self,current_time,n_errors,output_list,MAD_list):
    # #     current_time = 0
    # #     n_errors = 0
    # #     output_list = np.array([0])
    # #     MAD_list = np.array([])
    
    # def update_local_list(self, previous_HR, current_HR, current_time, new_MAD, output_list, MAD_list ):

    #     default_HR = 60 # use this HR value when there is an error
    #     HR_minimum = 50
    #     HR_maximum = 120
    #     HR_change_threshold = 20
    #     if current_HR == 0:
    #         #current_HR == 0 is an error. We use the previous_HR to replace the current_HR
    #         n_errors += 1
    #         if previous_HR!= 0:
    #             current_HR = previous_HR
    #         else:
    #             current_HR = default_HR
    #             previous_HR = default_HR
    #     else:
    #         if previous_HR != 0:
    #             #Check if the HR changes larger than theshold
    #             if abs(previous_HR - current_HR) > HR_change_threshold:
    #                 current_HR = previous_HR
    #                 n_errors += 1
    #         else:
    #             #for the first time previous_HR is 0
    #             if current_HR< HR_maximum and current_HR > HR_minimum:
    #                 previous_HR = current_HR
    #             else:
    #                 previous_HR = default_HR
    #                 current_HR = default_HR

    #     current_time = 60 / current_HR + current_time
    #     output_list = np.vstack((output_list, current_time))
    #     #MAD_list = np.append(MAD_list, input_table.loc[output_index, 2])
    #     MAD_list = np.append(MAD_list, new_MAD)
    #     return output_list, MAD_list, current_time
    #
    # def convert_bpm_to_hrv(self, bpm_output_folder_path):
    #     # Set up initial variables
    #     input_file_path = bpm_output_folder_path #"C:\\R\\bpm_results"
    #     #input_file_path_step1 = self.check_path_or_create(os.path.join(input_file_path, "Input"))
    #
    #
    #     filenames = [os.path.join(input_file_path, f) for f in os.listdir(input_file_path) if f.endswith(".csv")]
    #
    #     file_info = pd.DataFrame({"filename": filenames})
    #     file_info_ordered = file_info
    #     filenames_ordered = file_info_ordered["filename"].values
    #
    #
    #     #average_MAD =np.array(x)
    #     average_MAD = np.empty((0, 2))
    #     #average_MAD = np.empty(2)
    #     HR_change_error = np.empty((0, 3))
    #     #average_MAD = np.array([])
    #     #HR_change_error = np.array([])
    #     num_error = 0
    #     output_sequence = 0
    #     miminum_beats = 10 # Should be 10
    #     minimum_length = 120 #正常是120, 暫時改5, 做debug
    #
    #
    #     for file_index in range(len(filenames_ordered)):
    #         #Calculate the HRV one by one
    #         file_name = os.path.basename(filenames_ordered[file_index])
    #         current_time = 0
    #         output_list = np.array([0])
    #         MAD_list = np.array([])
    #         n_errors = 0
    #         input_table = pd.DataFrame()
    #         output_sequence = 0
    #         total_rows = 0
    #         previous_HR = 0
    #
    #         try:
    #             input_table = pd.read_csv(filenames_ordered[file_index], header=None)
    #             total_rows = len(input_table)
    #         except Exception as e:
    #             num_error += 1
    #             print(f"error {num_error}")
    #             print(e)
    #             continue #if there is an error in reading file, skip this file
    #
    #         if total_rows > 0:
    #             for output_index in range(total_rows):
    #                 current_HR = input_table.loc[output_index, 1]
    #
    #                 if output_index == total_rows - 1:
    #                     #To generate the one new result only when it is the last row or whenever the time exceeds the minimum
    #                     if current_time > minimum_length:
    #                         output_list, MAD_list, current_time = self.update_local_list(previous_HR, current_HR, current_time, input_table.loc[output_index, 2], output_list, MAD_list)
    #                         output_sequence, current_time,n_errors,output_list,MAD_list , average_MAD, HR_change_error = self.update_global_list(output_list,MAD_list, average_MAD, HR_change_error, file_name, input_file_path, total_rows, output_sequence, n_errors)
    #                     else:
    #                         pass
    #                 else:
    #                     #To generate the one new result only when it is the last row or whenever the time exceeds the minimum
    #                     if current_time < minimum_length:
    #                         output_list, MAD_list, current_time = self.update_local_list(previous_HR, current_HR, current_time, input_table.loc[output_index, 2], output_list, MAD_list)
    #                     else:
    #                         output_list, MAD_list, current_time = self.update_local_list(previous_HR, current_HR, current_time, input_table.loc[output_index, 2], output_list, MAD_list)
    #                         output_sequence, current_time,n_errors,output_list,MAD_list, average_MAD, HR_change_error = self.update_global_list(output_list, MAD_list, average_MAD, HR_change_error, file_name, input_file_path, total_rows, output_sequence, n_errors)
    #
    #
    #     filenames_step2 = [os.path.join(input_file_path,"output", f) for f in os.listdir(os.path.join(input_file_path,"output")) if f.endswith(".csv")]
    #     output_summay_path = self.check_path_or_create(os.path.join(input_file_path, "summary"))
    #
    #
    #     pd.DataFrame(average_MAD).to_csv(os.path.join(output_summay_path, "MAD.csv"), index=False, header=False)
    #     pd.DataFrame(HR_change_error).to_csv(os.path.join(output_summay_path, "HR_Error.csv"), index=False, header=False)
    #     data_output_list = pd.DataFrame()
    #
    #     import rpy2.robjects as robjects
    #     from rpy2.robjects import r, StrVector, ListVector, DataFrame
    #
    #     rpy2_setup_code = '''
    #     library(RHRV)
    #     suppressPackageStartupMessages(library(dplyr))
    #     '''
    #     robjects.r(rpy2_setup_code)
    #
    #
    #     r_globals = robjects.globalenv
    #
    #     data_output_list = []
    #
    #
    #     for index in range(1, len(filenames_step2) + 1):
    #         #print(index)
    #         full_file_name = filenames_step2[index - 1]
    #         file_base_name = os.path.basename(full_file_name)
    #
    #         # Read the data from the file
    #         with open(full_file_name, 'r') as file:
    #             lines = file.readlines()
    #
    #         # Remove leading/trailing whitespaces and create a comma-separated string
    #         time_series = ', '.join([line.strip() for line in lines])
    #
    #         r_code = f'''
    #             RR_series <- c({time_series})
    #             hrv.data <- CreateHRVData() |>
    #             LoadBeatVector(RR_series) |>
    #             BuildNIHR() |>
    #             InterpolateNIHR()
    #             hrv.data <- SetVerbose(hrv.data, TRUE)
    #             if (length(hrv.data$Beat$Time) > {miminum_beats}) {{
    #                 hrv.data <- suppressMessages(BuildNIHR(hrv.data))
    #                 hrv.data <- suppressMessages(FilterNIHR(hrv.data))
    #                 hrv.data$Beat <- hrv.data$Beat[hrv.data$Beat$niHR > 50, ]
    #                 hrv.data$Beat <- hrv.data$Beat[hrv.data$Beat$niHR < 120, ]
    #                 hrv.data <- suppressMessages(InterpolateNIHR(hrv.data, freqhr = 4, method = "spline"))
    #                 hrv.data <- suppressMessages(CreateTimeAnalysis(hrv.data, size = {minimum_length}, interval = 7.8125))
    #                 hrv.data <- suppressMessages(CreateFreqAnalysis(hrv.data))
    #                 hrv.data <- suppressMessages(CalculatePowerBand(hrv.data, indexFreqAnalysis = 1, size = {minimum_length}, shift = {minimum_length}, type = "fourier", ULFmin = 0, ULFmax = 0.03, VLFmin = 0.03, VLFmax = 0.05, LFmin = 0.05, LFmax = 0.15, HFmin = 0.15, HFmax = 0.4))
    #                 hr.median <- median(hrv.data$HR)
    #                 data.output <- c(hr.median, hrv.data$TimeAnalysis[[1]]$rMSSD, hrv.data$TimeAnalysis[[1]]$pNN50, hrv.data$TimeAnalysis[[1]]$SDNN, hrv.data$FreqAnalysis[[1]]$HF, hrv.data$FreqAnalysis[[1]]$LF, hrv.data$FreqAnalysis[[1]]$LFHF, hrv.data$FreqAnalysis[[1]]$VLF, hrv.data$FreqAnalysis[[1]]$ULF, hrv.data$Beat[, 1][length(hrv.data$Beat[, 1])])
    #             }} else {{
    #                 data.output <- NULL
    #             }}
    #             data.output
    #         '''
    #         data_output = robjects.r(r_code)
    #         data_output_1 = list(data_output)
    #         data_output_1.append(file_base_name)
    #         data_output_list.append(data_output_1)
    #
    #     data_output_list_df = pd.DataFrame(data_output_list, columns=['HR', 'rMSSD','pNN50','SDNN','HF','LF','LFHF','VLF','ULF','Time','ID'])
    #     data_output_list_df.to_csv(os.path.join(output_summay_path, "result.csv"), index=False, header=True)
    #     return data_output_list_df
    #