import researchtoolbox.remote_ppg.vhr.filters as filters
from os import listdir
from os.path import isfile, join
import os
import matplotlib.pyplot as plt
import numpy as np
import shutil
import pickle
from inspect import getmembers, isfunction
from importlib import import_module, util
from pyVHR.extraction.sig_processing import SignalProcessing, SignalProcessingParams, sig_windowing, get_fps
from pyVHR.extraction.skin_extraction_methods import SkinExtractionConvexHull, SkinExtractionFaceParsing, SkinProcessingParams
from pyVHR.BVP.BVP import *
from pyVHR.BPM.BPM import *
from pyVHR.BVP.methods import *
import pandas as pd
import researchtoolbox.utility.os as ros
"""
This module contains classes and methods for transforming a BVP signal in a BPM signal.
"""

class Video:

    """
    Manage (multi-channel, row-wise) BVP signals, and transforms them in BPMs.
    """
    def __init__(self, source_folder_path, target_folder_path, ldmks_list, same_frame_rate= True):
        self.source_folder_path = source_folder_path 
        self.target_folder_path =  ros.Path().check_path_or_create(os.path.join(target_folder_path, "video_output"))
        self.graph_output_folder_path = ros.Path().check_path_or_create(os.path.join(target_folder_path, "graph_output"))
        self.sig_output_folder_path = ros.Path().check_path_or_create(os.path.join(target_folder_path, "sig_output"))
        self.bvps_output_folder_path = ros.Path().check_path_or_create(os.path.join(target_folder_path, "bvps_output"))
        self.bpmES_output_folder_path = ros.Path().check_path_or_create(os.path.join(target_folder_path, "bpmES_output"))
        self.bpm_output_folder_path = ros.Path().check_path_or_create(os.path.join(target_folder_path, "bpm_output"))
        self.video_files_name = [f for f in listdir(self.source_folder_path) if isfile(join(self.source_folder_path, f))]
        self.sig_output_files_name = [f for f in listdir(self.sig_output_folder_path) if isfile(join(self.sig_output_folder_path, f))]
        self.sig_processing = SignalProcessing()
        self.to_recalculate = False  # 是否重新建計算而不要用舊的結果
        self.roi_approach="patches"
        self.roi_method="faceparsing"
        self.cuda=True
        self.method='cupy_POS'
        self.bpm_type='welch'
        self.pre_filt=False
        self.post_filt=True
        self.ldmks_list= ldmks_list 
        self.length_of_windows = 12
        self.extension = "mp4"
        self.move_file_after_processing = True  # 是否重新建計算而不要用舊的結果


    def convert_files_of_raw_signal_to_bpm(self):
        if (len(self.sig_output_files_name)) > 0:
            # if extracted signal exists, then the process start with the signal instead of the raw video
            for index in range(len(self.sig_output_files_name)):
                file_name = os.path.basename(self.sig_output_files_name[index]).replace("pkl",self.extension)
                bpm_file_name = os.path.join(self.bpm_output_folder_path, os.path.basename(self.sig_output_files_name[index]).replace("pkl","csv")) 
                # print("file exists" + bpm_file_name)
                if not os.path.exists(bpm_file_name):
                    self.convert_file_of_raw_signal_to_bpm(file_name)
        else:
            for index in range(len(self.video_files_name)):
                pkl_file_name = os.path.basename(self.video_files_name[index]).replace(self.extension,"csv")
                bpm_file_name = os.path.join(self.bpm_output_folder_path, os.path.basename(pkl_file_name)) 
                if not os.path.exists(bpm_file_name):
                    self.convert_file_of_raw_signal_to_bpm(self.video_files_name[index])
        return self.bpm_output_folder_path
    
    def convert_file_of_raw_signal_to_bpm(self, video_file_name):
        video_file_full_name =self.source_folder_path + "/" + video_file_name
        target_video_file_full_name =self.target_folder_path + "/" + video_file_name
        video_file_base_name = 	os.path.splitext(os.path.basename(video_file_name))[0]
        image_output_full_name = self.graph_output_folder_path + "/" + video_file_base_name + ".png"
        fps = get_fps(video_file_full_name)
        if fps == 0:
            # When raw video doesn't exist or when there exists error in extracting signals
            fps = 30
        # Determine sig_processing parameters
        if self.cuda:
            #self.sig_processing.display_cuda_device()
            self.sig_processing.choose_cuda_device(0)

        # set skin extractor
        target_device = 'GPU' if self.cuda else 'CPU'
        if self.roi_method == 'convexhull':
            self.sig_processing.set_skin_extractor(
            SkinExtractionConvexHull(target_device))
        elif self.roi_method == 'faceparsing':
            self.sig_processing.set_skin_extractor(
            SkinExtractionFaceParsing(target_device))
        else:
            raise ValueError("Unknown 'roi_method'")
        
        # set patches
        if self.roi_approach == 'patches':
            #ldmks_list = ast.literal_eval(landmarks_list)
            #if len(ldmks_list) > 0:
            self.sig_processing.set_landmarks(self.ldmks_list)
            # set squares patches side dimension
            self.sig_processing.set_square_patches_side(28.0)

        # set sig-processing and skin-processing params
        SignalProcessingParams.RGB_LOW_TH = 75
        SignalProcessingParams.RGB_HIGH_TH = 230
        SkinProcessingParams.RGB_LOW_TH = 75
        SkinProcessingParams.RGB_HIGH_TH = 230
        self.sig_processing.set_total_frames(0)
            
        # -- ROI selection
        #檢查sig是否完成
        sig_file_full_name = self.sig_output_folder_path  + "/" + video_file_base_name + ".pkl"
        sig = []

        if os.path.exists(sig_file_full_name) and not self.to_recalculate:
            pkl_file = open(sig_file_full_name, 'rb')
            sig = pickle.load(pkl_file)
            pkl_file.close()
        else:
            try: #r_01_05
                if self.roi_approach == 'hol':
                    # SIG extraction with holistic
                    sig = self.sig_processing.extract_holistic(video_file_full_name)
                elif self.roi_approach == 'patches':
                    # SIG extraction with patches
                    sig = self.sig_processing.extract_patches(video_file_full_name, 'squares', 'mean')
                output = open(sig_file_full_name, 'wb')
                pickle.dump(sig, output)
                output.close()
            except Exception as inst:
                print("error r_01_05")
                print(inst)
        


        #用6秒分割，再去计算心率，
        windowed_sig, timesES = sig_windowing(sig, self.length_of_windows , 1, fps)
        #如果不算心率變異, 下面這行可以取回全部
        #windowed_sig, timesES = sig_windowing(sig, np.floor(len(sig)/fps), np.floor(len(sig)/fps), fps)

        bvps_file_full_name = self.bvps_output_folder_path + "/" +  video_file_base_name + ".pkl"
        if os.path.exists(bvps_file_full_name) and not self.to_recalculate:
            pkl_file = open(bvps_file_full_name, 'rb')
            bvps = pickle.load(pkl_file)
            pkl_file.close()
        else:
            try: #r_01_02

                # -- PRE FILTERING
                filtered_windowed_sig = windowed_sig

                if self.pre_filt:
                    # -- color threshold - applied only with patches
                    # [Research_Tool_Box_Revised] This filter doesn't seem to be very meaningful
                    if self.roi_approach == 'patches':
                        filtered_windowed_sig = filters.apply_filter(windowed_sig,
                                                                filters.rgb_filter_th,
                                                                params={'RGB_LOW_TH':  75,
                                                                        'RGB_HIGH_TH': 230})

                # -- BVP Extraction
                module = import_module('pyVHR.BVP.methods')
                method_to_call = getattr(module, self.method)
                if 'cpu' in self.method:
                    method_device = 'cpu'
                elif 'torch' in self.method:
                    method_device = 'torch'
                elif 'cupy' in self.method:
                    method_device = 'cuda'

                if 'POS' in self.method:
                    pars = {'fps':'adaptive'}
                elif 'PCA' in self.method or 'ICA' in self.method:
                    pars = {'component': 'all_comp'}
                else:
                    pars = {}
                    #

                # Transform an input RGB windowed signal in a BVP windowed signal using a rPPG method (see pyVHR.BVP.methods).
                bvps = RGB_sig_to_BVP(filtered_windowed_sig, fps,
                                        device_type=method_device, method=method_to_call, params=pars)            
                output = open(bvps_file_full_name, 'wb')
                pickle.dump(bvps, output)
                output.close()
            except Exception as inst:
                print("error r_01_02")
                print(inst) 

         # -- POST FILTERING
        if self.post_filt:
            try: #r_01_03
                module = import_module('pyVHR.BVP.filters')
                method_to_call = getattr(module, 'BPfilter')
                bvps = filters.apply_filter(bvps, 
                                    method_to_call, 
                                    fps=fps, 
                                    params={'minHz':0.65, 'maxHz':4.0, 'fps':'adaptive', 'order':6})
            except Exception as inst:
                print("error r_01_03")
                print(inst) 

        if self.bpm_type == 'welch':
            if self.cuda:
                bpmES = BVP_to_BPM_cuda(bvps, fps, minHz=0.65, maxHz=4.0)
            else:
                bpmES = BVP_to_BPM(bvps, fps, minHz=0.65, maxHz=4.0)
        elif self.bpm_type == 'psd_clustering':
            if self.cuda:
                bpmES = BVP_to_BPM_PSD_clustering_cuda(bvps, fps, minHz=0.65, maxHz=4.0)
            else:
                bpmES = BVP_to_BPM_PSD_clustering(bvps, fps, minHz=0.65, maxHz=4.0)
        else:
            raise ValueError("Unknown 'bpm_type'")

        # median BPM from multiple estimators BPM
        median_bpmES, mad_bpmES = BPM_median(bpmES)


        bpm_results = pd.DataFrame(
        {'time': timesES,
        'median': median_bpmES,
        'lst3Title': mad_bpmES
        })

        
        plt.title(video_file_base_name)
        #plt.figure()
        plt.plot(timesES, median_bpmES)
        plt.fill_between(timesES, median_bpmES-mad_bpmES, median_bpmES+mad_bpmES, alpha=0.2)
        plt.savefig(image_output_full_name)
        fig1 = plt.figure(video_file_base_name)

        bpmES_file_full_name = self.bpmES_output_folder_path + "/" + video_file_base_name + ".csv"
        try:
            bpmES_frame = pd.DataFrame(bpmES) 
            np.savetxt(bpmES_file_full_name,bpmES_frame,delimiter=",")
        except Exception as inst:
            print("error r_01_07")
            print(inst) 

        bpm_file_full_name = self.bpm_output_folder_path + "/" + video_file_base_name + ".csv"
        np.savetxt(bpm_file_full_name,bpm_results,delimiter=",")

        try: #r_01_03
            if self.move_file_after_processing:
                shutil.move(video_file_full_name, target_video_file_full_name)
        except Exception as inst:
            pass
            #print("error r_01_08")
            #print(inst) 

            
