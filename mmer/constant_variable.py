hrv_variable_list = ["HR", "SDNN", "rMSSD", "pNN50", "ln(HF)", "ln(LF)", "ln(LF/HF)"]
hrv_variable_index = ["{}".format(i) for i in hrv_variable_list]  # remote PPG
hrv_difference_index = ["HR_difference", "SDNN_difference", "rMSSD_difference", "pNN50_difference", "HF_difference", "LF_difference", "LF/HF_difference"]


thermal_index_list = [i for i in range(61)]
thermal_selected_index_list = [18, 25, 21, 22, 58, 28, 29, 30, 32, 34, 49, 52, 53, 51, 48, 50, 55, 57, 54, 56, 59, 60]
thermal_point_difference_index = [str(ind) + "_difference" for ind in thermal_index_list]
thermal_selected_point_difference_index = [str(ind) + "_difference" for ind in thermal_selected_index_list]
thermal_point_mean_index = [str(ind) + "_mean" for ind in thermal_index_list]
thermal_selected_point_mean_index = [str(ind) + "_mean" for ind in thermal_selected_index_list]

point_RR_index = [str(ind)+"_std" for ind in thermal_selected_index_list]
point_mean_index = [str(ind)+"_mean" for ind in thermal_selected_index_list]
point_difference_index = [str(ind)+"_difference" for ind in thermal_selected_index_list]
point_start_index = [str(ind)+"_start" for ind in thermal_selected_index_list]
point_end_index = [str(ind)+"_end" for ind in thermal_selected_index_list]

index_of_left_eyebrow_outer = 18
index_of_right_eyebrow_outer = 25
index_of_left_eyebrow_inner = 21
index_of_right_eyebrow_inner = 22
index_of_fore_head = 58
index_of_upper_nose = 28
index_of_middle_nose = 29
index_of_nose_tip = 30
index_of_left_nostril = 32
index_of_right_nostril = 34
index_of_left_nose = 55
index_of_right_nose = 57
index_of_left_cheek = 54
index_of_right_cheek = 56
index_of_chin = 59
index_of_throat = 60
index_of_left_lip = 48
index_of_upper_lip_outer = 49
index_of_upper_lip = 52
index_of_right_lip = 50
index_of_lower_iip_outer = 51
index_of_lower_iip = 53

point_list = range(61)