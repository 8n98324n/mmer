import cv2
import os
import numpy as np
from researchtoolbox.constant_variable import *
import pandas as pd

def rotate_image(raw_frame):  # 把图像转正
    raw_frame = cv2.transpose(raw_frame)
    raw_frame = cv2.flip(raw_frame, 0)
    raw_frame = cv2.transpose(raw_frame)
    rotate_frame = cv2.flip(raw_frame, 0)
    return rotate_frame


def get_fps(video_path):
    capture = cv2.VideoCapture(video_path)
    return int(capture.get(cv2.CAP_PROP_FPS))


def file_output(output__csv_path, base_name,sequence,new_output, summary_list):
    new_row = []
    output_file_basename = base_name + "_" + str(sequence)
    output_file_name = output__csv_path + "/" + output_file_basename + ".csv"
    if os.path.exists(output_file_name):
        return

    new_row.append(output_file_basename)
    for point_index in range(len(point_list)):
        column = np.asarray([rows[point_list[point_index]] for rows in new_output]).astype(float)
        sum_diff = 0
        previous_value = 0
        for value_index in range(len(column)):
            if value_index>1 and previous_value>0:
                sum_diff = sum_diff+ abs(column[value_index] - previous_value)
            previous_value = column[value_index]
            pass
        new_row.append(np.mean(column))
        new_row.append(np.median(column))
        new_row.append(sum_diff/len(column))
        #new_row.append(np.std(column))

    summary_list.append(new_row)

    df = pd.DataFrame(new_output)
    df.to_csv(output_file_name)

    return summary_list

