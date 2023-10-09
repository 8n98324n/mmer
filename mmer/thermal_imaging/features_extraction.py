# import the necessary packages
from imutils import paths
from imutils import face_utils
import imutils
import numpy as np
import dlib
import cv2
import os
import matplotlib.pyplot as plt

import researchtoolbox.utility.os as ruo

# 如果想要旋转图像的话，见researchtoolbox.thermal_imaging.utils

class features_extraction:
    def __init__(self, model_path, input_file_path, output_file_path, para_dict):
        cp = ruo.Path()

        self.model_path = model_path
        self.input_file_path = input_file_path
        self.output_mp4_path = output_file_path+"video_output/"
        self.no_rectangle_image_path = output_file_path+"test_images_output/"
        self.visualization_path = output_file_path+"visualization_output/"
        self.output_csv_path = output_file_path+"csv_output/"

        cp.check_path_or_create(self.model_path)
        cp.check_path_or_create(self.input_file_path)
        cp.check_path_or_create(self.output_mp4_path)
        cp.check_path_or_create(self.no_rectangle_image_path)
        cp.check_path_or_create(self.output_csv_path)
        cp.check_path_or_create(self.visualization_path)

        self.n_feature_points = para_dict.get("n_feature_points", 54+7)
        self.n_sampling = para_dict.get("n_sampling", 30)
        self.resize_height = para_dict.get("resize_height", 600)
        self.upsampling_times = para_dict.get("upsampling_times", 1)
        self.minimum_gray_color = para_dict.get("minimum_gray_color", 30)

        self.height_lower_threshold = np.int16(self.resize_height*0.2)
        self.height_upper_threshold = np.int16(self.resize_height * 0.8)
        self.adaptive_length = np.int16(30 * 10 / self.n_sampling)  # 多少禎以後，限制數值變化的範圍


    def run(self):
        # load the face detector (HOG-SVM)
        print("[INFO] loading dlib thermal face detector...")
        detector = dlib.simple_object_detector(os.path.join(self.model_path, "dlib_face_detector.svm"))

        # load the facial landmarks predictor
        print("[INFO] loading facial landmark predictor...")
        predictor = dlib.shape_predictor(os.path.join(self.model_path, "dlib_landmark_predictor.dat"))
        video_files = list(paths.list_files(self.input_file_path))
        # loop over the images
        for ind, input_file_fullname in enumerate(video_files, 1):
            n_error = 0
            print("[INFO] Processing video: {}/{}".format(ind, len(video_files)))
            file_name = os.path.splitext(os.path.basename(input_file_fullname))[0]
            output_file_fullname = self.output_mp4_path + "output_" + file_name + ".mp4"
            output_csv_file_fullname = self.output_csv_path + file_name + ".csv"

            # initialize the video stream
            vs = cv2.VideoCapture(input_file_fullname)
            length = int(vs.get(cv2. CAP_PROP_FRAME_COUNT))
            # initialize the video writer
            writer = None
            (W, H) = (None, None)
            gray_list = []
            gray_list_for_chart = []
            # 取得平均值 54+7維
            average_list = np.zeros(self.n_feature_points)
            # 禎數
            n_frame = 0
            n_grab = 0
            # 取得平均寛度
            rec_height_list = []
            x_list = []
            y_list = []
            previous_gray_sub_list = np.zeros(self.n_feature_points)
            while True:
                #print(str(n_grab)+" out of "+str(length))
                # read the next frame from the file
                (grabbed, raw_frame) = vs.read()
                n_grab = n_grab+1

                # break the loop if the frame
                # was not grabbed
                if not grabbed:
                    break

                if np.remainder(n_grab, self.n_sampling) != 0:  # 说明不是要抓取的帧
                    continue

                # resize the frame
                frame = imutils.resize(raw_frame, height=self.resize_height)

                # copy the frame
                frame_copy = frame.copy()

                # convert the frame to grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

                # detect faces in the frame
                rects = detector(frame, upsample_num_times=self.upsampling_times)

                if len(rec_height_list) == self.adaptive_length:
                    height_lower_threshold = np.median(rec_height_list[-1*self.adaptive_length:])*0.7
                    height_upper_threshold = np.median(rec_height_list[-1*self.adaptive_length:])*1.3
                    x_average = np.median(x_list[-1*self.adaptive_length:])
                    y_average = np.median(y_list[-1*self.adaptive_length:])

                # loop over the bounding boxes
                if len(rects) > 0:
                    for rect in rects:  # 抓取多张脸
                        # if len(rects) >= 1:
                        # rect = rects[0]  # 只抓取1张脸
                        # convert the dlib rectangle into an OpenCV bounding box
                        (x, y, w, h) = face_utils.rect_to_bb(rect)

                        if len(rec_height_list) < self.adaptive_length:
                            y_average = y

                        y_list.append(y)
                        x_list.append(x)
                        if self.height_lower_threshold < h < self.height_upper_threshold and np.abs(y-y_average)/y < 0.5*h:
                            rec_height_list.append(h)  # 方框只在合規的範圍內調整
                            try:
                                # draw a bounding box surrounding the face
                                cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

                                # predict the location of facial landmark coordinates then
                                # convert the prediction to an easily parsable NumPy array
                                shape = predictor(frame, rect)
                                shape = face_utils.shape_to_np(shape)

                                # 左側臉頰加點
                                point_4_x = shape[3][0]
                                point_4_y = shape[3][1]
                                point_32_x = shape[31][0]
                                point_32_y = shape[31][1]
                                point_left_cheek_x = np.int16(0.7*point_4_x+0.3*point_32_x)
                                point_left_cheek_y = np.int16(0.7*point_4_y+0.3*point_32_y)
                                shape = np.vstack([shape, [point_left_cheek_x, point_left_cheek_y]])
                                point_left_nose_x = np.int16(0.2*point_4_x+0.8*point_32_x)
                                point_left_nose_y = np.int16(0.2*point_4_y+0.8*point_32_y)
                                shape = np.vstack([shape, [point_left_nose_x, point_left_nose_y]])
                                # 右側臉頰加點
                                point_14_x = shape[13][0]
                                point_14_y = shape[13][1]
                                point_36_x = shape[35][0]
                                point_36_y = shape[35][1]
                                point_right_cheek_x = np.int16(0.7*point_14_x+0.3*point_36_x)
                                point_right_cheek_y = np.int16(0.7*point_14_y+0.3*point_36_y)
                                shape = np.vstack([shape, [point_right_cheek_x, point_right_cheek_y]])
                                point_right_nose_x = np.int16(0.2*point_14_x+0.8*point_36_x)
                                point_right_nose_y = np.int16(0.2*point_14_y+0.8*point_36_y)
                                shape = np.vstack([shape, [point_right_nose_x, point_right_nose_y]])
                                # forehead
                                point_22_x = shape[21][0]
                                point_22_y = shape[21][1]
                                point_23_x = shape[22][0]
                                point_23_y = shape[22][1]
                                point_28_x = shape[27][0]
                                point_28_y = shape[27][1]
                                point_fore_head_x = np.int16(point_22_x + point_23_x - point_28_x)
                                point_fore_head_y = np.int16(0.75*point_22_y + 0.75*point_23_y - 0.5*point_28_y)
                                shape = np.vstack([shape, [point_fore_head_x, point_fore_head_y]])
                                # chin
                                point_9_x = shape[8][0]
                                point_9_y = shape[8][1]
                                point_52_x = shape[51][0]
                                point_52_y = shape[51][1]
                                point_chin_x = np.int16(0.5*point_9_x+0.5*point_52_x)
                                point_chin_y = np.int16(0.5*point_9_y+0.5*point_52_y)
                                shape = np.vstack([shape, [point_chin_x, point_chin_y]])
                                # throat
                                # 臉向右的話,取左側,臉向左,取右側

                                point_52_x = shape[51][0]
                                point_52_y = shape[51][1]
                                if shape[30][0]-shape[27][0] > 0:
                                    point_8_x = shape[7][0]
                                    point_8_y = shape[7][1]
                                    point_throat_x = np.int16(1.4*point_8_x-0.4*point_52_x)
                                    point_throat_y = np.int16(1.4*point_8_y-0.4*point_52_y)
                                else:
                                    point_10_x = shape[9][0]
                                    point_10_y = shape[9][1]
                                    point_throat_x = np.int16(1.4*point_10_x-0.4*point_52_x)
                                    point_throat_y = np.int16(1.4*point_10_y-0.4*point_52_y)
                                shape = np.vstack([shape, [point_throat_x, point_throat_y]])
                                gray_sub_list = []

                                n_frame = n_frame+1
                                for i in range(self.n_feature_points):
                                    selected_point_y = shape[i][1]
                                    selected_point_x = shape[i][0]
                                    gray_color = frame[selected_point_y, selected_point_x]
                                    # 在adaptive_length之後才開始檢查數值是否在一個合理的範圍
                                    if n_frame > self.adaptive_length:
                                        if (np.abs(gray_color-average_list[i])/average_list[i]) < 0.5:
                                            gray_sub_list.append(gray_color)
                                            average_list[i] = (average_list[i]*(n_frame-1)+gray_color)/n_frame  # 更新平均值
                                        else:
                                            if gray_color < self.minimum_gray_color:
                                                # 小於最低值的,捨棄改用minimum_gray_color代替
                                                gray_sub_list.append(average_list[i])
                                                average_list[i] = average_list[i]  # 這句非必要只是容易看
                                            else:
                                                gray_sub_list.append(average_list[i])  # 變動超過範圍的,回傳平均值
                                                average_list[i] = (average_list[i]*(n_frame-1)+gray_color)/n_frame  # 更新平均值
                                    else:
                                        if gray_color < self.minimum_gray_color:
                                            # 小於最低值的,捨棄改用minimum_gray_color代替
                                            gray_sub_list.append(self.minimum_gray_color)
                                            average_list[i] = average_list[i]  # 這句非必要只是容易看
                                        else:
                                            gray_sub_list.append(gray_color)
                                            average_list[i] = (average_list[i]*(n_frame-1)+gray_color)/n_frame  # 更新平均值

                                previous_gray_sub_list = gray_sub_list
                                gray_list.append(gray_sub_list)
                                gray_list_for_chart.append(gray_sub_list)

                                # loop over the (x, y)-coordinates from our dlib shape
                                # predictor model draw them on the image
                                for (sx, sy) in shape:
                                    cv2.circle(frame_copy, (sx, sy), 2, (255, 0, 0), -1)

                                # 加點
                                cv2.circle(frame_copy, (point_left_cheek_x, point_left_cheek_y), 1, (0, 0, 255), -1)
                                cv2.circle(frame_copy, (point_left_nose_x, point_left_nose_y), 1, (0, 0, 255), -1)
                                cv2.circle(frame_copy, (point_right_cheek_x, point_right_cheek_y), 1, (0, 0, 255), -1)
                                cv2.circle(frame_copy, (point_right_nose_x, point_right_nose_y), 1, (0, 0, 255), -1)
                                cv2.circle(frame_copy, (point_fore_head_x, point_fore_head_y), 1, (0, 0, 255), -1)
                                cv2.circle(frame_copy, (point_chin_x, point_chin_y), 1, (0, 0, 255), -1)
                                cv2.circle(frame_copy, (point_throat_x, point_throat_y), 1, (0, 0, 255), -1)
                            except Exception as inst:
                                n_error = n_error + 1
                                print("error:")
                                print(n_error)
                                print(inst)
                                gray_list_for_chart.append(previous_gray_sub_list)  # 補足frame以畫圖
                        else:
                            print("h out of boundary or y changes")
                            print("upper:" + str(self.height_upper_threshold) + ";h=" + str(h) + ";lower=" + str(self.height_lower_threshold))
                            print("y=" + str(y) + ";y_average=" + str(y_average))
                            gray_list_for_chart.append(previous_gray_sub_list)  # 補足frame以畫圖
                else:
                    print("len(rects)==0")
                    testing_image_fullname = self.no_rectangle_image_path + str(n_grab) + ".png"
                    cv2.imwrite(testing_image_fullname, frame)
                    gray_list_for_chart.append(previous_gray_sub_list)  # 補足frame以畫圖

                # if the frame dimensions are empty, grab them
                if W is None or H is None:
                    (H, W) = frame.shape[:2]

                # check if the video writer is None
                if writer is None:
                    # initialize our video writer
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(output_file_fullname, fourcc, 28, (frame.shape[1], frame.shape[0]), True)

                # push the frame to the writer
                writer.write(frame_copy)

            if len(gray_list_for_chart) > 0:
                print(file_name)
                # print(gray_list_for_chart)
                plt.rcParams["figure.figsize"] = (15, 12)
                plt.title(file_name)
                plt.ylim([0, 255])
                rolling_n = 10  # default=10
                import pandas as pd
                row_21 = pd.DataFrame([row[21] for row in gray_list_for_chart]).rolling(rolling_n).mean()[rolling_n:]
                row_22 = pd.DataFrame([row[22] for row in gray_list_for_chart]).rolling(rolling_n).mean()[rolling_n:]
                row_58 = pd.DataFrame([row[58] for row in gray_list_for_chart]).rolling(rolling_n).mean()[rolling_n:]
                row_29 = pd.DataFrame([row[29] for row in gray_list_for_chart]).rolling(rolling_n).mean()[rolling_n:]
                row_30 = pd.DataFrame([row[30] for row in gray_list_for_chart]).rolling(rolling_n).mean()[rolling_n:]
                row_54 = pd.DataFrame([row[54] for row in gray_list_for_chart]).rolling(rolling_n).mean()[rolling_n:]
                row_55 = pd.DataFrame([row[55] for row in gray_list_for_chart]).rolling(rolling_n).mean()[rolling_n:]
                row_59 = pd.DataFrame([row[59] for row in gray_list_for_chart]).rolling(rolling_n).mean()[rolling_n:]
                row_60 = pd.DataFrame([row[60] for row in gray_list_for_chart]).rolling(rolling_n).mean()[rolling_n:]

                plt.plot(row_21, label="Left Eyebrow")
                plt.plot(row_22, label="Right Eyebrow")
                plt.plot(row_58, label="Fore Head")
                plt.plot(row_29, label="Next To Nose Tip")
                plt.plot(row_30, label="Nose Tip")
                plt.plot(row_54, label="Left Cheek")
                plt.plot(row_55, label="Left Nose")
                plt.plot(row_59, label="Chin")
                plt.plot(row_60, label="Throat")
                # print(row_21.to_string(), row_22, row_58, row_29, row_30,  row_54, row_55, row_59, row_60)
                # a = [row[21] for row in gray_list_for_chart]
                # print(a)
                # print(pd.DataFrame(a).rolling(10).mean()[10:])
                # plt.legend(loc="upper left")
                plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
                # plt.title('test')
                plt.tight_layout()
                output_image_fullname = os.path.join(os.getcwd(), self.visualization_path + file_name + ".png")
                plt.savefig(output_image_fullname)
                fig1 = plt.figure(file_name)
                # plt.show()
                print(os.path.join(os.getcwd(), output_csv_file_fullname))
                np.savetxt(os.path.join(os.getcwd(), output_csv_file_fullname), gray_list, delimiter=",")

        # do a bit cleanup
        cv2.destroyAllWindows()
        vs.release()
        writer.release()
        print('finished')