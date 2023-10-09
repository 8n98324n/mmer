import cv2
import numpy as np
import mediapipe as mp
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error as mae
"""
This module contains classes and methods for transforming a BVP signal in a BPM signal.
"""

class Comparison:

    """
    Manage (multi-channel, row-wise) BVP signals, and transforms them in BPMs.
    """

    def __init__(self):
        pass

    def compare_two_data_frame(self, df_1, df_2, column_names):
        df_new_1 = pd.DataFrame()
        df_new_2 = pd.DataFrame()
        for index in range(len(df_1["ID"])):
            row = df_2.loc[df_2['ID'] == df_1["ID"][index]]
            if len(row)>0:
                df_new_1 = pd.concat([df_new_1,df_1.loc[index]],axis=1,ignore_index=True)
                df_new_2 = pd.concat([df_new_2,row.iloc[0]],axis=1,ignore_index=True)  # 如果存在两条及以上的切片，则取前2分钟的值
        df_new_1 = df_new_1.transpose()  # 转置
        df_new_2 = df_new_2.transpose()

        import math
        mininum_threshold = 0
        maximum_threshold = 10000
        
        def isfloat(num):
            try:
                float(num)
                if math.isnan(num):
                    return False
                return True
            except ValueError:
                return False
            
        def is_valid(num):
            if mininum_threshold < num < maximum_threshold:
                return True
            else:
                return False
            
        for index in range(len(column_names)):
            selected_column_name = column_names[index]
            seris_1 = df_new_1[selected_column_name]
            seris_2 = df_new_2[selected_column_name]
            df_combined = pd.concat([seris_1,seris_2],axis=1)
            df_combined.columns = ['ECG','r-PPG']
            df_filetered = df_combined[df_combined.iloc[:, 0].apply(lambda x: isfloat(x))]
            df_filetered = df_filetered[df_filetered.iloc[:, 1].apply(lambda x: isfloat(x))]
            df_filetered = df_filetered[df_filetered.iloc[:, 0].apply(lambda x: is_valid(x))]
            df_filetered = df_filetered[df_filetered.iloc[:, 1].apply(lambda x: is_valid(x))]
            fig = px.scatter(df_filetered, x="ECG", y="r-PPG", trendline="ols", width=600, height=400)
            annotation_x = max(df_filetered.iloc[:,0]) - 0.2*(max(df_filetered.iloc[:,0])-min(df_filetered.iloc[:,0]))

            r = np.corrcoef(seris_1.to_numpy().astype(float), seris_2.to_numpy().astype(float))[1,0]
            r= float("{:.2f}".format(r))
            # calculate MAE
            mae_result = mae(seris_1.to_numpy().astype(float), seris_2.to_numpy().astype(float))
            mae_result= float("{:.2f}".format(mae_result))
            fig.add_annotation(text="r=" + str(r) + ",MAE="+ str(mae_result)+ ",n="+ str(len(seris_1.to_numpy().astype(float))), showarrow=False,x=annotation_x, y=min(df_filetered.iloc[:,1]))
            fig.update_layout(
                title_text=selected_column_name
            )
            fig.show()


    def compare_two_data_source(self, input_file_1, input_file_2, column_names):
        df_1 = pd.read_csv(input_file_1)
        df_2 = pd.read_csv(input_file_2)
        self.compare_two_data_frame(df_1, df_2, column_names)
        # df_new_1 = pd.DataFrame()
        # df_new_2 = pd.DataFrame()
        # for index in range(len(df_1["ID"])):
        #     row = df_2.loc[df_2['ID'] == df_1["ID"][index]]
        #     if len(row)>0:
        #         df_new_1 = pd.concat([df_new_1,df_1.loc[index]],axis=1,ignore_index=True)
        #         df_new_2 = pd.concat([df_new_2,row.iloc[0]],axis=1,ignore_index=True)  # 如果存在两条及以上的切片，则取前2分钟的值
        # df_new_1 = df_new_1.transpose()  # 转置
        # df_new_2 = df_new_2.transpose()

        # import math
        # mininum_threshold = 0
        # maximum_threshold = 10000
        
        # def isfloat(num):
        #     try:
        #         float(num)
        #         if math.isnan(num):
        #             return False
        #         return True
        #     except ValueError:
        #         return False
            
        # def is_valid(num):
        #     if mininum_threshold < num < maximum_threshold:
        #         return True
        #     else:
        #         return False
            
        # for index in range(len(column_names)):
        #     selected_column_name = column_names[index]
        #     seris_1 = df_new_1[selected_column_name]
        #     seris_2 = df_new_2[selected_column_name]
        #     df_combined = pd.concat([seris_1,seris_2],axis=1)
        #     df_combined.columns = ['ECG','r-PPG']
        #     df_filetered = df_combined[df_combined.iloc[:, 0].apply(lambda x: isfloat(x))]
        #     df_filetered = df_filetered[df_filetered.iloc[:, 1].apply(lambda x: isfloat(x))]
        #     df_filetered = df_filetered[df_filetered.iloc[:, 0].apply(lambda x: is_valid(x))]
        #     df_filetered = df_filetered[df_filetered.iloc[:, 1].apply(lambda x: is_valid(x))]
        #     fig = px.scatter(df_filetered, x="ECG", y="r-PPG", trendline="ols", width=600, height=400)
        #     annotation_x = max(df_filetered.iloc[:,0]) - 0.2*(max(df_filetered.iloc[:,0])-min(df_filetered.iloc[:,0]))
        #     fig.add_annotation(text="Text annotation without arrow", showarrow=False,x=annotation_x, y=min(df_filetered.iloc[:,1]))
        #     fig.update_layout(
        #         title_text=selected_column_name
        #     )
        #     fig.show()

class Feature:
    """
    Manage (multi-channel, row-wise) BVP signals, and transforms them in BPMs.
    """

    def __init__(self):
        pass


    def get_face_with_feature_points(self, file_of_face, landmarks_list):
        #    #檢查landmark點用
        PRESENCE_THRESHOLD = 0.5
        VISIBILITY_THRESHOLD = 0.5

        imag = cv2.imread(file_of_face, cv2.COLOR_RGB2BGR)
        imag = cv2.cvtColor(imag, cv2.COLOR_BGR2RGB)
        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                min_detection_confidence=0.5) as face_mesh:
            image = cv2.imread(file_of_face)
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            width = image.shape[1]
            height = image.shape[0]
            face_landmarks = results.multi_face_landmarks[0]
            ldmks = np.zeros((468, 3), dtype=np.float32)
            for idx, landmark in enumerate(face_landmarks.landmark):
                if ((landmark.HasField('visibility') and landmark.visibility < VISIBILITY_THRESHOLD)
                        or (landmark.HasField('presence') and landmark.presence < PRESENCE_THRESHOLD)):
                    ldmks[idx, 0] = -1.0
                    ldmks[idx, 1] = -1.0
                    ldmks[idx, 2] = -1.0
                else:
                    coords = mp_drawing._normalized_to_pixel_coordinates(
                        landmark.x, landmark.y, width, height)
                    if coords:
                        ldmks[idx, 0] = coords[0]
                        ldmks[idx, 1] = coords[1]
                        ldmks[idx, 2] = idx
                    else:
                        ldmks[idx, 0] = -1.0
                        ldmks[idx, 1] = -1.0
                        ldmks[idx, 2] = -1.0

        filtered_ldmks = []
        if landmarks_list is not None:
            for idx in landmarks_list:
                filtered_ldmks.append(ldmks[idx])
            filtered_ldmks = np.array(filtered_ldmks, dtype=np.float32)
        else:
            filtered_ldmks = ldmks    
        image = cv2.imread(file_of_face, cv2.COLOR_RGB2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fig = px.imshow(image)
        for l in filtered_ldmks:
            name = '特征_' + str(int(l[2]))
            fig.add_trace(go.Scatter(x=(l[0],), y=(l[1],), name=name, mode='markers', 
                                    marker=dict(color='red', size=5)))
        fig.update_xaxes(range=[0,image.shape[1]])
        fig.update_yaxes(range=[image.shape[0],0])
        fig.update_layout(paper_bgcolor='#eee') 
        return fig

    


