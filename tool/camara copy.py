import cv2
import mediapipe as mp
import copy
import numpy as np
import time
import os

# 關節點對應表
joint = {"head": 0, "neck": 1, "rshoulder": 2, "rarm": 3, "rhand": 4,
         "lshoulder": 5, "larm": 6, "lhand": 7, "pelvis": 8,
         "rthing": 9, "rknee": 10, "rankle": 11, "lthing": 12, "lknee": 13, "lankle": 14}
         
# mmpose 3D關節點
# 0: 'root (pelvis)',
# 1: 'right_hip',
# 2: 'right_knee',
# 3: 'right_foot',
# 4: 'left_hip',
# 5: 'left_knee',
# 6: 'left_foot',
# 7: 'spine',
# 8: 'thorax',
# 9: 'neck_base',
# 10: 'head',
# 11: 'left_shoulder',
# 12: 'left_elbow',
# 13: 'left_wrist',
# 14: 'right_shoulder',
# 15: 'right_elbow',
# 16: 'right_wrist'

def add(a, b):  # 取兩關節點的中點
    tmp = copy.deepcopy(a)
    tmp.x = (a.x+b.x)/2
    tmp.y = (a.y+b.y)/2
    tmp.z = (a.z+b.z)/2
    tmp.visibility = (a.visibility+b.visibility)/2
    return tmp


def draw_line(img, x, y):  # 畫出關節點之間的線
    # head to neck
    cv2.line(img, (x[0], y[0]), (x[1], y[1]), (0, 0, 255), 2)
    # neck to rshoulder
    cv2.line(img, (x[1], y[1]), (x[2], y[2]), (0, 0, 255), 2)
    # rshoulder to rarm
    cv2.line(img, (x[2], y[2]), (x[3], y[3]), (0, 0, 255), 2)
    # rarm to rhand
    # cv2.line(img, (x[3], y[3]), (x[4], y[4]), (0, 0, 255), 2)
    # print("rarm to rhand", x[3], y[3], x[4], y[4])
    # print("left arm = ", x[3], y[3])
    # print("left hand1 = ", x[4], y[4])
    # neck to lshoulder
    cv2.line(img, (x[1], y[1]), (x[5], y[5]), (255, 0, 0), 2)
    # lshoulder to larm
    cv2.line(img, (x[5], y[5]), (x[6], y[6]), (255, 0, 0), 2)
    # larm to lhand
    # cv2.line(img, (x[6], y[6]), (x[7], y[7]), (255, 0, 0), 2)

    # neck to pelvis
    cv2.line(img, (x[1], y[1]), (x[8], y[8]), (0, 0, 255), 2)

    # pelvis to rthing
    cv2.line(img, (x[8], y[8]), (x[9], y[9]), (0, 0, 255), 2)
    # rthing to rknee
    cv2.line(img, (x[9], y[9]), (x[10], y[10]), (0, 0, 255), 2)
    # rknee to rankle
    cv2.line(img, (x[10], y[10]), (x[11], y[11]), (0, 0, 255), 2)

    # pelvis to lthing
    cv2.line(img, (x[8], y[8]), (x[12], y[12]), (255, 0, 0), 2)
    # lthing to lknee
    cv2.line(img, (x[12], y[12]), (x[13], y[13]), (255, 0, 0), 2)
    # lknee to lankle
    cv2.line(img, (x[13], y[13]), (x[14], y[14]), (255, 0, 0), 2)


def mediapipe_point_to_lab(p):
    # 0, (11+12)/2, 11, 13, (17+19)/2, 12, 14, (18+20)/2, (23+24)/2, 23, 25, 27, 24, 26, 28
    point_list = []
    point_list.append(p[0])                 # head
    point_list.append(add(p[11], p[12]))    # neck
    point_list.append(p[12])                # rshoulder
    point_list.append(p[14])                # rarm
    # point_list.append(add(p[18], p[20]))    # rhand
    point_list.append(p[16])  # rhand
    point_list.append(p[11])                # lshoulder
    point_list.append(p[13])                # larm
    # point_list.append(add(p[17], p[19]))    # lhand
    point_list.append(p[15])  # lhand
    point_list.append(add(p[23], p[24]))    # pelvis
    point_list.append(p[24])                # rthing
    point_list.append(p[26])                # rknee
    point_list.append(p[28])                # rankle
    point_list.append(p[23])                # lthing
    point_list.append(p[25])                # lknee
    point_list.append(p[27])                # lankle
    return point_list


if __name__ == '__main__':
    mp_drawing = mp.solutions.drawing_utils  # mediapipe 繪圖方法
    mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
    mp_holistic = mp.solutions.holistic  # mediapipe 全身偵測方法

    filepath = '/home/peter/3d_motion_generator/data/video/lhand_dribble/test.mp4'
    save_dir = '/home/peter/3d_motion_generator/data/npy/'
    if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
    cap = cv2.VideoCapture(filepath)
    pTime = 0  # 設置第一幀開始處理的起始時間
    frame_cnt = 0  # 計算幀數
    file_idx = 0
    cut_flag = 0
    last_cut_frame = 0
    first_save = 0
    # cap = cv2.VideoCapture(0)  # 用前鏡頭拍自己
    line_x = []
    line_y = []
    ouput = np.zeros((45))
    output = np.empty(45)
    ouput_list = []
    with mp_holistic.Holistic(
        static_image_mode=False,  # 靜態圖模式，False代表置信度高時繼續跟蹤，True代表實時跟蹤檢測新的結果
        smooth_landmarks=True,  # 平滑，一般為True
        min_detection_confidence=0.9,  # 檢測置信度
        min_tracking_confidence=0.9  # 跟蹤置信度
    ) as holistic:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frame_cnt = frame_cnt + 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = holistic.process(frame)
            # print(results.left_hand_landmarks)
            # print("--------------")
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 畫面部關節點
            # mp_drawing.draw_landmarks(
            #     frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()  # 幫臉上顏色
            # )
            # 畫左手關節點
            # mp_drawing.draw_landmarks(
            #     frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            #     mp_drawing_styles.get_default_hand_landmarks_style(),  # 幫手指關節點上色
            #     mp_drawing_styles.get_default_hand_connections_style()  # 幫手指關節點之間的線上色
            # )
            # 畫右手關節點
            # cv2.circle(frame, (results.left_hand_landmarks.landmark[0].x,
            #            results.left_hand_landmarks.landmark[0].y), 5, (255, 0, 0), cv2.FILLED)
            # mp_drawing.draw_landmarks(
            #     frame, results.right_hand_landmarks,
            #     mp_holistic.HAND_CONNECTIONS,
            #     mp_drawing_styles.get_default_hand_landmarks_style(),  # 幫手指關節點上色
            #     mp_drawing_styles.get_default_hand_connections_style()  # 幫手指關節點之間的連線上色
            # )
            # 畫身體關節點 mediapipe原本的
            # mp_drawing.draw_landmarks(
            #     frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.pose_landmarks:
                # point_list = mediapipe_point_to_lab(
                #     results.pose_landmarks.landmark, results.right_hand_landmarks.landmark[0], results.left_hand_landmarks.landmark[0])
                point_list = mediapipe_point_to_lab(
                    results.pose_landmarks.landmark)
                for index, lm in enumerate(point_list):
                    # print("index = ", index)
                    # 保存每幀圖像的寬、高、通道數
                    h, w, c = frame.shape

                    # 得到的關鍵點坐標x/y/z/visibility都是比例坐標，在[0,1]之間
                    # 轉換為像素坐標(cx,cy)，圖像的實際長寬乘以比例，像素坐標一定是整數
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    # 印出坐標信息
                    # print(index, cx, cy, lm.visibility)
                    # print(index, lm.x, lm.y, lm.z)

                    # 在關鍵點上畫圓圈，img畫板，以(cx,cy)為圓心，半徑3，顏色紅色，填充圓圈
                    if (index == 4 or index == 7):
                        pass
                    elif (index == 5 or index == 6 or index == 12 or index == 13 or index == 14):
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                    else:
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

                    # 保存坐標信息
                    line_x.append(cx)
                    line_y.append(cy)
                    # print("lm_x = ", lm.x)
                    ouput[index*3+0] = lm.x
                    ouput[index*3+1] = lm.y
                    ouput[index*3+2] = lm.z
                ouput_list.append(ouput.tolist())
                # print(np.array(ouput_list).shape)
                # 畫出關節點之間的線
                # print("line_x = ", line_x)
                draw_line(frame, line_x, line_y)
                line_x.clear()
                line_y.clear()
                # print("left hand1 = ", int(
                #     results.pose_landmarks.landmark[15].x * w), int(results.pose_landmarks.landmark[15].y * h))

            # 畫左臂到左手的線
            if results.left_hand_landmarks:
                # print("left arm = ", int(
                #     ouput[6][0] * w), int(ouput[6][1] * h))
                # print("left hand1 = ",
                #       int(point_list[6].x * w), int(point_list[6].y * h))
                # print("left hand2 = ", int(results.left_hand_landmarks.landmark[0].x * w),
                #       int(results.left_hand_landmarks.landmark[0].y * h))
                cv2.line(frame, (int(results.left_hand_landmarks.landmark[0].x * w), int(results.left_hand_landmarks.landmark[0].y * h)), (
                    int(point_list[6].x * w), int(point_list[6].y * h)), (255, 0, 0), 2)

            # 畫右臂到右手的線
            if results.right_hand_landmarks:
                print("ok = ", frame_cnt)
                # print(frame_cnt, last_cut_frame)
                if (frame_cnt - last_cut_frame >= 10):
                    # print("ready to cut")
                    cut_flag = 1
                # print("right hand1 = ",
                #       int(point_list[3].x * w), int(point_list[3].y * h))
                # print("right hand2 = ", int(results.right_hand_landmarks.landmark[0].x * w),
                #       int(results.right_hand_landmarks.landmark[0].y * h))
                # cv2.line(frame, (int(results.right_hand_landmarks.landmark[0].x * w), int(results.right_hand_landmarks.landmark[0].y * h)), (
                #     int(point_list[3].x * w), int(point_list[3].y * h)), (0, 0, 255), 2)
                # print("ok-----------------")
                # print(results.right_hand_landmarks.landmark[0])
            if (not results.right_hand_landmarks and cut_flag == 1):
                # print(np.array(ouput_list).shape)
                with open(save_dir + str(file_idx) + '.npy', 'wb') as f:
                    np.save(f, ouput_list)
                output = np.vstack([output,ouput_list])
                # print(output)
                file_idx += 1
                cut_flag = 0
                # print(np.array(output).shape)
                # print("last frame = ",last_cut_frame)
                # print("cut len = ",frame_cnt - last_cut_frame)
                print("cut frame = ",frame_cnt)
                print("--------------------------------")
                last_cut_frame = frame_cnt
                ouput_list.clear()
            elif (not results.right_hand_landmarks and cut_flag == 0):
                last_cut_frame = frame_cnt
                print("no hand = ",frame_cnt)
                ouput_list.clear()

            # cTime = time.time()  # 處理完一幀圖像的時間
            # fps = 1/(cTime-pTime)
            # pTime = cTime  # 重置起始時間
            # length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # # print("len = ", length)
            # # print("frame = ", frame_cnt)
            # # 在視頻上顯示fps信息，先轉換成整數再變成字符串形式，文本顯示坐標，文本字體，文本大小
            # cv2.putText(frame, str(int(frame_cnt)), (70, 50),
            #             cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
            # # 因為攝像頭是鏡像的，所以將攝像頭水平翻轉
            # # frame = cv2.flip(frame, 1)
            # cv2.namedWindow("MediaPipe", 0)
            # cv2.resizeWindow("MediaPipe", 1000, 750)
            # cv2.imshow('MediaPipe', frame)
            # if cv2.waitKey(1) & 0xFF == 27:
            #     break

    # with open(save_dir + 'all_3.npy', 'wb') as f:
    #     np.save(f, ouput)

    cap.release()
    cv2.destroyAllWindows()
