from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from PoseComparator import PoseComparator
from dotenv import dotenv_values
from MPPose import MPPose
import numpy as np
import cv2
import os
import csv

if __name__ == '__main__':

    dotenv = dotenv_values(".env")
    model_path = dotenv['MODEL_PATH_HEAVY']

    
    image_folder_1 = dotenv['IMAGE_DATASET_FOLDER_1']
    # image_folder_1 = dotenv['IMAGENS_IA']

    mppose = MPPose(model_path, 'image')

    # for image_folder in [image_folder_1, image_folder_2]:
    for image_folder in [image_folder_1]:

        with open('pose.csv', 'w', newline='') as csvfile:

            spamwriter = csv.writer(csvfile, delimiter=';',quotechar=';', quoting=csv.QUOTE_MINIMAL)

            spamwriter.writerow(
                # ['nome_pose'] +
                ['right_arm_angle'] + ['right_arm_direction'] +
                ['right_arm_director_i'] + ['right_arm_director_j'] + ['right_arm_director_k'] + 
                ['right_forearm_angle'] + 
                ['right_forearm_director_i'] + ['right_forearm_director_j'] + ['right_forearm_director_k'] + 
                ['left_arm_angle'] + ['left_arm_direction'] +
                ['left_arm_director_i'] + ['left_arm_director_j'] + ['left_arm_director_k'] +  
                ['left_forearm_angle'] + 
                ['left_forearm_director_i'] + ['left_forearm_director_j'] + ['left_forearm_director_k'] + 
                ['shoulders_i'] + ['shoulders_j'] + ['shoulders_k'] +
                ['neck_i'] + ['neck_j'] + ['neck_k']
            )

            for file in os.listdir(image_folder):
                filename = os.fsdecode(file)
                print(image_folder+ '\\' +file)
                image_path = image_folder+ '\\' +file
                image = mppose.detect_pose(image_path=image_path)
                # input(image[1].pose_world_landmarks)
                image_result = PoseComparator.analyse_torso_wo_affine(image[1].pose_world_landmarks)

# ---------------------------------------------------------
                # image[1].pose_world_landmarks[0].append( NormalizedLandmark(
                #     x = (image[1].pose_world_landmarks[0][11].x +  image[1].pose_world_landmarks[0][12].x) / 2,
                #     y = (image[1].pose_world_landmarks[0][11].y +  image[1].pose_world_landmarks[0][12].y) / 2,
                #     z = (image[1].pose_world_landmarks[0][11].z +  image[1].pose_world_landmarks[0][12].z) / 2,
                #     visibility = (image[1].pose_world_landmarks[0][11].visibility +  image[1].pose_world_landmarks[0][12].visibility) / 2,
                #     presence   = (image[1].pose_world_landmarks[0][11].presence   +  image[1].pose_world_landmarks[0][12].presence) / 2
                # ))
# ---------------------------------------------------------

                # imagem_result = mppose.draw_landmarks_on_image(image[0], image[1])
                # imagem_result = cv2.resize(imagem_result, None, fx=0.2, fy=0.2, interpolation = cv2.INTER_CUBIC)
                # cv2.imshow(file, cv2.cvtColor(imagem_result, cv2.COLOR_RGB2BGR))
                # cv2.waitKey(0)
                # nome_pose = input('Pose: ')
                # cv2.destroyAllWindows()

                right_arm_angle        = image_result['upper_limbs']['right']['arm']['angle']
                right_arm_direction    = PoseComparator.is_up(
                    [image[1].pose_world_landmarks[0][12].x , image[1].pose_world_landmarks[0][12].y , image[1].pose_world_landmarks[0][12].z], 
                    [image[1].pose_world_landmarks[0][14].x , image[1].pose_world_landmarks[0][14].y , image[1].pose_world_landmarks[0][14].z], 
                    [image[1].pose_world_landmarks[0][16].x , image[1].pose_world_landmarks[0][16].y , image[1].pose_world_landmarks[0][16].z]
                )[1]
                right_arm_director_i   = image_result['upper_limbs']['right']['arm']['director'][0]
                right_arm_director_j   = image_result['upper_limbs']['right']['arm']['director'][1]
                right_arm_director_k   = image_result['upper_limbs']['right']['arm']['director'][2]
                right_forearm_angle    = image_result['upper_limbs']['right']['forearm']['angle']
                right_forearm_director_i   = image_result['upper_limbs']['right']['forearm']['director'][0]
                right_forearm_director_j   = image_result['upper_limbs']['right']['forearm']['director'][1]
                right_forearm_director_k   = image_result['upper_limbs']['right']['forearm']['director'][2]

                left_arm_angle        = image_result['upper_limbs']['left']['arm']['angle'] 
                left_arm_direction    = PoseComparator.is_up(
                    [image[1].pose_world_landmarks[0][11].x , image[1].pose_world_landmarks[0][11].y , image[1].pose_world_landmarks[0][11].z], 
                    [image[1].pose_world_landmarks[0][13].x , image[1].pose_world_landmarks[0][13].y , image[1].pose_world_landmarks[0][13].z], 
                    [image[1].pose_world_landmarks[0][15].x , image[1].pose_world_landmarks[0][15].y , image[1].pose_world_landmarks[0][15].z]
                )[1]
                left_arm_director_i   = image_result['upper_limbs']['left']['arm']['director'][0]
                left_arm_director_j   = image_result['upper_limbs']['left']['arm']['director'][1]
                left_arm_director_k   = image_result['upper_limbs']['left']['arm']['director'][2]
                left_forearm_angle    = image_result['upper_limbs']['left']['forearm']['angle'] 
                left_forearm_director_i   = image_result['upper_limbs']['left']['forearm']['director'][0]
                left_forearm_director_j   = image_result['upper_limbs']['left']['forearm']['director'][1]
                left_forearm_director_k   = image_result['upper_limbs']['left']['forearm']['director'][2]

                shoulders_i = image_result['shoulders'][0]
                shoulders_j = image_result['shoulders'][1]
                shoulders_k = image_result['shoulders'][2]
                # shoulders_x_diference = abs(image[1].pose_world_landmarks[0][11].x - image[1].pose_world_landmarks[0][12].x)
                # shoulders_y_diference = abs(image[1].pose_world_landmarks[0][11].y - image[1].pose_world_landmarks[0][12].y)

                neck_i = image_result['neck'][0]
                neck_j = image_result['neck'][1]
                neck_k = image_result['neck'][2]
                # neck_x_diference = abs(image[1].pose_world_landmarks[0][0].x - image[1].pose_world_landmarks[0][33].x)
                # neck_y_diference = abs(image[1].pose_world_landmarks[0][0].y - image[1].pose_world_landmarks[0][33].y)

                # spamwriter.writerow([right_arm_angle, right_arm_director, right_forearm_angle, right_forearm_director, left_arm_angle, left_arm_director, left_forearm_angle, left_forearm_director, shoulders])
                spamwriter.writerow(
                    [
                        # nome_pose,
                        right_arm_angle, right_arm_direction,
                        right_arm_director_i, right_arm_director_j, right_arm_director_k, 
                        right_forearm_angle, 
                        right_forearm_director_i, right_forearm_director_j, right_forearm_director_k,
                        left_arm_angle, left_arm_direction,
                        left_arm_director_i, left_arm_director_j, left_arm_director_k,  
                        left_forearm_angle, 
                        left_forearm_director_i, left_forearm_director_j, left_forearm_director_k, 
                        shoulders_i, shoulders_j, shoulders_k,
                        # shoulders_x_diference, shoulders_y_diference,
                        neck_i, neck_j, neck_k,
                        # neck_x_diference, neck_y_diference
                    ]
                )
                # input('---')
                # break
                pass # for
        pass
    pass