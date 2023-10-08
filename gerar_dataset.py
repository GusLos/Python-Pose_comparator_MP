from MPPose import MPPose
from PoseComparator import PoseComparator
from dotenv import dotenv_values
import numpy as np
import cv2
import os
import csv

if __name__ == '__main__':

    dotenv = dotenv_values(".env")
    model_path  = dotenv['MODEL_PATH_FULL']
    image_folder = dotenv['IMAGE_DATASET_FOLDER']

    mppose = MPPose(model_path, 'image')

    with open('eggs.csv', 'w', newline='') as csvfile:

        spamwriter = csv.writer(csvfile, delimiter=';',quotechar=';', quoting=csv.QUOTE_MINIMAL)
        # spamwriter.writerow(['right_arm_angle'] + ['right_arm_director'] + ['right_forearm_angle'] + ['right_forearm_director'] + ['left_arm_angle'] + ['left_arm_director'] + ['left_forearm_angle'] + ['left_forearm_director'] + ['shoulders'])
        spamwriter.writerow(['right_arm_angle'] + 
                            ['right_arm_director_i'] + ['right_arm_director_j'] + ['right_arm_director_k'] + 
                            ['right_forearm_angle'] + 
                            ['right_forearm_director_i'] + ['right_forearm_director_j'] + ['right_forearm_director_k'] + 
                            ['left_arm_angle'] + 
                            ['left_arm_director_i'] + ['left_arm_director_j'] + ['left_arm_director_k'] +  
                            ['left_forearm_angle'] + 
                            ['left_forearm_director_i'] + ['left_forearm_director_j'] + ['left_forearm_director_k'] + 
                            ['shoulders_i'] + ['shoulders_j'] + ['shoulders_k'] +
                            ['neck_i'] + ['neck_j'] + ['neck_k']
                            )

        for file in os.listdir(image_folder):
            filename = os.fsdecode(file)
            # print(image_folder+ '\\' +file)
            image_path = image_folder+ '\\' +file
            image = mppose.detect_pose(image_path=image_path)
            # print(image[1].pose_world_landmarks[0][0].x) # visibility // presence
            image_result = PoseComparator.analyse_torso_wo_affine(image[1].pose_world_landmarks)
            # print(image_result['shoulders'])

            right_arm_angle        = image_result['upper_limbs']['right']['arm']['angle']
            right_arm_director_i   = image_result['upper_limbs']['right']['arm']['director'][0]
            right_arm_director_j   = image_result['upper_limbs']['right']['arm']['director'][1]
            right_arm_director_k   = image_result['upper_limbs']['right']['arm']['director'][2]
            right_forearm_angle    = image_result['upper_limbs']['right']['forearm']['angle']
            right_forearm_director_i   = image_result['upper_limbs']['right']['forearm']['director'][0]
            right_forearm_director_j   = image_result['upper_limbs']['right']['forearm']['director'][1]
            right_forearm_director_k   = image_result['upper_limbs']['right']['forearm']['director'][2]

            left_arm_angle        = image_result['upper_limbs']['left']['arm']['angle'] 
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

            neck_i = image_result['neck'][0]
            neck_j = image_result['neck'][1]
            neck_k = image_result['neck'][2]

            # spamwriter.writerow([right_arm_angle, right_arm_director, right_forearm_angle, right_forearm_director, left_arm_angle, left_arm_director, left_forearm_angle, left_forearm_director, shoulders])
            spamwriter.writerow([right_arm_angle, 
                                 right_arm_director_i, right_arm_director_j, right_arm_director_k, 
                                 right_forearm_angle, 
                                 right_forearm_director_i, right_forearm_director_j, right_forearm_director_k,
                                 left_arm_angle, 
                                 left_arm_director_i, left_arm_director_j, left_arm_director_k,  
                                 left_forearm_angle, 
                                 left_forearm_director_i, left_forearm_director_j, left_forearm_director_k, 
                                 shoulders_i, shoulders_j, shoulders_k,
                                 neck_i, neck_j, neck_k,
                                 ])

            # break
            pass # for

    # ----------------

    # with open('eggs.csv', 'w', newline='') as csvfile:
        # spamwriter = csv.writer(csvfile, delimiter=';',
                            # quotechar=';', quoting=csv.QUOTE_MINIMAL)
        # spamwriter.writerow(['right_arm_angle'] + ['right_arm_director'] + ['right_forearm_angle'] + ['right_forearm_director'] + ['left_arm_angle'] + ['left_arm_director'] + ['left_forearm_angle'] + ['left_forearm_director'] + ['shoulders'])
        # spamwriter.writerow([118, np.array([100, 98, 166]), 109, np.array([119, 45, 121])])


    pass