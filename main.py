from PoseComparator import PoseComparator as PC
from MPPose import MPPose
from dotenv import dotenv_values
import numpy as np
import cv2
import os
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

global pose_now_smooth
global standard_pose

def update_array(target_array: np.array, number: int or float) -> np.array:
    target_array = np.delete(target_array, 0)
    # target_array = np.append(target_array, number).astype(int)
    target_array = np.append(target_array, number)
    return target_array
    pass

def update_pose_smooth_normalized(result_array=None, result=None , n=10):
    global pose_now_smooth
    if result:
        for landmark_id in pose_now_smooth.keys(): 
            pose_now_smooth[landmark_id].x = update_array(pose_now_smooth[landmark_id].x, result.pose_world_landmarks[0][landmark_id].x)
            pose_now_smooth[landmark_id].y = update_array(pose_now_smooth[landmark_id].y, result.pose_world_landmarks[0][landmark_id].y)
            pose_now_smooth[landmark_id].z = update_array(pose_now_smooth[landmark_id].z, result.pose_world_landmarks[0][landmark_id].z)
            pose_now_smooth[landmark_id].presence = update_array(pose_now_smooth[landmark_id].presence, result.pose_world_landmarks[0][landmark_id].presence)
            pose_now_smooth[landmark_id].visibility = update_array(pose_now_smooth[landmark_id].visibility, result.pose_world_landmarks[0][landmark_id].visibility)
            pass # for
        return

    for landmark in range(len(pose_now_smooth)):
        pose_now_smooth[landmark] = ((pose_now_smooth[landmark]* (n-1)) + result_array[landmark]) / n
        pass
    return
    pass # update_pose_smooth_normalized

def coordinates(id):
    global pose_now_smooth
    x = pose_now_smooth[id].x.mean()
    y = pose_now_smooth[id].y.mean()
    z = pose_now_smooth[id].z.mean()
    # return NormalizedLandmark(x=x, y=y, z=z)
    return np.array([x, y, z])
    pass

def verificar_angulo(ang_padrao, ang_teste, margem_erro: int = 2):
    return (ang_teste <= ang_padrao + margem_erro) and (ang_teste >= ang_padrao - margem_erro)
    pass # verificar_angulo

def verificar_diretores(diretor_padrao, diretor_teste, margem_erro: int = 2):
    return all(diretor_teste <= diretor_padrao + margem_erro) and all(diretor_teste >= diretor_padrao - margem_erro)
    pass # verificar_diretores

def select_torso(array_result=None, list_result=None) -> np.array:
    '''
    'ombro_esquerdo': 11,   # 0
    'ombro_direito': 12,    # 1
    'cotovelo_esquerdo': 13,# 2
    'cotovelo_direito': 14, # 3
    'pulso_esquerdo': 15,   # 4
    'pulso_direito': 16,    # 5
    '''
    if list_result:
        list_torso = list_result.pose_world_landmarks[0][11: 17]
        torso_array = np.array(list_torso)
        return torso_array


    torso_array = array_result[11:17]
    return torso_array
    pass # select_torso

def _to_array() -> np.array:
    global pose_now_smooth
    list_result = []
    for landmark_id in pose_now_smooth.keys():
        # landmark = []
        # landmark.append(coordinates(landmark_id))
        list_result.append(coordinates(landmark_id))
        pass # for
    return np.array(list_result)
    pass

def live_stream_function(result, output_image, timestamp_ms):
    global pose_now_smooth
    global standard_pose
    os.system('cls')

    # result_array = PC.landmarks_result_to_array(result.pose_world_landmarks[0])
    update_pose_smooth_normalized(result=result)

    result_array = _to_array()
    standard_pose_torso = select_torso(array_result=standard_pose)

    result_transform, A_torso = PC.affine_transformation(standard_pose_torso, result_array)


    # Para um array com todas as landmarks
    # right_arm_angle     = PC.angle_between_limbs(result_transform[12], result_transform[14], result_transform[16])
    # right_forearm_angle = PC.angle_between_limbs(result_transform[14], result_transform[12], result_transform[11])
    
    # Para array q tem apenas o torso em ordem (11 até 16)
    right_arm_angle     = PC.angle_between_limbs(result_transform[1], result_transform[3], result_transform[5])
    right_forearm_angle = PC.angle_between_limbs(result_transform[3], result_transform[1], result_transform[0])

    # Para array que possui todas as landmarks
    # right_arm_director     = PC.analyse_limb(start_point_array = result_transform[14], final_point_array = result_transform[16])
    # right_forearm_director = PC.analyse_limb(start_point_array = result_transform[12], final_point_array = result_transform[14])

    # Para array q tem apenas o torso em ordem (11 até 16)
    right_arm_director     = PC.analyse_limb(start_point_array = result_transform[3], final_point_array = result_transform[5])
    right_forearm_director = PC.analyse_limb(start_point_array = result_transform[1], final_point_array = result_transform[3])

    # Para array que possui todas as landmarks
    # shoulders_director = PC.analyse_limb(start_point_array = standard_pose[11], final_point_array = standard_pose[12])

    # Para array q tem apenas o torso em ordem (11 até 16)
    shoulders_director = PC.analyse_limb(start_point_array = standard_pose[0], final_point_array = standard_pose[1])

    result_pose_jason = {
        'membros_superiores' : {
            'direito' : {
                'braco': {
                    'angulo':  right_arm_angle,
                    'diretor': right_arm_director,
                },
                'antebraco': {
                    'angulo':  right_forearm_angle,
                    'diretor': right_forearm_director,
                },
            },
            'esquerdo': {
                'braco': {
                    'angulo':  0,
                    'diretor': np.array([0, 0, 0]),
                },
                'antebraco': {
                    'angulo':  0,
                    'diretor': np.array([0, 0, 0]),
                },
            },
        },
        'ombros': shoulders_director,   # só diretor ? ou ang tambem com o eixo X?
        'pescoco': np.array([5, 5, 5]), # só diretor ? ou ang tambem com o eixo y?
    }

    model_arm_angle = model_pose_jason['membros_superiores']['direito']['braco']['angulo']

    print('input angle: ', right_arm_angle)
    print('model angle: ', model_arm_angle)

    ang_ok = verificar_angulo(model_arm_angle, right_arm_angle, 2)

    if ang_ok:
        print('\033[32m')
        print('Ang OK')
        pass
    else:
        print('\033[31m')
        print('Ang Nao OK')
        pass

    print('\033[37m')

    print(result_pose_jason)

    return
    pass # live_stream_function

if __name__ == '__main__':

        # Lendo váriaveis
    dotenv = dotenv_values(".env")
    model_path  = dotenv['MODEL_PATH_FULL']
    # model_path  = dotenv['MODEL_PATH_HEAVY']
    model_image = dotenv['MODEL_IMAGE']
    # Iniciando o modelo
    mppose = MPPose(model_path, 'image')

    # Analisar imagem ----------------------------------------------------------------------------------------------
    
    standard       = mppose.detect_pose(image_path=model_image)
    standard_image = mppose.draw_landmarks_on_image(standard[0], standard[1])
    standard_image = cv2.resize(standard_image,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

    cv2.imshow('Iagem', cv2.cvtColor(standard_image, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)

    standard_pose = PC.landmarks_result_to_array(standard[1].pose_world_landmarks[0])

    standard_right_arm_director     = PC.analyse_limb(start_point_array = standard_pose[14], final_point_array = standard_pose[16])
    standard_right_forearm_director = PC.analyse_limb(start_point_array = standard_pose[12], final_point_array = standard_pose[14])
    standard_shoulder_angle         = PC.analyse_limb(start_point_array = standard_pose[11], final_point_array = standard_pose[12])
    standard_right_arm_angle        = PC.angle_between_limbs(standard_pose[12], standard_pose[14], standard_pose[16])
    standard_right_forearm_angle    = PC.angle_between_limbs(standard_pose[14], standard_pose[12], standard_pose[11])

    model_pose_jason = {
        'membros_superiores' : {
            'direito' : {
                'braco': {
                    'angulo':  standard_right_arm_angle,
                    'diretor': standard_right_arm_director,
                },
                'antebraco': {
                    'angulo':  standard_right_forearm_angle,
                    'diretor': standard_right_forearm_director,
                },
            },
            'esquerdo': {
                'braco': {
                    'angulo':  0,
                    'diretor': np.array([0, 0, 0]),
                },
                'antebraco': {
                    'angulo':  0,
                    'diretor': np.array([0, 0, 0]),
                },
            },
        },
        'ombros': standard_shoulder_angle,   # só diretor ? ou ang tambem com o eixo X?
        'pescoco': np.array([5, 5, 5]), # só diretor ? ou ang tambem com o eixo y?
    }

    # Fim analisar imagem ----------------------------------------------------------------------------------------------

    # Area de testes ----------------------------------------------------------------------------------------------
    
    # Método 1 para salvar dados: criar array q salva 16 dados (em array) e depois faz a média
    nn = 16
    
    pose_now_smooth = {
        11: NormalizedLandmark(x= np.arange(nn) , y= np.arange(nn), z= np.arange(nn), visibility= np.arange(nn), presence= np.arange(nn)),
        12: NormalizedLandmark(x= np.arange(nn) , y= np.arange(nn), z= np.arange(nn), visibility= np.arange(nn), presence= np.arange(nn)),
        13: NormalizedLandmark(x= np.arange(nn) , y= np.arange(nn), z= np.arange(nn), visibility= np.arange(nn), presence= np.arange(nn)),
        14: NormalizedLandmark(x= np.arange(nn) , y= np.arange(nn), z= np.arange(nn), visibility= np.arange(nn), presence= np.arange(nn)),
        15: NormalizedLandmark(x= np.arange(nn) , y= np.arange(nn), z= np.arange(nn), visibility= np.arange(nn), presence= np.arange(nn)),
        16: NormalizedLandmark(x= np.arange(nn) , y= np.arange(nn), z= np.arange(nn), visibility= np.arange(nn), presence= np.arange(nn))
    }

    # Método 2 para salvar dados: fazer um array com um dado para cada landmark, na hora de tirar a média, multiplicar por um número (15) e adicionar o input, e dividir pelo "total" (16) (15 números iguais + o input) 
    #   media = ((ja_salvo) * 15 + input) / 16

    # pose_now_smooth = standard_pose

    # Método 3 juntar o 1 e 2, fazer um array de arrays
    # ??

    # Fim area de testes ------------------------------------------------------------------------------------------

    # Trocando o modelo
    mppose.set_modo_operacao('live_stream')
    mppose.set_live_stream_method(live_stream_function)
    mppose.set_show_live_stream(True)
    # Definindo webcam como captura de imagem
    cap = cv2.VideoCapture(0)
    # Verifica se consegui abrir a camera
    if not cap.isOpened():
        print('Não conseguiu abrir a camera')
        exit()
        pass#if
    # Começa a captura de imagem e processamento dela
    while True:
        ret, frame = cap.read()
        # Verifica de ainda consegue ler frames
        if not ret:
            print('Não está pegando mais frames (acabou?). Saindo...')
            break
            pass#if

        # Faz a detecção da pose
        mppose.detect_pose(cap=cap, frame=frame)

        # Comando para parar o programa
        if cv2.waitKey(1) == ord('q'):
            print('Terminando o programa, até mais.')
            break
            pass#if
        pass#while
    # Fecha tudo 
    cap.release()
    cv2.destroyAllWindows()
    pass