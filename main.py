from PoseComparator import PoseComparator as PC
from MPPose import MPPose
from dotenv import dotenv_values
import numpy as np
import cv2
import os

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

def update_pose_smooth_normalized(result):
    global pose_now_smooth
    for landmark_id in pose_now_smooth.keys(): 
        pose_now_smooth[landmark_id].x = update_array(pose_now_smooth[landmark_id].x, result.pose_world_landmarks[0][landmark_id].x)
        pose_now_smooth[landmark_id].y = update_array(pose_now_smooth[landmark_id].y, result.pose_world_landmarks[0][landmark_id].y)
        pose_now_smooth[landmark_id].z = update_array(pose_now_smooth[landmark_id].z, result.pose_world_landmarks[0][landmark_id].z)
        pose_now_smooth[landmark_id].presence = update_array(pose_now_smooth[landmark_id].presence, result.pose_world_landmarks[0][landmark_id].presence)
        pose_now_smooth[landmark_id].visibility = update_array(pose_now_smooth[landmark_id].visibility, result.pose_world_landmarks[0][landmark_id].visibility)
    
        # pose_now_smooth_normalized[landmark_id].x = update_array(pose_now_smooth_normalized[landmark_id].x, result.pose_landmarks[0][landmark_id].x)
        # pose_now_smooth_normalized[landmark_id].y = update_array(pose_now_smooth_normalized[landmark_id].y, result.pose_landmarks[0][landmark_id].y)
        # pose_now_smooth_normalized[landmark_id].z = update_array(pose_now_smooth_normalized[landmark_id].z, result.pose_landmarks[0][landmark_id].z)
        # pose_now_smooth_normalized[landmark_id].presence = update_array(pose_now_smooth_normalized[landmark_id].presence, result.pose_landmarks[0][landmark_id].presence)
        # pose_now_smooth_normalized[landmark_id].visibility = update_array(pose_now_smooth_normalized[landmark_id].visibility, result.pose_landmarks[0][landmark_id].visibility)
        pass # for
    pass # update_pose_smooth_normalized

def coordinates(id):
    global pose_now_smooth
    x = pose_now_smooth[id].x.mean()
    y = pose_now_smooth[id].y.mean()
    z = pose_now_smooth[id].z.mean()
    return NormalizedLandmark(x=x, y=y, z=z)
    pass

def verificar_angulo(ang_padrao, ang_teste, margem_erro: int = 2):
    return (ang_teste <= ang_padrao + margem_erro) and (ang_teste >= ang_padrao - margem_erro)
    pass # verificar_angulo

def verificar_diretores(diretor_padrao, diretor_teste, margem_erro: int = 2):
    return all(diretor_teste <= diretor_padrao + margem_erro) and all(diretor_teste >= diretor_padrao - margem_erro)
    pass # verificar_diretores

def live_stream_function(result, output_image, timestamp_ms):
    global pose_now_smooth
    global standard_pose
    os.system('cls')
    # print(type(result.pose_landmarks[0][0]))
    # PoseLandmarkerResult()
    # NormalizedLandmark()
    
    update_pose_smooth_normalized(result)

    right_arm_director_now     = PC.analyse_limb(coordinates(14), coordinates(16))
    right_forearm_director_now = PC.analyse_limb(coordinates(12), coordinates(14))
    right_arm_angle_now        = PC.angle_between_limbs(coordinates(16), coordinates(14), coordinates(12))

    right_arm_angle_ok        = verificar_angulo(standard_pose['right']['arm']['angle'], right_arm_angle_now,2)
    right_arm_director_ok     = verificar_diretores(standard_pose['right']['arm']['director'], right_arm_director_now,2)
    right_forearm_director_ok = verificar_diretores(standard_pose['right']['forearm']['director'], right_forearm_director_now,2)

    if right_arm_angle_ok:
        print('\033[32m')
    else:
        print('\033[31m')

    print('Angulo na imagem: ', standard_pose['right']['arm']['angle'])
    print('Seu angulo: ', right_arm_angle_now)
    print('Angulo está ', 'ok' if right_arm_angle_ok else 'not ok')

    # FUNCIONA
    if right_arm_director_ok:
        print('\033[32m')
    else:
        print('\033[31m')
    print('Braço da imagem tem diretores: ', standard_pose['right']['arm']['director'])
    print('Seu braço tem diretores: ', right_arm_director_now)
    print('Angulo está ', 'ok' if right_arm_director_ok else 'not ok')
    # if (standard_pose[1][0] < resposta_diretor_braco_d[0]):
    #     print('Trazer pulso direito mais para esquerda (mais perto de vc).')
    #     pass
    # else:
    #     print('Trazer pulso mais para a direita (mais longe de vc).')
    #     pass
    # FIM FUNCIONA

    # FUNCIONA +-
    # print(standard_pose[1][1])
    # print(resposta_diretor_braco_d[1])
    # if (standard_pose[1][1] < resposta_diretor_braco_d[1]):
    #     print('Trazer pulso direito mais para baixo.')
    #     pass
    # else:
    #     print('Trazer pulso mais para cima.')
    #     pass
    # FIM FUNCIONA +-


    # print(standard_pose[1][2])
    # print(resposta_diretor_braco_d[2])
    # if (standard_pose[1][2] < resposta_diretor_braco_d[2]):
    #     print('Trazer pulso direito mais para frente.')
    #     pass
    # else:
    #     print('Trazer pulso mais para traz.')
    #     pass




    # print('Diretor braco ', 'ok' if diretor_braco_d_ok else 'not ok')

    # print(pose_standard[2])
    # print(resposta_diretor_antebraco_d)
    # print('Diretor antebraco ', 'ok' if diretor_antebraco_d_ok else 'not ok')

    # if angulo_braco_d_ok and diretor_braco_d_ok and diretor_antebraco_d_ok:
    #     print('POSE OK')

    # print(resposta_angulo)


    # sentido_braco_d = PC.sentido(coordinates(12), coordinates(14), coordinates(16))

    # print(sentido_braco_standard_d)
    # print(sentido_braco_d)

    print('\033[37m')
    pass # live_stream_function

if __name__ == '__main__':

        # Lendo váriaveis
    dotenv = dotenv_values(".env")
    model_path  = dotenv['MODEL_PATH_FULL']
    model_image = dotenv['MODEL_IMAGE']
    # Iniciando o modelo
    mppose = MPPose(model_path, 'image')

    # Analisar imagem ----------------------------------------------------------------------------------------------
    
    standard       = mppose.detect_pose(image_path=model_image)
    standard_image = mppose.draw_landmarks_on_image(standard[0], standard[1])
    standard_image = cv2.resize(standard_image,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

    cv2.imshow('Iagem', cv2.cvtColor(standard_image, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)

    standard_right_arm_director     = PC.analyse_limb(standard[1].pose_world_landmarks[0][14], standard[1].pose_world_landmarks[0][16])
    standard_right_forearm_director = PC.analyse_limb(standard[1].pose_world_landmarks[0][12], standard[1].pose_world_landmarks[0][14])
    standard_right_angle        = PC.angle_between_limbs(standard[1].pose_world_landmarks[0][16], standard[1].pose_world_landmarks[0][14], standard[1].pose_world_landmarks[0][12])

    standard_right_arm_direction = PC.sentido(standard[1].pose_world_landmarks[0][12], standard[1].pose_world_landmarks[0][14], standard[1].pose_world_landmarks[0][16])

    standard_pose = {
        'right': {
            'arm': {
                'director': standard_right_arm_director,
                'angle': standard_right_angle,
                'direction': standard_right_arm_direction
            },
            'forearm': {
                'director': standard_right_forearm_director
            }
        },
        'left': {
            'arm': {}
        }
    }

    # Fim analisar imagem ----------------------------------------------------------------------------------------------

    # Area de testes ----------------------------------------------------------------------------------------------
    
    nn = 16
    # right_arm = [16, 14, 12]
    
    pose_now_smooth = {
        11: NormalizedLandmark(x= np.arange(nn) , y= np.arange(nn), z= np.arange(nn), visibility= np.arange(nn), presence= np.arange(nn)),
        12: NormalizedLandmark(x= np.arange(nn) , y= np.arange(nn), z= np.arange(nn), visibility= np.arange(nn), presence= np.arange(nn)),
        13: NormalizedLandmark(x= np.arange(nn) , y= np.arange(nn), z= np.arange(nn), visibility= np.arange(nn), presence= np.arange(nn)),
        14: NormalizedLandmark(x= np.arange(nn) , y= np.arange(nn), z= np.arange(nn), visibility= np.arange(nn), presence= np.arange(nn)),
        15: NormalizedLandmark(x= np.arange(nn) , y= np.arange(nn), z= np.arange(nn), visibility= np.arange(nn), presence= np.arange(nn)),
        16: NormalizedLandmark(x= np.arange(nn) , y= np.arange(nn), z= np.arange(nn), visibility= np.arange(nn), presence= np.arange(nn))
    }    

    # Fim area de testes ------------------------------------------------------------------------------------------

    # Trocando o modelo
    mppose.set_modo_operacao('live_stream')
    mppose.set_live_stream_method(live_stream_function)
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
            break
            pass#if
        pass#while
    # Fecha tudo 
    cap.release()
    cv2.destroyAllWindows()
    pass