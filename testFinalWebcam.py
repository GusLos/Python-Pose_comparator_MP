from PoseComparator import PoseComparator as PC
from MPPose import MPPose
from dotenv import dotenv_values
import numpy as np
import cv2
import os

from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

global pose_now_smooth_normalized
global pose_padrao

def update_array(target_array: np.array, number: int or float) -> np.array:
    target_array = np.delete(target_array, 0)
    # target_array = np.append(target_array, number).astype(int)
    target_array = np.append(target_array, number)
    return target_array
    pass

def update_pose_smooth_normalized(result):
    global pose_now_smooth_normalized
    for landmark_id in pose_now_smooth_normalized.keys(): 
        pose_now_smooth_normalized[landmark_id].x = update_array(pose_now_smooth_normalized[landmark_id].x, result.pose_world_landmarks[0][landmark_id].x)
        pose_now_smooth_normalized[landmark_id].y = update_array(pose_now_smooth_normalized[landmark_id].y, result.pose_world_landmarks[0][landmark_id].y)
        pose_now_smooth_normalized[landmark_id].z = update_array(pose_now_smooth_normalized[landmark_id].z, result.pose_world_landmarks[0][landmark_id].z)
        pose_now_smooth_normalized[landmark_id].presence = update_array(pose_now_smooth_normalized[landmark_id].presence, result.pose_world_landmarks[0][landmark_id].presence)
        pose_now_smooth_normalized[landmark_id].visibility = update_array(pose_now_smooth_normalized[landmark_id].visibility, result.pose_world_landmarks[0][landmark_id].visibility)
    
        # pose_now_smooth_normalized[landmark_id].x = update_array(pose_now_smooth_normalized[landmark_id].x, result.pose_landmarks[0][landmark_id].x)
        # pose_now_smooth_normalized[landmark_id].y = update_array(pose_now_smooth_normalized[landmark_id].y, result.pose_landmarks[0][landmark_id].y)
        # pose_now_smooth_normalized[landmark_id].z = update_array(pose_now_smooth_normalized[landmark_id].z, result.pose_landmarks[0][landmark_id].z)
        # pose_now_smooth_normalized[landmark_id].presence = update_array(pose_now_smooth_normalized[landmark_id].presence, result.pose_landmarks[0][landmark_id].presence)
        # pose_now_smooth_normalized[landmark_id].visibility = update_array(pose_now_smooth_normalized[landmark_id].visibility, result.pose_landmarks[0][landmark_id].visibility)
        pass # for
    pass # update_pose_smooth_normalized

def coordinates(id):
    global pose_now_smooth_normalized
    x = pose_now_smooth_normalized[id].x.mean()
    y = pose_now_smooth_normalized[id].y.mean()
    z = pose_now_smooth_normalized[id].z.mean()
    return NormalizedLandmark(x=x, y=y, z=z)
    pass

def verificar_angulo(ang_padrao, ang_teste):
    return (ang_teste <= ang_padrao + 2) and (ang_teste >= ang_padrao - 2)
    pass # verificar_angulo

def verificar_diretores(diretor_padrao, diretor_teste):
    return all(diretor_teste <= diretor_padrao + 2) and all(diretor_teste >= diretor_padrao - 2)
    pass # verificar_diretores

def live_stream_function(result, output_image, timestamp_ms):
    global pose_now_smooth_normalized
    global pose_padrao
    os.system('cls')
    # print(type(result.pose_landmarks[0][0]))
    # PoseLandmarkerResult()
    # NormalizedLandmark()
    
    update_pose_smooth_normalized(result)

    resposta_diretor_braco_d     = PC.analyse_limb(coordinates(14), coordinates(16))
    resposta_diretor_antebraco_d = PC.analyse_limb(coordinates(12), coordinates(14))
    resposta_angulo_d = PC.angle_between_limbs(coordinates(16), coordinates(14), coordinates(12))

    print('entrei')

    angulo_braco_d_ok      = verificar_angulo(pose_padrao[0], resposta_angulo_d)
    diretor_braco_d_ok     = verificar_diretores(pose_padrao[1], resposta_diretor_braco_d)
    diretor_antebraco_d_ok = verificar_diretores(pose_padrao[2], resposta_diretor_antebraco_d)


    print('Angulo ', 'ok' if angulo_braco_d_ok else 'not ok')
    print(pose_padrao[0])
    print(resposta_angulo_d)
    
    # FUNCIONA
    # print(pose_padrao[1][0])
    # print(resposta_diretor_braco_d[0])
    # if (pose_padrao[1][0] < resposta_diretor_braco_d[0]):
    #     print('Trazer pulso direito mais para esquerda (mais perto de vc).')
    #     pass
    # else:
    #     print('Trazer pulso mais para a direita (mais longe de vc).')
    #     pass
    # FIM FUNCIONA

    # FUNCIONA +-
    # print(pose_padrao[1][1])
    # print(resposta_diretor_braco_d[1])
    # if (pose_padrao[1][1] < resposta_diretor_braco_d[1]):
    #     print('Trazer pulso direito mais para baixo.')
    #     pass
    # else:
    #     print('Trazer pulso mais para cima.')
    #     pass
    # FIM FUNCIONA +-


    # print(pose_padrao[1][2])
    # print(resposta_diretor_braco_d[2])
    # if (pose_padrao[1][2] < resposta_diretor_braco_d[2]):
    #     print('Trazer pulso direito mais para frente.')
    #     pass
    # else:
    #     print('Trazer pulso mais para traz.')
    #     pass




    # print('Diretor braco ', 'ok' if diretor_braco_d_ok else 'not ok')

    # print(pose_padrao[2])
    # print(resposta_diretor_antebraco_d)
    # print('Diretor antebraco ', 'ok' if diretor_antebraco_d_ok else 'not ok')

    # if angulo_braco_d_ok and diretor_braco_d_ok and diretor_antebraco_d_ok:
    #     print('POSE OK')

    # print(resposta_angulo)


    sentido_braco_d = PC.sentido(coordinates(12), coordinates(14), coordinates(16))

    print(sentido_braco_padrao_d)
    print(sentido_braco_d)


    pass # live_stream_function

if __name__ == '__main__':

        # Lendo váriaveis
    dotenv = dotenv_values(".env")
    model_path  = dotenv['MODEL_PATH_FULL']
    model_image = dotenv['MODEL_IMAGE']
    # Iniciando o modelo
    mppose = MPPose(model_path, 'image')

    # Analisar imagem ----------------------------------------------------------------------------------------------
    
    padrao        = mppose.detect_pose(image_path=model_image)
    imagem_padrao = mppose.draw_landmarks_on_image(padrao[0], padrao[1])
    imagem_padrao = cv2.resize(imagem_padrao,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

    cv2.imshow('Iagem', cv2.cvtColor(imagem_padrao, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)

    padrao_diretor_braco_d     = PC.analyse_limb(padrao[1].pose_world_landmarks[0][14], padrao[1].pose_world_landmarks[0][16])
    padrao_diretor_antebraco_d = PC.analyse_limb(padrao[1].pose_world_landmarks[0][12], padrao[1].pose_world_landmarks[0][14])
    padrao_angulo_d = PC.angle_between_limbs(padrao[1].pose_world_landmarks[0][16], padrao[1].pose_world_landmarks[0][14], padrao[1].pose_world_landmarks[0][12])

    sentido_braco_padrao_d = PC.sentido(padrao[1].pose_world_landmarks[0][12], padrao[1].pose_world_landmarks[0][14], padrao[1].pose_world_landmarks[0][16])

    # print('diretor braco', padrao_diretor_braco_d)
    # print('diretor antebraco', padrao_diretor_antebraco_d)
    # print('angulo', padrao_angulo_d)

    pose_padrao = [padrao_angulo_d, padrao_diretor_braco_d, padrao_diretor_antebraco_d]

    # Fim analisar imagem ----------------------------------------------------------------------------------------------

    # Area de testes ----------------------------------------------------------------------------------------------
    
    nn = 16
    # right_arm = [16, 14, 12]
    
    pose_now_smooth_normalized = {
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