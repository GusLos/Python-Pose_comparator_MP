from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.framework.formats import landmark_pb2
from PoseComparator import PoseComparator as PC
from dotenv import dotenv_values
from mediapipe import solutions
from read_csv import ReadCSV
from MPPose import MPPose
import numpy as np
import cv2
import os


# Iniciando variáveis globais para usar na função assíncrona
global model_data
global pose_now


# global landmark_colors
# global connection_colors
# global pose_connections
landmark_colors = {
	1: 	DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
	2: 	DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
	3: 	DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
	7: 	DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
	9: 	DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
	11: DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
	13: DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
	15: DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
	17: DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
	19: DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
	21: DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
	23: DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
	25: DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
	27:	DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
	29:	DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
	31: DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
	32: DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
	4: 	DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
	5: 	DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
	6: 	DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
	8: 	DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
	10: DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
	12: DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
	14: DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
	16: DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
	18: DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
	20: DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
    22: DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
    24: DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
    26: DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
    28: DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
    30: DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
	0: 	DrawingSpec(color=(224, 224, 224), thickness=2, circle_radius=2)
}
connection_colors = {
    (0, 1): DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
    (1, 2): DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
    (2, 3): DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
    (3, 7): DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
    (0, 4): DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
    (4, 5): DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
    (5, 6): DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
    (6, 8): DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
    (9, 10): DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
    (11, 12): DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
    (11, 13): DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
    (13, 15): DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
    (15, 17): DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
    (15, 19): DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
    (15, 21): DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
    (17, 19): DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
    (12, 14): DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
    (14, 16): DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
    (16, 18): DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
    (16, 20): DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
    (16, 22): DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
    (18, 20): DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
    (11, 23): DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
    (12, 24): DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
    (23, 24): DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
    (23, 25): DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
    (24, 26): DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
    (25, 27): DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
    (26, 28): DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
    (27, 29): DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
    (28, 30): DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
    (29, 31): DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
    (30, 32): DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
    (27, 31): DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), 
    (28, 32): DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2)
    }
pose_connections = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
    (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
    (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
    (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
    (29, 31), (30, 32), (27, 31), (28, 32)]


# Calcula a cor (em BGR (0, 0, 0)) para pintar o alvo, quanto mais perto do valor alvo, mais verde
def color_feedback(target_value, value, value_std, min_value = 0, max_value = 50):
    
    # Verifica a diferença entre o alvo e o valor testado
    difference = abs( target_value - value )

    # Se estiver dentro do desvio padrão, considera como certo
    if difference <= value_std: return (0, 255, 0)
    
    # Define a intensidade máxima de verde (G) quando os valores são iguais
    max_green = 200
    
    # Calcula a diferença com base no intervalo definido
    # diferenca = abs((target_value - value) / (max_value - min_value))
    difference_n = difference / (max_value - min_value)

    # Calcula a intensidade do verde (G) com base na diferença
    green_intensity = int(200 - (difference_n * 255))
    
    # Calcula a intensidade do vermelho (R) como o complemento da intensidade de verde
    red_intensity = max_green - green_intensity
    
    # Retorna a cor no formato BGR (Blue, Green, Red)
    color = (0, green_intensity, red_intensity)
    
    return color


# Função para desenhar o resultado na imagem
def draw_landmarks_on_image(
        rgb_image, 
        detection_result, 
        pose_connections = pose_connections, 
        landmark_colors = landmark_colors, 
        connection_colors = connection_colors
        ):
    
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image.numpy_view())

    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            pose_connections,
            landmark_colors,
            connection_colors)

    return annotated_image


# Método para atualizar o array
def update_array(target_array, number):
    target_array = np.delete(target_array, 0)
    target_array = np.append(target_array, number)
    return target_array
    pass # update_array


# Método para atualizar dados de entrada de forma suave
def update_pose_now(result):
    # Carrega a variável global
    global pose_now

    # Atualiza todas as coordenadas
    for landmark in range(33):
        pose_now[0][landmark].x = update_array(pose_now[0][landmark].x, result[0][landmark].x)
        pose_now[0][landmark].y = update_array(pose_now[0][landmark].y, result[0][landmark].y)
        pose_now[0][landmark].z = update_array(pose_now[0][landmark].z, result[0][landmark].z)
        pose_now[0][landmark].presence   = update_array(pose_now[0][landmark].presence  , result[0][landmark].presence)
        pose_now[0][landmark].visibility = update_array(pose_now[0][landmark].visibility, result[0][landmark].visibility)
        landmark = NormalizedLandmark()
        pass # for
    pass # update_pose_now


# Método para transformar 
def pose_now_mean():
    # Carrega variavel global com arrays no lugar de dados únicos
    global pose_now

    # Cria variável para com méida dos dados na variável global
    pose_now_mean = [[x for x in range(33)]]

    # Calcula a média de cada dado na variável global e guarda na variável "temporária"
    for landmark_id in range(33):
        x = pose_now[0][landmark_id].x.mean()
        y = pose_now[0][landmark_id].y.mean()
        z = pose_now[0][landmark_id].z.mean()
        visibility = pose_now[0][landmark_id].visibility.mean()
        presence = pose_now[0][landmark_id].presence.mean()
        landmark = NormalizedLandmark(x=x , y=y, z=z, visibility=visibility, presence=presence)
        pose_now_mean[0][landmark_id] = landmark
        pass # for

    # Devolve a variável temporária
    return pose_now_mean
    pass # pose_now_to_array


# Barra que enche comforme mais perto se chaga do valor alvo
def bar_feedback(barra, valor_atual, limite_inferior=1):
    tamanho = len(barra)
    preenchimento = int( ((limite_inferior - valor_atual) / limite_inferior) * tamanho )
    feedback = barra[:preenchimento] + '-' * (tamanho - preenchimento)
    return feedback
    pass # barra_feedback


# Comparação de angulos apenas dos braços
def arm_angle_feedback(angle_input, angle_model, angle_std):
    # Calcula a diferença entre o atual e o modelo
    angle_diff = angle_input - angle_model

    # Se estiver dentro do desvio padrão, pode ser considerado certo
    if (abs(angle_diff) <= angle_std) : return 'OK' 

    # Caso contrario, propor um feedback
    feedback = f'{angle_diff:.0f}', 'Fechar mais os braços ou colocar cotovelo mais para frente.'  if angle_diff > 0  else 'Abrir mais os braços ou colocar cotovelo mais para trás.'
    return feedback
    pass # angle_feedback


# Comparação de angulo apenas dos antebraços
def forearm_angle_feedback(angle_input, angle_model, angle_std):
    # Calcula a diferença entre o atual e o modelo
    angle_diff = angle_input - angle_model

    # Se estiver dentro do desvio padrão, pode ser considerado certo
    if (abs(angle_diff) <= angle_std) : return 'OK'

    # Caso contrario, propor um feedback
    feedback = f'{angle_diff:.0f}', 'Cotovelo mais perto do corpo ou colocar cotovelo mais para frente.'  if angle_diff > 0  else 'Cotovelo mais longe do corpo ou colocar cotovelo mais para trás.'
    return feedback
    pass # forearm_angle_feedback


# Verifica tanto o braço direito quanto esquerdo, se está para cima e por "quanto"
def directions(pose):
    # Pegando pontos principais para se calcular a direção do braço
    left_shoulder  = [ pose[0][11].x, pose[0][11].y, pose[0][11].z ]
    right_shoulder = [ pose[0][12].x, pose[0][12].y, pose[0][12].z ]
    left_elbow     = [ pose[0][13].x, pose[0][13].y, pose[0][13].z ]
    right_elbow    = [ pose[0][14].x, pose[0][14].y, pose[0][14].z ]
    left_wrist     = [ pose[0][15].x, pose[0][15].y, pose[0][15].z ]
    right_wrist    = [ pose[0][16].x, pose[0][16].y, pose[0][16].z ]

    # Após calculo, recebe a mensagem ideal, baseada na mensagem dada
    left_arm_up  = PC.is_up(left_shoulder, left_elbow, left_wrist)
    right_arm_up = PC.is_up(right_shoulder, right_elbow, right_wrist)

    # Develve mensagem ideal tanto para direita quanto para esquerda
    return left_arm_up, right_arm_up


# Função para gerar cores das landmarks de acordo com a proximidade do angulo certo
def get_custom_landmark_colors(base_data, pose_data):
    # Carregando as cores padrões para desenhar na imagem
    custom_landmark_colors = landmark_colors

    # Calculando cores para usar nos resultados
    right_arm_color = color_feedback(target_value=base_data['upper_limbs']['right']['arm']['angle'],value=pose_data['upper_limbs']['right']['arm']['angle'], value_std=base_data['upper_limbs']['right']['arm']['angle_std'])
    left_arm_color  = color_feedback(target_value=base_data['upper_limbs']['left']['arm']['angle'],value=pose_data['upper_limbs']['left']['arm']['angle'], value_std=base_data['upper_limbs']['left']['arm']['angle_std'])

    # Aplicando cores nas landmarks
    custom_landmark_colors[14].color = right_arm_color
    custom_landmark_colors[14].thickness = 10

    custom_landmark_colors[13].color = left_arm_color
    custom_landmark_colors[13].thickness = 10

    return custom_landmark_colors
    pass


def get_custom_connection_colors(base_data, pose_data):

    custom_connection_colors = connection_colors

    # color_feedback(target_value=base_data['upper_limbs']['right']['arm']['angle'],value=pose_data['upper_limbs']['right']['arm']['angle'], value_std=base_data['upper_limbs']['right']['arm']['angle_std'])
    color_feedback()

    print(pose_data)

    # Definindo cores de acordo com a "landmark"
    # custom_connection_colors[(14, 16)].color = right_arm_color
    # custom_connection_colors[(12, 14)].color = right_arm_color
    # custom_connection_colors[(14, 16)].thickness = 4
    # custom_connection_colors[(12, 14)].thickness = 4

    # custom_connection_colors[(13, 15)].color = left_arm_color
    # custom_connection_colors[(11, 13)].color = left_arm_color
    # custom_connection_colors[(13, 15)].thickness = 4
    # custom_connection_colors[(11, 13)].thickness = 4

    pass


# FAZER...
# Método para validar pose
def validar_pose(model_data, pose_data):
    # Carregando as cores customizadas para desenhar na imagem
    custom_landmark_colors = get_custom_connection_colors(model_data, pose_data)
    custom_connections_colors = get_custom_connection_colors(model_data, pose_data)
    
    return (custom_landmark_colors, custom_connections_colors)
    pass # validar_pose



# Método que vai ser usado na função assíncrona
def live_stream_method(result, output_image, timestamp_ms):
    # Carregando as variáveis globais
    global model_data
    global pose_now

    # Limpa tela da ultima iteração
    os.system('cls')

    # Update da variável global com dados mais novos
    update_pose_now(result.pose_world_landmarks)

    # Pega a média dos dados salvos na variável global pose_now
    pose_mean = pose_now_mean()

    # teste de verificar o sentido
    teste_sentido = directions(pose_mean)

    # Analisar a pose atual / Extrair informações / Angulos e diretores
    pose_analysed = PC.analyse_torso_wo_affine(pose_mean)

    # Carregando as cores customizadas para desenhar na imagem
    # custom_colors = validar_pose(model_data, pose_analysed)

    # Desenhando resultado na imagem
    annotated_image = draw_landmarks_on_image(rgb_image = output_image, detection_result = result)#, landmark_colors = custom_colors[0], connection_colors = custom_colors[1])

    # Mostrando a imagem
    cv2.imshow('frame', annotated_image )
    if cv2.waitKey(1) == ord('q'):
        os.abort()

    pass


if __name__ == "__main__":

    # Ler variáveis
    dotenv = dotenv_values('.env')
    mp_model_path    = dotenv['MODEL_PATH_FULL']
    model_image_path = dotenv['MODEL_IMAGE']

    # Iniciando o mediapipe
    mppose = MPPose(model_path=mp_model_path, running_mode='live_stream', live_stream_method=live_stream_method, show_live_stream_result=False)

    # Carregando dados da pose
    model_data = ReadCSV().read_pose_data('eggs.csv')

    # Cria variável para ler a entrada de jeito mais suave
    nn = 16
    pose_now = [[x for x in range(33)]]
    for landmark in range(33):
        pose_now[0][landmark] = NormalizedLandmark(x= np.arange(nn) , y= np.arange(nn), z= np.arange(nn), visibility= np.arange(nn), presence= np.arange(nn))
        pass

    # Mostrando imagem modelo para usuário
    model_image = cv2.imread(model_image_path)
    model_image = cv2.resize(model_image,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    cv2.imshow('Iagem', model_image)

    # Começando captura de imagem
    # Definindo webcam como entrada/captura de imagem
    cap = cv2.VideoCapture(0)

    # Verifica se consegue abrir a camera
    if not cap.isOpened():
        print('Não consegui abrir a camera.')
        exit()
        pass#if

    # Captura de imagem e processamento
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
    # Fecha telas do opencv
    cap.release()
    cv2.destroyAllWindows()
    pass