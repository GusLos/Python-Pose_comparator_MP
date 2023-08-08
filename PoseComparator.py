import numpy as np
from mediapipe.tasks.python.components.containers.landmark import Landmark
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.components.containers.landmark import Landmark

# class ComparadorPose(metaclass=SingletonMeta):
class PoseComparator():

  @staticmethod
  def landmark_to_array_default(landmark: Landmark) -> np.array:
    return np.array([landmark.x, landmark.y, landmark.z])
    pass

  @staticmethod
  def landmark_to_array_as_int(landmark: Landmark, exponent: int = 5) -> np.array:
    result = np.power(10, exponent)
    return (np.array([landmark.x, landmark.y, landmark.z]) * result).astype(int)
    pass

  @staticmethod
  def encontrar_vetor (ponto_inicial: np.array, ponto_final: np.array) -> np.array:
    return ponto_final - ponto_inicial
    pass

  @staticmethod
  def calcular_angulo_entre_vetores(vetor1, vetor2):
    produto_escalar = np.dot(vetor1, vetor2)
    norma_vetor1 = np.linalg.norm(vetor1)
    norma_vetor2 = np.linalg.norm(vetor2)
    cos_theta = produto_escalar / (norma_vetor1 * norma_vetor2)
    theta_rad = np.arccos(cos_theta)
    theta_graus = np.degrees(theta_rad)
    return theta_graus

  @classmethod
  def analisar_braco(cls, lista_pose_landmarkers: list[Landmark], left: bool = 1):
    ombro  = cls.landmark_to_array_default(lista_pose_landmarkers[12 - left])
    cotovelo = cls.landmark_to_array_default(lista_pose_landmarkers[14 - left])
    pulso    = cls.landmark_to_array_default(lista_pose_landmarkers[16 - left])

    braco     = cls.encontrar_vetor(cotovelo, ombro)
    antebraco = cls.encontrar_vetor(cotovelo, pulso)

    angulo_braco_antebraco = cls.calcular_angulo_entre_vetores(braco, antebraco)
    sentido_braco_antebraco = cls.sentido(ombro, cotovelo, pulso)

    ombro_oposto  = cls.landmark_to_array_default(lista_pose_landmarkers[11 + left])

    braco       = cls.encontrar_vetor(ombro, cotovelo)
    ombro_ombro = cls.encontrar_vetor(ombro, ombro_oposto)

    angulo_ombro = cls.calcular_angulo_entre_vetores(braco, ombro_ombro)
    sentido_ombro = cls.sentido(ombro_oposto, ombro, cotovelo)

    return angulo_braco_antebraco, sentido_braco_antebraco, angulo_ombro, sentido_ombro
    pass









  @classmethod
  def analyse(cls, detection_result: list[Landmark]) -> dict:
    # pose_landmarkers = detection_result.pose_landmarks[0]
    right_arm        = cls.analisar_braco(detection_result, left=0)
    left_arm         = cls.analisar_braco(detection_result, left=1)

    result = {
      'right_arm': {
        'elbow':    {'angle': right_arm[0], 'direction': right_arm[1]},
        'shoulder': { 'angle': right_arm[2], 'direction': right_arm[3]}},
      'left_arm': {
        'elbow':    {'angle': left_arm[0], 'direction': left_arm[1]},
        'shoulder': { 'angle': left_arm[2], 'direction': left_arm[3]}}
                 }

    return result
    pass

  







  @staticmethod
  def isLeft(inicio, meio, fim) -> list:
    '''Testar (recebe pontos)'''
    x = ((meio[0] - inicio[0])*(fim[1] - inicio[1]) - (meio[1] - inicio[1])*(fim[0] - inicio[0]))
    return [x > 0 , x]
    pass

  @classmethod
  def sentido_XY (cls, ponto_inicial: Landmark, ponto_meio: Landmark, ponto_final: Landmark) -> bool:
    '''Testar
    ! fazer mecanismo para margem de erro'''
    ponto_inicial_XY = [ponto_inicial[0] , ponto_inicial[1]]
    ponto_meio_XY    = [ponto_meio[0]    , ponto_meio[1]   ]
    ponto_final_XY   = [ponto_final[0]   , ponto_final[1]  ]
    return cls.isLeft(ponto_inicial_XY, ponto_meio_XY, ponto_final_XY)

  @classmethod
  def sentido_ZX (cls, ponto_inicial, ponto_meio, ponto_final) -> bool:
    '''Testar
    ! fazer mecanismo para margem de erro'''
    ponto_inicial_ZX = [ponto_inicial[2] , ponto_inicial[0]]
    ponto_meio_ZX    = [ponto_meio[2]    , ponto_meio[0]   ]
    ponto_final_ZX   = [ponto_final[2]   , ponto_final[0]  ]
    return cls.isLeft(ponto_inicial_ZX, ponto_meio_ZX, ponto_final_ZX)

  @classmethod
  def sentido_YZ (cls, ponto_inicial: Landmark, ponto_meio: Landmark, ponto_final: Landmark) -> bool:
    '''Testar
    ! fazer mecanismo para margem de erro'''
    ponto_inicial_YZ = [ponto_inicial[1] , ponto_inicial[2]]
    ponto_meio_YZ    = [ponto_meio[1]    , ponto_meio[2]   ]
    ponto_final_YZ   = [ponto_final[1]   , ponto_final[2]  ]
    return cls.isLeft(ponto_inicial_YZ, ponto_meio_YZ, ponto_final_YZ)

  @classmethod
  def sentido (cls, ponto_inicial: Landmark, ponto_meio: Landmark, ponto_final: Landmark) -> dict:
    '''Testar
    ! fazer mecanismo para margem de erro'''
    sentido_xy = cls.sentido_XY(ponto_inicial, ponto_meio, ponto_final)
    sentido_zx = cls.sentido_ZX(ponto_inicial, ponto_meio, ponto_final)
    sentido_yz = cls.sentido_YZ(ponto_inicial, ponto_meio, ponto_final)
    return {'xy': sentido_xy, 'zx': sentido_zx, 'yz': sentido_yz}

  pass