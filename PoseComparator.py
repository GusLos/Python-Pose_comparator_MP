import numpy as np
from mediapipe.tasks.python.components.containers.landmark import Landmark
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.components.containers.landmark import Landmark

# class ComparadorPose(metaclass=SingletonMeta):
class PoseComparator():

  @staticmethod
  def landmark_to_array_default(landmark: Landmark) -> np.array:
    '''Tranforma/Retorna em array'''
    return np.array([landmark.x, landmark.y, landmark.z])
    pass

  @staticmethod
  def landmark_to_array_as_int(landmark: Landmark, exponent: int = 5) -> np.array:
    result = np.power(10, exponent)
    return (np.array([landmark.x, landmark.y, landmark.z]) * result).astype(int)
    pass

  @staticmethod
  def encontrar_vetor (ponto_inicial: np.array, ponto_final: np.array) -> np.array:
    '''Retorna Array'''
    return ponto_final - ponto_inicial
    pass

  @staticmethod
  def director_angles(vector) -> np.array:
    '''Recebe vetor np.array , return np.array(i, j, k) em graus, melhor radianos?  p/ graus usar np.rad2deg()'''
    vector_module = np.sqrt(vector.dot(vector)) # módulo do vetor
    director_angle_i = np.rad2deg(np.arccos(vector[0]/vector_module))
    director_angle_j = np.rad2deg(np.arccos(vector[1]/vector_module))
    director_angle_k = np.rad2deg(np.arccos(vector[2]/vector_module))
    # director_angle_i = np.arccos(vector[0]/vector_module)
    # director_angle_j = np.arccos(vector[1]/vector_module)
    # director_angle_k = np.arccos(vector[2]/vector_module)
    return np.array([ director_angle_i, director_angle_j, director_angle_k ])
    pass # director_angles
  
  @staticmethod
  def calcular_angulo_entre_vetores(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    angle = np.arccos(dot_product / (magnitude1 * magnitude2))
    angle = np.degrees(angle)
    return angle
    pass

  @classmethod
  def angle_between_limbs(cls, start_end, common_end, final_end):
    '''Funciona apenas com world_pose_landmaks, retorna o angulo entre membros em graus, melhor radiano?'''
    start_end_coordinates  = cls.landmark_to_array_default(start_end)
    final_end_coordinates  = cls.landmark_to_array_default(final_end)
    common_end_coordinates = cls.landmark_to_array_default(common_end)

    limb_vector1 = cls.encontrar_vetor(common_end_coordinates, start_end_coordinates)
    limb_vector2 = cls.encontrar_vetor(common_end_coordinates, final_end_coordinates)

    limbs_angle = cls.calcular_angulo_entre_vetores(limb_vector2, limb_vector1)

    return limbs_angle
    pass

  @classmethod
  def analyse_limb(cls, start_end, final_end) -> np.array:
    start_end_coordinates = cls.landmark_to_array_default(start_end)
    final_end_coordinates = cls.landmark_to_array_default(final_end)

    limb_vector = cls.encontrar_vetor(start_end_coordinates, final_end_coordinates)

    limb_director_angle = cls.director_angles(limb_vector)

    return limb_director_angle
    pass

  @staticmethod
  def isLeft(inicio, meio, fim) -> list:
    '''Testar (recebe pontos), retorna lista [T/F , num]'''
    x = ((meio[0] - inicio[0])*(fim[1] - inicio[1]) - (meio[1] - inicio[1])*(fim[0] - inicio[0]))
    # return [x > 0 , x]
    return x
    pass

  @classmethod
  def sentido_XY (cls, ponto_inicial: Landmark, ponto_meio: Landmark, ponto_final: Landmark) -> bool:
    '''Testar
    ! fazer mecanismo para margem de erro'''
    # ponto_inicial_XY = [ponto_inicial[0] , ponto_inicial[1]]
    # ponto_meio_XY    = [ponto_meio[0]    , ponto_meio[1]   ]
    # ponto_final_XY   = [ponto_final[0]   , ponto_final[1]  ]
    ponto_inicial_XY = [ponto_inicial.x , ponto_inicial.y]
    ponto_meio_XY    = [ponto_meio.x    , ponto_meio.y   ]
    ponto_final_XY   = [ponto_final.x   , ponto_final.y  ]
    return cls.isLeft(ponto_inicial_XY, ponto_meio_XY, ponto_final_XY)

  @classmethod
  def sentido_ZX (cls, ponto_inicial, ponto_meio, ponto_final) -> bool:
    '''Testar
    ! fazer mecanismo para margem de erro'''
    ponto_inicial_ZX = [ponto_inicial.z , ponto_inicial.x]
    ponto_meio_ZX    = [ponto_meio.z    , ponto_meio.x   ]
    ponto_final_ZX   = [ponto_final.z   , ponto_final.x  ]
    return cls.isLeft(ponto_inicial_ZX, ponto_meio_ZX, ponto_final_ZX)

  @classmethod
  def sentido_YZ (cls, ponto_inicial: Landmark, ponto_meio: Landmark, ponto_final: Landmark) -> bool:
    '''Testar
    ! fazer mecanismo para margem de erro'''
    ponto_inicial_YZ = [ponto_inicial.y , ponto_inicial.z]
    ponto_meio_YZ    = [ponto_meio.y    , ponto_meio.z   ]
    ponto_final_YZ   = [ponto_final.y   , ponto_final.z  ]
    return cls.isLeft(ponto_inicial_YZ, ponto_meio_YZ, ponto_final_YZ)

  @classmethod
  def sentido (cls, ponto_inicial: Landmark, ponto_meio: Landmark, ponto_final: Landmark):
    '''Testar
    ! fazer mecanismo para margem de erro'''
    sentido_xy = cls.sentido_XY(ponto_inicial, ponto_meio, ponto_final)
    sentido_zx = cls.sentido_ZX(ponto_inicial, ponto_meio, ponto_final)
    sentido_yz = cls.sentido_YZ(ponto_inicial, ponto_meio, ponto_final)
    # return {'xy': sentido_xy, 'zx': sentido_zx, 'yz': sentido_yz}
    return np.array([sentido_xy, sentido_zx, sentido_yz])




  @classmethod
  def affine_transformation (cls) -> None:
    pass # affine_transformation

  pass