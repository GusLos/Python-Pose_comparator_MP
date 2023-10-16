import numpy as np
from mediapipe.tasks.python.components.containers.landmark import Landmark
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.components.containers.landmark import Landmark

# class ComparadorPose(metaclass=SingletonMeta):
class PoseComparator():


  @staticmethod
  def landmark_to_array_default(landmark: Landmark) -> np.array:
    '''
    Receives a landmark and turn it into an array.

    #### Parameters
      landmark (Landmark):
        One "landmark result" of MediaPipe pose detection, it should have x, y and z attributes.
    
    #### Returns
      np.array:
        The coordinates of the landmark in an array format.
    '''
    return np.array([landmark.x, landmark.y, landmark.z])
    pass


  @staticmethod
  def landmark_to_array_as_int(landmark: Landmark, exponent: int = 5) -> np.array:
    '''
    Receives a landmark and turn it into an integer array. Same thing as `landmark_to_array_default`, but return an array with integers.

    #### Parameters
      landmark (Landmark):
        One "landmark result" of MediaPipe pose detection, it should have x, y and z attributes.
    
    #### Returns
      np.array:
        The coordinates of the landmark in an (integer) array format.
    '''
    result = np.power(10, exponent)
    return (np.array([landmark.x, landmark.y, landmark.z]) * result).astype(int)
    pass


  @staticmethod
  def landmarks_result_to_array(landmark_result_list: list) -> np.array:
    '''
    Method receives a list of landmark result and turn it into an array.

    ### Parameters
      landmark_result_list (list[landmark]):
        List of pose landmark, the result of MediaPipe pose detection.

    ### Returns
      landmark_result_array (np.array([np.array, ...])):
        The results, but in array format.
    '''
    # Add visibility and presence ?
    landmark_result_array = []
    for landmark in landmark_result_list:
      landmark_result_array.append( np.array([landmark.x, landmark.y, landmark.z]) )
      pass # for
    return np.array(landmark_result_array)
    pass # landmarks_result_to_array


  @staticmethod
  def create_vector (initial_point: np.array, end_point: np.array) -> np.array:
    '''
    Create an array given two points.

    #### Parameters
      initial_point (np.array):
        An array with coordinates of the start of the vector.
      end_point (np.array):
        An array with coordinates of the end of the vector.
    
    #### Returns
      np.array:
        An vector, in array format, from the start point to the end point.
    '''
    return end_point - initial_point
    pass


  @staticmethod
  def director_angles(vector: np.array) -> np.array:
    '''
    Calculates the directors angles for a given vector.

    #### Parameters
      vector (np.array):
        A vector in array format.
    
    #### Returns
      np.array
        An array with the director angles: i, j, k (respectively) in degree.
    '''
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
  def angle_between_vectors(vector1: np.array, vector2: np.array) -> float:
    '''
    Calculates the angle between two given vectors.

    #### Parameters
      vector1 (np.array):
        A vector in array format.
      vector2 (np.array):
        A vector in array format.
      
    #### Returns
      angle (float)
        The angle between two vectors.
    '''
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    angle = np.arccos(dot_product / (magnitude1 * magnitude2))
    angle = np.degrees(angle)
    return angle
    pass


  @classmethod
  def angle_between_limbs(cls, start_end, common_end, final_end):
    '''
    Given three pose landmarks (in array format or landmark format), calculates the angle shaped by these limbs.

    #### Parameters
      start_end (np.array or landmark):
        One of the extremes points, cannot be the common point, either in array format or landmark format (has the attributes x, y, z)
      common_end (np.array or landmark):
        The common point betweed the other two, either in array format or landmark format (has the attributes x, y, z)
      final_end (np.array or landmark):
        One of the extremes points, cannot be the common point, either in array format or landmark format (has the attributes x, y, z)

    #### Returns
      angle (float):
        Angle shaped by the given three points.
    '''
    try: # se já for array, n precisa transformar
      start_end  = cls.landmark_to_array_default(start_end)
      final_end  = cls.landmark_to_array_default(final_end)
      common_end = cls.landmark_to_array_default(common_end)
    except:
      pass

    limb_vector1 = cls.create_vector(common_end, start_end)
    limb_vector2 = cls.create_vector(common_end, final_end)

    limbs_angle = cls.angle_between_vectors(limb_vector2, limb_vector1)

    return limbs_angle
    pass


  @classmethod
  def analyse_limb(cls, start_point_landmark = None, final_point_landmark = None, start_point_array = None, final_point_array = None) -> np.array:
    '''
    Calculates the directors angles of the limb, given the start point and final point.

    #### Parameters
      start_point_landmark (landmark):
        The star point of the limb in landmark format.
      final_point_landmark (landmark):
        The final point of the limb in landmark format.
      start_point_array (np.array):
        The final point of the limb in array format.
      final_point_array (np.array):
        The final point of the limb in array format.
    
    #### Returns
      director angles (np.array):
        An array with the director angles: i, j, k (respectively) in degree.
    '''
    if start_point_landmark and final_point_landmark:
      start_end_coordinates = cls.landmark_to_array_default(start_point_landmark)
      final_end_coordinates = cls.landmark_to_array_default(final_point_landmark)
    elif start_point_array.any() and final_point_array.any():
      start_end_coordinates = start_point_array
      final_end_coordinates = final_point_array

    # assert start_end_coordinates and final_end_coordinates, 'At least one pair of values should be used, either a pair of landmark or array values.'
    assert np.logical_and(start_end_coordinates, final_end_coordinates).sum() , 'At least one pair of values should be used, either a pair of landmark or array values.'

    limb_vector = cls.create_vector(start_end_coordinates, final_end_coordinates)

    limb_director_angle = cls.director_angles(limb_vector)

    return limb_director_angle
    pass


  @staticmethod
  def isLeft(start, middle, end) -> float:
    '''
    checks if the end point is to the left of the line.

    #### Parameters
      start (np.array or int or float):
        Coordinate of a point in the line.
      middle
        Coordinate of a different point in the line, that connects the start to the end.
      end
        Coordinate of a point outside the line.
        
    #### Returns
      float:
        If returned number is greater than zero (> 0), end point is to the left of the line.
    '''
    x = ((middle[0] - start[0])*(end[1] - start[1]) - (middle[1] - start[1])*(end[0] - start[0]))
    # return [x > 0 , x]
    return x
    pass


  @classmethod
  def isLeft_XY (cls, start_point: Landmark, middle_point: Landmark, final_point: Landmark) -> float:
    '''
    Caution
    It should checks if the final landmark (final point) is to the left of the line made by the other two.
    
    #### Parameters
      start_point (Landmark):
        The start landmark point (it needs to have x, y and z attributes).
      middle_point (Landmark):
        The middle landmark point (it needs to have x, y and z attributes).
      final_point (Landmark):
        The final landmark point (it needs to have x, y and z attributes).

    #### Returns
      float:
        If returned number is greater than zero (> 0), final point is to the left of the line.
    '''
    # ponto_inicial_XY = [ponto_inicial[0] , ponto_inicial[1]]
    # ponto_meio_XY    = [ponto_meio[0]    , ponto_meio[1]   ]
    # ponto_final_XY   = [ponto_final[0]   , ponto_final[1]  ]
    start_point_XY = [start_point.x , start_point.y]
    middle_point_XY    = [middle_point.x    , middle_point.y   ]
    final_point_XY   = [final_point.x   , final_point.y  ]
    return cls.isLeft(start_point_XY, middle_point_XY, final_point_XY)


  @classmethod
  def isLeft_ZX (cls, start_point, middle_point, final_point) -> bool:
    '''
    Caution
    It should checks if the final landmark (final point) is to the left of the line made by the other two.
    
    #### Parameters
      start_point (Landmark):
        The start landmark point (it needs to have x, y and z attributes).
      middle_point (Landmark):
        The middle landmark point (it needs to have x, y and z attributes).
      final_point (Landmark):
        The final landmark point (it needs to have x, y and z attributes).

    #### Returns
      float:
        If returned number is greater than zero (> 0), final point is to the left of the line.
    '''
    start_point_ZX = [start_point.z , start_point.x]
    middle_point_ZX    = [middle_point.z    , middle_point.x   ]
    final_point_ZX   = [final_point.z   , final_point.x  ]
    return cls.isLeft(start_point_ZX, middle_point_ZX, final_point_ZX)


  @classmethod
  def isLeft_YZ (cls, start_point: Landmark, middle_point: Landmark, final_point: Landmark) -> float:
    '''
    Caution
    It should checks if the final landmark (final point) is to the left of the line made by the other two.
    
    #### Parameters
      start_point (Landmark):
        The start landmark point (it needs to have x, y and z attributes).
      middle_point (Landmark):
        The middle landmark point (it needs to have x, y and z attributes).
      final_point (Landmark):
        The final landmark point (it needs to have x, y and z attributes).

    #### Returns
      float:
        If returned number is greater than zero (> 0), final point is to the left of the line.
    '''
    start_point_YZ  = [start_point.y , start_point.z ]
    middle_point_YZ = [middle_point.y, middle_point.z]
    final_point_YZ  = [final_point.y , final_point.z ]
    return cls.isLeft(start_point_YZ, middle_point_YZ, final_point_YZ)


  @classmethod
  def isLeft_XYZ (cls, start_point: Landmark, middle_point: Landmark, final_point: Landmark) -> np.array:
    '''
    Caution
    It should checks if the final landmark (final point) is to the left of the line made by the other two.
    
    #### Parameters
      start_point (Landmark):
        The start landmark point (it needs to have x, y and z attributes).
      middle_point (Landmark):
        The middle landmark point (it needs to have x, y and z attributes).
      final_point (Landmark):
        The final landmark point (it needs to have x, y and z attributes).

    #### Returns
      np.array:
        An array checking if isLeft xy, zx and yz (respectively), if number is greater than zero (> 0), final point is to the left of the line (in that plan).
    '''
    isLeft_xy = cls.isLeft_XY(start_point, middle_point, final_point)
    isLeft_zx = cls.isLeft_ZX(start_point, middle_point, final_point)
    isLeft_yz = cls.isLeft_YZ(start_point, middle_point, final_point)
    # return {'xy': isLeft_xy, 'zx': isLeft_zx, 'yz': isLeft_yz}
    return np.array([isLeft_xy, isLeft_zx, isLeft_yz])


  # Mais usado nos braços, baseado no ponto final (se está para "cima" ou para "baixo"), uma mensagem é disponibilizada.
  @classmethod
  def is_up(cls, start_point, midle_point, end_point):
      # No mediapipe pose, as coordenadas tem o centro de referencia proximo ao centro da cintura, logo
      # o lado direito possui coordenada negativa para o x (-), mas o lado esquedo é positivo para x (x).
      # Para ver se um ponto está a esquerda do braço basta ver se o resultado é negativo (-) ou positivo (+).

      # Calcula em qual lado da reta está o último ponto, direita ou esquerda
      # e no caso, verifica se o ponto pulso está a direita ou a esquerda da linha antebraço
      arm_direction = cls.isLeft(start_point, midle_point, end_point)

      # Pensando no braço direito, primeiro vejo se está a esquerda, ou no caso, para cima,
      # depois verifico se a coordenada x é negativa, indicando que está no lado direito ou esquerdo
      up = True if arm_direction <= 0 else False

      if end_point[0] > 0:
          arm_direction *= -1
      else: 
          up = not up

      # Após as verificações retirna se está para cima
      return up, arm_direction
      pass


  @classmethod
  def affine_transformation (cls, base_model: np.array, input_pose: np.array) -> tuple[np.array, np.array]:
    '''
    Calculates the affine transformation for the input model.

    #### Parameters
      base_model (np.array):
        An array of the base model.
      input_pose (np.array)
        An array with the input pose model.

    #### Returns
      tuple[ np.array, np.array ]
      (input_transform , A)
        The input transformed to fit the model.


    Thanks bilgeckers from https://becominghuman.ai/
    https://becominghuman.ai/single-pose-comparison-a-fun-application-using-human-pose-estimation-part-2-4fd16a8bf0d3
    '''
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
    unpad = lambda x: x[:, :-1]
    base = pad(base_model)
    pose = pad(input_pose)
    A, res, rank, s = np.linalg.lstsq(pose, base)
    A[np.abs(A) < 1e-10] = 0
    transform = lambda x: unpad(np.dot(pad(x), A))
    input_transform = transform(input_pose)
    return input_transform, A
    pass # affine_transformation


  @classmethod
  def array_of_landmarks_to_array_array (cls, land_arr):
    for land in range(len(land_arr)):
      land_arr[land] = cls.landmark_to_array_default(land_arr[land])
    return land_arr
    pass


  @classmethod
  def analyse_torso_w_affine(cls, model_landmarks_result, input_landmarks_result) -> dict:
    '''
    Given a model landmarks result and a input landmarks result, analyse the torso, returning the angle and directors of the upper limbs.

    #### Parameters
      model_landmarks_result (landmark_results):
        Results of the MediaPipe pose detection for the model pose.
      input_landmarks_result (landmark_results):
        Results of the MediaPipe pose detection for the input pose.
    
    #### Results
      dict:
        Dictionary with the angles and directors for the limbs.
    '''
    # input_torso = input_landmarks_result.pose_world_landmarks[0][11: 17]
    # model_torso = model_landmarks_result.pose_world_landmarks[0][11: 17]
    # print(len(input_landmarks_result[0]))
    # print(len(model_landmarks_result[0]))
    input_torso = input_landmarks_result[0][11: 17]
    model_torso = model_landmarks_result[0][11: 17]

    input_torso = cls.array_of_landmarks_to_array_array(input_torso)
    model_torso = cls.array_of_landmarks_to_array_array(model_torso)

    input_torso = np.array(input_torso)
    model_torso = np.array(model_torso)

    input_transform, A_torso = cls.affine_transformation(model_torso, input_torso)

    right_arm_angle        = cls.angle_between_limbs(input_transform[1], input_transform[3], input_transform[5])
    right_forearm_angle    = cls.angle_between_limbs(input_transform[3], input_transform[1], input_transform[0])
    right_arm_director     = cls.analyse_limb(start_point_array = input_transform[3], final_point_array = input_transform[5])
    right_forearm_director = cls.analyse_limb(start_point_array = input_transform[1], final_point_array = input_transform[3])

    left_arm_angle        = cls.angle_between_limbs(input_transform[0], input_transform[2], input_transform[4])
    left_forearm_angle    = cls.angle_between_limbs(input_transform[2], input_transform[0], input_transform[1])
    left_arm_director     = cls.analyse_limb(start_point_array = input_transform[2], final_point_array = input_transform[4])
    left_forearm_director = cls.analyse_limb(start_point_array = input_transform[0], final_point_array = input_transform[2])

    shoulders_director = cls.analyse_limb(start_point_array = input_transform[0], final_point_array = input_transform[1])

    result = {
      'upper_limbs' : {
        'right' : {
          'arm': {
            'angle':  right_arm_angle,
            'director': right_arm_director,
          },
          'forearm': {
            'angle':  right_forearm_angle,
            'director': right_forearm_director,
          },
        },
        'left': {
          'arm': {
            'angle':  left_arm_angle,
            'director': left_arm_director,
          },
          'forearm': {
            'angle':  left_forearm_angle,
            'director': left_forearm_director,
          },
        },
      },
      'shoulders': shoulders_director,   # só diretor ? ou ang tambem com o eixo X?
      # 'pescoco': np.array([5, 5, 5]), # só diretor ? ou ang tambem com o eixo y?
      'torso_affine': A_torso
    }

    return result
    pass # analyse_torso_w_affine


  @classmethod # arm <-> forearm concertado
  def analyse_torso_wo_affine(cls, input_landmarks_result) -> dict:
    '''
    Given a model landmarks result and a input landmarks result, analyse the torso, returning the angle and directors of the upper limbs.

    #### Parameters
      model_landmarks_result (landmark_results):
        Results of the MediaPipe pose detection for the model pose.
      input_landmarks_result (landmark_results):
        Results of the MediaPipe pose detection for the input pose.
    
    #### Results
      dict:
        Dictionary with the angles and directors for the limbs.
    '''
    # input_torso = input_landmarks_result.pose_world_landmarks[0][11: 17]
    # print(len(input_landmarks_result[0]))
    input_torso = input_landmarks_result[0][11: 17]
    nose = input_landmarks_result[0][0]

    input_torso = cls.array_of_landmarks_to_array_array(input_torso)
    nose = cls.landmark_to_array_default(nose)

    input_torso = np.array(input_torso)
    
    # Right arm/limb analysis
    right_forearm_angle        = cls.angle_between_limbs(input_torso[1], input_torso[3], input_torso[5])
    right_arm_angle    = cls.angle_between_limbs(input_torso[3], input_torso[1], input_torso[0])
    right_forearm_director     = cls.analyse_limb(start_point_array = input_torso[3], final_point_array = input_torso[5])
    right_arm_director = cls.analyse_limb(start_point_array = input_torso[1], final_point_array = input_torso[3])

    # Left arm/limb analysis
    left_forearm_angle        = cls.angle_between_limbs(input_torso[0], input_torso[2], input_torso[4])
    left_arm_angle    = cls.angle_between_limbs(input_torso[2], input_torso[0], input_torso[1])
    left_forearm_director     = cls.analyse_limb(start_point_array = input_torso[2], final_point_array = input_torso[4])
    left_arm_director = cls.analyse_limb(start_point_array = input_torso[0], final_point_array = input_torso[2])

    # Shoulder analysis
    shoulders_director = cls.analyse_limb(start_point_array = input_torso[0], final_point_array = input_torso[1])

    # Neck analysis
    neck_base_x = (input_torso[0][0] + input_torso[1][0]) / 2
    neck_base_y = (input_torso[0][1] + input_torso[1][1]) / 2
    neck_base_z = (input_torso[0][2] + input_torso[1][2]) / 2

    neck_base = np.array([neck_base_x, neck_base_y, neck_base_z])

    neck_vector = cls.create_vector(neck_base, nose)
    neck_director = cls.director_angles(neck_vector)

    result = {
      'upper_limbs' : {
        'right' : {
          'arm': {
            'angle':  right_arm_angle,
            'director': right_arm_director,
          },
          'forearm': {
            'angle':  right_forearm_angle,
            'director': right_forearm_director,
          },
        },
        'left': {
          'arm': {
            'angle':  left_arm_angle,
            'director': left_arm_director,
          },
          'forearm': {
            'angle':  left_forearm_angle,
            'director': left_forearm_director,
          },
        },
      },
      'shoulders': shoulders_director,   # só diretor ? ou ang tambem com o eixo X?
      'neck': neck_director, # só diretor ? ou ang tambem com o eixo y?
    }

    return result
    pass # analyse_torso_w_affine


  @classmethod
  def compare_score(cls, input_dict, model_dict) -> float:
    '''Closer to 0 = better.'''
    acum_err = 0

    acum_err += abs(input_dict['upper_limbs']['right']['arm']['angle'] -  model_dict['upper_limbs']['right']['arm']['angle']) 
    # acum_err += abs(input_dict['upper_limbs']['right']['arm']['director'] -  model_dict['upper_limbs']['right']['arm']['director']).sum()
    acum_err += abs(input_dict['upper_limbs']['right']['arm']['director'] -  model_dict['upper_limbs']['right']['arm']['director']).mean()
    acum_err += abs(input_dict['upper_limbs']['right']['forearm']['angle'] -  model_dict['upper_limbs']['right']['forearm']['angle']) 
    # acum_err += abs(input_dict['upper_limbs']['right']['forearm']['director'] -  model_dict['upper_limbs']['right']['forearm']['director']).sum()
    acum_err += abs(input_dict['upper_limbs']['right']['forearm']['director'] -  model_dict['upper_limbs']['right']['forearm']['director']).mean()

    acum_err += abs(input_dict['upper_limbs']['left']['arm']['angle'] -  model_dict['upper_limbs']['left']['arm']['angle']) 
    # acum_err += abs(input_dict['upper_limbs']['left']['arm']['director'] -  model_dict['upper_limbs']['left']['arm']['director']).sum()
    acum_err += abs(input_dict['upper_limbs']['left']['arm']['director'] -  model_dict['upper_limbs']['left']['arm']['director']).mean() 
    acum_err += abs(input_dict['upper_limbs']['left']['forearm']['angle'] -  model_dict['upper_limbs']['left']['forearm']['angle']) 
    # acum_err += abs(input_dict['upper_limbs']['left']['forearm']['director'] -  model_dict['upper_limbs']['left']['forearm']['director']).sum() 
    acum_err += abs(input_dict['upper_limbs']['left']['forearm']['director'] -  model_dict['upper_limbs']['left']['forearm']['director']).mean() 

    return acum_err
    pass


  @classmethod
  def check_limb(cls, input_limb, model_limb, model_std):

    diff = abs(input_limb -  model_limb) 
    aceptable = True if diff < model_std else False

    if (diff > (3*model_std)) : diff = (3*model_std)
    if (diff < model_std) : diff = diff/2

    max_diff = 3*model_std

    # aceptable = False
    # if (input_limb < (model_limb + model_std)) and (input_limb > (model_limb - model_std)):
    #   aceptable = True

    return [ aceptable, diff , max_diff]

    pass # check_limb


  @classmethod
  def check_limb_direction_1(cls, input_limb, model_limb, model_std):
    # Preciso comparar angulos, se > 90 está em um sentido, se < 90 em outro sentido
    # Mas e se o angulo input for 95 e o desvio padrão for 8?
    #   1 - verificar se +- a DP vai deixar nos dois sentidos, nesse caso deixa tudo true pois estã muito perto de 90
    #       e o resto compara normal
    #   2 - ver a diferença/ distancia para o limite? se o input +- o DP entrar no campo do modelo +- DP, considera verdadeiro?
    #       (isso n é = a fazer 2*DP e ver se entra?)

    if ((model_limb + model_std) > 90) and ((model_limb - model_std) < 90):
      return [True, 0, 0]
      pass

    model_direction = model_limb > 90
    input_direction = input_limb > 90

    return [model_direction == input_direction , 0, 0]

    pass # check_limb


  @classmethod
  def check_pose(cls, input_dict, pose_data_dict):
    '''
    input_dict = {
      'upper_limbs' : {
        'right' : {
          'arm': {
            'angle':  right_arm_angle,
            'director': right_arm_director,
          },
          'forearm': {
            'angle':  right_forearm_angle,
            'director': right_forearm_director,
          },
        },
        'left': {
          'arm': {
            'angle':  left_arm_angle,
            'director': left_arm_director,
          },
          'forearm': {
            'angle':  left_forearm_angle,
            'director': left_forearm_director,
          },
        },
      },
      'shoulders': shoulders_director,   # só diretor ? ou ang tambem com o eixo X?
      'neck': neck_director, # só diretor ? ou ang tambem com o eixo y?
      # 'torso_affine': A_torso
    }


    pose_data_dict = {
      'upper_limbs' : {
        'right' : {
          'arm': {
            'angle'         : right_arm_angle,
            'angle_std'     : right_arm_angle_std
            'director_i'    : right_arm_director_i,
            'director_i_std': right_arm_director_i_std,
            'director_j'    : right_arm_director_j,
            'director_j_std': right_arm_director_j_std,
            'director_k'    : right_arm_director_k,
            'director_k_std': right_arm_director_k_std,
          },
          'forearm': {
            'angle'         : right_forearm_angle,
            'angle_std'     : right_forearm_angle_std
            'director_i'    : right_forearm_director_i,
            'director_i_std': right_forearm_director_i_std,
            'director_j'    : right_forearm_director_j,
            'director_j_std': right_forearm_director_j_std,
            'director_k'    : right_forearm_director_k,
            'director_k_std': right_forearm_director_k_std,
          },
        },
        'left' : {
          'arm': {
            'angle'         : left_arm_angle,
            'angle_std'     : left_arm_angle_std
            'director_i'    : left_arm_director_i,
            'director_i_std': left_arm_director_i_std,
            'director_j'    : left_arm_director_j,
            'director_j_std': left_arm_director_j_std,
            'director_k'    : left_arm_director_k,
            'director_k_std': left_arm_director_k_std,
          },
          'forearm': {
            'angle'         : left_forearm_angle,
            'angle_std'     : left_forearm_angle_std
            'director_i'    : left_forearm_director_i,
            'director_i_std': left_forearm_director_i_std,
            'director_j'    : left_forearm_director_j,
            'director_j_std': left_forearm_director_j_std,
            'director_k'    : left_forearm_director_k,
            'director_k_std': left_forearm_director_k_std,
          },
        },
      },
      'shoulders': {
        director_i     : shoulders_director_i,
        director_i_std : shoulders_director_i_std,
        director_j     : shoulders_director_j,
        director_j_std : shoulders_director_j_std,
        director_k     : shoulders_director_k,
        director_k_std : shoulders_director_k_std,
      },   
      'neck': {
        director_i     : neck_director_i,
        director_i_std : neck_director_i_std,
        director_j     : neck_director_j,
        director_j_std : neck_director_j_std,
        director_k     : neck_director_k,
        director_k_std : neck_director_k_std,
      }
    }
    '''
    
    result = {
      'upper_limbs' : {
        'right' : {
          'arm': {
            'angle'      : cls.check_limb(input_dict['upper_limbs']['right']['arm']['angle']      , pose_data_dict['upper_limbs']['right']['arm']['angle']     , pose_data_dict['upper_limbs']['right']['arm']['angle_std']), 
            'director_i' : cls.check_limb(input_dict['upper_limbs']['right']['arm']['director'][0], pose_data_dict['upper_limbs']['right']['arm']['director_i'], pose_data_dict['upper_limbs']['right']['arm']['director_i_std']),
            'director_j' : cls.check_limb(input_dict['upper_limbs']['right']['arm']['director'][1], pose_data_dict['upper_limbs']['right']['arm']['director_j'], pose_data_dict['upper_limbs']['right']['arm']['director_j_std']),
            'director_k' : cls.check_limb(input_dict['upper_limbs']['right']['arm']['director'][2], pose_data_dict['upper_limbs']['right']['arm']['director_k'], pose_data_dict['upper_limbs']['right']['arm']['director_k_std']),
            # 'director_i' : cls.check_limb_direction_1(input_dict['upper_limbs']['right']['arm']['director'][0], pose_data_dict['upper_limbs']['right']['arm']['director_i'], pose_data_dict['upper_limbs']['right']['arm']['director_i_std']),
            # 'director_j' : cls.check_limb_direction_1(input_dict['upper_limbs']['right']['arm']['director'][1], pose_data_dict['upper_limbs']['right']['arm']['director_j'], pose_data_dict['upper_limbs']['right']['arm']['director_j_std']),
            # 'director_k' : cls.check_limb_direction_1(input_dict['upper_limbs']['right']['arm']['director'][2], pose_data_dict['upper_limbs']['right']['arm']['director_k'], pose_data_dict['upper_limbs']['right']['arm']['director_k_std']),
          },
          'forearm': {
            'angle'      : cls.check_limb(input_dict['upper_limbs']['right']['forearm']['angle']      , pose_data_dict['upper_limbs']['right']['forearm']['angle']     , pose_data_dict['upper_limbs']['right']['forearm']['angle_std']),
            'director_i' : cls.check_limb(input_dict['upper_limbs']['right']['forearm']['director'][0], pose_data_dict['upper_limbs']['right']['forearm']['director_i'], pose_data_dict['upper_limbs']['right']['forearm']['director_i_std']),
            'director_j' : cls.check_limb(input_dict['upper_limbs']['right']['forearm']['director'][1], pose_data_dict['upper_limbs']['right']['forearm']['director_j'], pose_data_dict['upper_limbs']['right']['forearm']['director_j_std']),
            'director_k' : cls.check_limb(input_dict['upper_limbs']['right']['forearm']['director'][2], pose_data_dict['upper_limbs']['right']['forearm']['director_k'], pose_data_dict['upper_limbs']['right']['forearm']['director_k_std']),
          },
        },
        'left' : {
          'arm': {
            'angle'      : cls.check_limb(input_dict['upper_limbs']['left']['arm']['angle']      , pose_data_dict['upper_limbs']['left']['arm']['angle']     , pose_data_dict['upper_limbs']['left']['arm']['angle_std']), 
            'director_i' : cls.check_limb(input_dict['upper_limbs']['left']['arm']['director'][0], pose_data_dict['upper_limbs']['left']['arm']['director_i'], pose_data_dict['upper_limbs']['left']['arm']['director_i_std']),
            'director_j' : cls.check_limb(input_dict['upper_limbs']['left']['arm']['director'][1], pose_data_dict['upper_limbs']['left']['arm']['director_j'], pose_data_dict['upper_limbs']['left']['arm']['director_j_std']),
            'director_k' : cls.check_limb(input_dict['upper_limbs']['left']['arm']['director'][2], pose_data_dict['upper_limbs']['left']['arm']['director_k'], pose_data_dict['upper_limbs']['left']['arm']['director_k_std']),
          },
          'forearm': {
            'angle'      : cls.check_limb(input_dict['upper_limbs']['left']['forearm']['angle']      , pose_data_dict['upper_limbs']['left']['forearm']['angle']     , pose_data_dict['upper_limbs']['left']['forearm']['angle_std']),
            'director_i' : cls.check_limb(input_dict['upper_limbs']['left']['forearm']['director'][0], pose_data_dict['upper_limbs']['left']['forearm']['director_i'], pose_data_dict['upper_limbs']['left']['forearm']['director_i_std']),
            'director_j' : cls.check_limb(input_dict['upper_limbs']['left']['forearm']['director'][1], pose_data_dict['upper_limbs']['left']['forearm']['director_j'], pose_data_dict['upper_limbs']['left']['forearm']['director_j_std']),
            'director_k' : cls.check_limb(input_dict['upper_limbs']['left']['forearm']['director'][2], pose_data_dict['upper_limbs']['left']['forearm']['director_k'], pose_data_dict['upper_limbs']['left']['forearm']['director_k_std']),
          },
        },
      },
      'shoulders': {
        'director_i' : cls.check_limb(input_dict['shoulders'][0], pose_data_dict['shoulders']['director_i'], pose_data_dict['shoulders']['director_i_std']),
        'director_j' : cls.check_limb(input_dict['shoulders'][1], pose_data_dict['shoulders']['director_j'], pose_data_dict['shoulders']['director_j_std']),
        'director_k' : cls.check_limb(input_dict['shoulders'][2], pose_data_dict['shoulders']['director_k'], pose_data_dict['shoulders']['director_k_std']),
      },   
      'neck': {
        'director_i' : cls.check_limb(input_dict['neck'][0], pose_data_dict['neck']['director_i'], pose_data_dict['neck']['director_i_std']),
        'director_j' : cls.check_limb(input_dict['neck'][1], pose_data_dict['neck']['director_j'], pose_data_dict['neck']['director_j_std']),
        'director_k' : cls.check_limb(input_dict['neck'][2], pose_data_dict['neck']['director_k'], pose_data_dict['neck']['director_k_std']),
      }
    }

    return result

    pass


  pass