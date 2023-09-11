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
    input_torso = input_landmarks_result[0][11: 17]
    model_torso = model_landmarks_result[0][11: 17]

    input_torso = cls.array_of_landmarks_to_array_array(input_torso)
    model_torso = cls.array_of_landmarks_to_array_array(model_torso)

    input_torso = np.array(input_torso)
    model_torso = np.array(model_torso)

    input_transform, A_torso = cls.affine_transformation(model_torso, input_torso)

    # TESTAR: affine antes de selecionar X affine depois de selecionar
    
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

  pass