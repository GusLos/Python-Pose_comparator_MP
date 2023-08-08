from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import mediapipe as mp
import numpy as np
import cv2
import os


# class MPPose(metaclass=SingletonMeta):
class MPPose():
  '''
  Class that uses Mediapipe pose landmark detection to track the pose.\n
  It is designed to make easier the use of mediapipe pose.\n
  In 'live_stream' mode you need to define a function to be executed in each frame of the stream, asynchronous. 
  See `MPPose.live_stream_method_explained()` to get an idea of what your function shuold have to work.\n\n
  For more information, visit the mediapipe documentation.
  '''

  def __init__(self, model_path: str, running_mode: str, live_stream_method = None, show_live_stream_result: bool = True) -> None:
    '''
    Generates an instance of MPPose

    #### Parameters
        model_path (str):
            Path to the model that will be used. You can download a model in Mediapipe pose documentation.
        running_mode (str):
            Mode that the model will be working. Choose between 'image', 'video' and 'live_stream'
        live_stream_method (function):
            Function/Method that will be executed with each frame of the stream.
        show_live_stream_result (bool):
            If the default live_stream_function (that show the image/frame with the results) should be executed.
    '''
    self.model_path = model_path
    self.running_mode = running_mode
    self.live_stream_method = live_stream_method
    self.show_live_stream = show_live_stream_result
    self._mp_pose_init()
    pass

  def live_stream_method_explained(self, result: mp.tasks.vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
      ''' 
      This method does nothing, it is only a short example of how your method should look like.\n
      For more information, visit the MediaPipe Pose landmark detection documentation.\n
      \n\n
      Your method must accept 3 parameters, it doesn't matter if it will be used or not. 
      It doesn't need to return anything, the method will work asynchronously\n
      ### Parameters
            result (mediapipe.tasks.vision.PoseLandmarkerResult) :
                The analysis result, one way to access it: result.pose_landmarks[0][0].x (it will get the x coordinate of the nose)
            output_image (mp.Image) : 
                Not sure...
            timestamp_ms (int) : 
                Timestamp of current frame
      '''
      print('pose landmarker result: {}'.format(result))
    
  def _live_stream_method(self, result: mp.tasks.vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    '''
    Method that will be used in live stream mode, asynchronously.
    It executes custom method stored in class parameter.
    Used when creating the model, in result_callback.

    Feel free to overwrite this method with something more suitable.
    '''

    if self.show_live_stream:
      try:
        # os.system('cls')
        # print('I can see you.')
        annotated_image = self.draw_landmarks_on_image(output_image, result)
        cv2.imshow('frame', annotated_image )
        if cv2.waitKey(1) == ord('q'):
          os.abort()
          pass # if cv2.waitKey(1) == ord('q')
      except:
        print("I can't see you.")

    if self.live_stream_method:
      try:
        self.live_stream_method(result, output_image, timestamp_ms)
      except:
        print('Something went wrong in your live stream method.\n    See/Read `MPPose.live_stream_method_explained()`')
      pass # if self.live_stream_method

    pass # live_stream_method

  def _mp_pose_init(self):
    '''
    Initializes the model based on class parameters. Should not be executed before __init__.
    It should be automatically executed when at least one class parameter change.
    '''
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

    options = { 
      'image': PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=self.model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        output_segmentation_masks=True),
      'video':  PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=self.model_path),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        output_segmentation_masks=True),
      'live_stream': PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=self.model_path),
        running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
        result_callback=self._live_stream_method)
    }

    self.detector = mp.tasks.vision.PoseLandmarker.create_from_options(options[self.running_mode])
    pass # def _options  

  def can_run_image(self, image_path: str) -> bool:
    '''
    Method to verify if, acording to running_mode ('image'), the parameters exists (image_path).

    ### Parameters
        image_path (str):
            Path to image that will be analysed.
    
    ### Returns
        bool:
            True if model has everything to work with. False if something is missing or the running_mode is not 'image'.
    '''
    return (image_path and (self.running_mode == 'image'))
    pass # can_run_image

  def can_run_video_or_stream(self, cap: cv2.VideoCapture, frame: np.ndarray) -> bool:
    '''
    Method to verify if, acording to running_mode ('video' or 'live_stream'), the parameters exists (cap and frame).

    ### Parameters
        cap (cv2.VideoCapture):
            Something like: `cap = cv2.VideoCapture(video_path)`, to access the video information.
        frame (numpy.ndarray):
            The "image" (frame at the moment) to be analysed.
    
    ### Returns
        bool:
            True if model has everything to work with. False if something is missing or the running_mode is not 'video' or 'live_stream'.
    '''
    return ((cap is not None) and (frame is not None)) and ( (self.running_mode == 'video') or (self.running_mode == 'live_stream') )
    pass # can_run_video_or_stream

  def detect_pose(self, image_path: str = None, cap: cv2.VideoCapture = None, frame: np.ndarray = None) -> tuple(()):
    '''
    Detects pose in image/frame according to the running mode. If 'image', image_path is required. If 'video' or 'live_stream', cap and frame are required.

    ### Parameters
        image_path (str):
            Path to the image that will be analysed. Required if running_mode is 'image'.
        cap (cv2.VideoCapture):
            Something like: `cap = cv2.VideoCapture(video_path)`
        frame (numpy.ndarray):
            The "image" that will be analysed.

    ### Returns
        mp_image (mp.Image):
            Image/Frame in mediapipe format.(?)
        detection_result (mp.tasks.python.vision.pose_landmarker.PoseLandmarkerResult):
            The result of analysis. `PoseLandmarkerResult(pose_landmarks=[[]], pose_world_landmarks=[[]], segmentation_masks=[])`.
            To access try: `detection_result.pose_landmarks[0][x]`, where x go from 0 to 32, see mediapipe pose documentation for more information.
    '''
    message = 'Missing arguments for defined running mode.'
    assert self.can_run_image(image_path) or self.can_run_video_or_stream(cap, frame) , message
    mp_method = {
            'image':  self._detect_pose_image(image_path)       if self.running_mode == 'image'       else None,
            'video':  self._detect_pose_video(cap, frame)       if self.running_mode == 'video'       else None,
      'live_stream':  self._detect_pose_live_stream(cap, frame) if self.running_mode == 'live_stream' else None
    }
    return mp_method[self.running_mode]
    pass # _generate_mp_image

  def _detect_pose_image(self, image_path: str) -> tuple(()):
    '''
    Method to detect pose in image, using the model defined.

    ### Parameters
        image_path (str) :
            Path to the image that will be analysed.
    
    ### Returns
        tuple((mp_image, detection_result))
        mp_image (mp.Image):
            Image in mediapipe format.
        detection_result (mediapipe.tasks.python.vision.pose_landmarker.PoseLandmarkerResult):
            A kind of dictionary object with the results of the analysis. It has `.pose_landmarks`, `.pose_world_landmarks` and `.segmentation_masks`
    '''
    mp_image = mp.Image.create_from_file(image_path)
    detection_result = self.detector.detect(mp_image)
    return mp_image, detection_result
    pass

  def _detect_pose_video(self, cap: cv2.VideoCapture, frame: np.ndarray) -> tuple(()):
    '''
    Method to detect pose in video, using the model defined.

    ### Parameters
        cap (cv2.VideoCapture) :
            Information about the video. Something like: `cap = cv2.VideoCapture(video_path)`
        frame (np.ndarray) :
            The "image" part of the video. 
    
    ### Returns
        tuple((mp_image, detection_result))\n
        mp_image (mp.Image):
            Image in mediapipe format.
        detection_result (mediapipe.tasks.python.vision.pose_landmarker.PoseLandmarkerResult):
            A kind of dictionary object with the results of the analysis. It has `.pose_landmarks`, `.pose_world_landmarks` and `.segmentation_masks`
    '''
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = self.detector.detect_for_video(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))
    return mp_image, detection_result
    pass

  def _detect_pose_live_stream(self, cap: cv2.VideoCapture, frame: np.ndarray) -> None:
    '''
    Method to detect pose in live stream (webcam), using the model defined. 
    It will execute an asynchronous method `live_stream_function`, so it can uses the results.
    `live_stream_function` can be defined and changed, see `MPPose.live_stream_function_explained` or Mediapipe pose documentation;

    ### Parameters
        cap (cv2.VideoCapture) :
            Information about the video. Something like: `cap = cv2.VideoCapture(0)`
        frame (np.ndarray) :
            The "image" part of the video. 
    '''
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    cframe = cap.get(cv2.CAP_PROP_POS_MSEC)
    self.detector.detect_async(mp_image, int(cframe))
    pass

  def draw_landmarks_on_image(self, rgb_image: mp.Image, detection_result) -> np.ndarray:
    '''
    Draws the results on top of the image. Uses the both results of `MPPose.detect_pose()` method.\n
    A simple example of use:
    ```
    result = mppose.detect_pose(image_path)
    annotated_image = mppose.draw_landmarks_on_image(result[0], result[1])
    ```

    ### Parameters
        rgb_image (mediapipe.python._framework_bindings.image.Image) : 
            MediaPipe image, result of the method `mediapipe.Image()`
        detection_result (mediapipe.tasks.python.vision.pose_landmarker.PoseLandmarkerResult) :
            PoseLandmarkerResult that contains: pose_landmarks, pose_world_landmarks and segmentation_masks. 
     \t`PoseLandmarkerResult(pose_landmarks=[[...]], pose_world_landmarks=[[...]], segmentation_masks=[...])`
    
            
    ### Returns
        annotated_image (numpy.ndarray):
            Original image, but with the landmarks on it. It is ready to be shown.
    '''
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
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style())

    return annotated_image

  def set_model_path(self, new_model_path: str) -> None:
    '''
    Change the current model

    ### Parameters
        new_model_path (str) : 
            Path to the model that will be used
    '''
    self.model_path = new_model_path
    self._mp_pose_init()
    pass

  def set_modo_operacao(self, new_running_mode: str) -> None:
    '''
    Change current running mode

    ### Parameters
        new_running_mode (str) : 
            The desired running mode (image, video, live_stream)
    '''
    self.running_mode = new_running_mode
    self._mp_pose_init()
    pass

  def set_live_stream_method(self, live_stream_method) -> None:
    '''
    Change current live stream function to execute

    ### Parameters
        live_stream_function (function) :
            Custom function to be executed asyncronys with each frame of the live stream. 
            Check `MPPose.live_stream_function_explained()` for more information.
    '''
    self.live_stream_method = live_stream_method
    self._mp_pose_init()
    pass # set_live_stream_method

  pass

if __name__ == '__main__':
  
  pass