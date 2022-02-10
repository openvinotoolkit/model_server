import cv2
from queue import Queue
from logger import get_logger

logger = get_logger(__name__)

class Postprocessor:

    def __init__(self, enable_visualization):
        self.visualize = enable_visualization
        self.postprocessed_frames_queue = Queue()
        self.postprocessing_routines = {}

    def add_postprocessing_routine(self, name, function):
        self.postprocessing_routines[name] = function

    def _create_frame_with_predictions(self, frame, prediction_result):
        CLASSES = ["None", "Pedestrian", "Vehicle", "Bike", "Other"]
        COLORS = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 128)]
        CONFIDENCE_THRESHOLD = 0.75

        for batch, data in enumerate(prediction_result):
            pred = data[0]
            for values in enumerate(pred):
                # tuple
                index = values[0]
                l_pred = values[1]

                # actual predictions
                img_id = l_pred[0]
                label = l_pred[1]
                conf = l_pred[2]
                x_min = l_pred[3]
                y_min = l_pred[4]
                x_max = l_pred[5]
                y_max = l_pred[6]

                # preventing any wrong array indexing (for RMNet)
                if label > 4:
                    # Unsupported class label detected. Change to `other`.
                    label = 4

                # Do you want confidence level to be passed from command line?
                if img_id != -1 and conf >= CONFIDENCE_THRESHOLD:
                    #print("Detected an object")
                    height, width = frame.shape[0:2]
                    #print("height: {} ; width: {} ;".format(height, width))
                    #print("x_min: {} ; y_min: {} ; x_max: {} ; y_max: {} ;".format(x_min, y_min, x_max, y_max))
                    #print("x_min: {} ; y_min: {} ; x_max: {} ; y_max: {} ;".format(int(width * x_min), int(height * y_min),
                    #      int(width * x_max), int(height * y_max)))
                    # draw the bounding boxes on the frame
                    cv2.rectangle(frame, (int(width * x_min), int(height * y_min)),
                        (int(width * x_max), int(height * y_max)), COLORS[int(label)], 1)
                    cv2.putText(frame, str(CLASSES[int(label)]), (int(width * x_min)-10,
                        int(height * y_min)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        COLORS[int(label)], 1)
        return frame

    def postprocess(self, frame, prediction_result):
        if self.visualize:
            frame = self._create_frame_with_predictions(frame, prediction_result)
            self.postprocessed_frames_queue.put(frame)
            logger.debug("Frame with inference vizualized put into buffer. Current queue size: {}".format(self.postprocessed_frames_queue.qsize()))
        for _, function in self.postprocessing_routines.items():
            function(prediction_result)
        logger.debug("Frame post processed")

    def get_postprocessed_frame(self):
        frame = self.postprocessed_frames_queue.get()
        logger.debug("Frame with inference visualized pulled from buffer")
        return True, frame
        
