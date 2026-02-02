import os
from typing import Dict, Optional, Callable, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import requests
from alivestreets.features.street_view_feature import StreetViewFeatureExtractor
from PIL import Image, ImageOps
import cv2

class FacadeFeatureExtractor(StreetViewFeatureExtractor):

    def __init__(
        self,
        method: str = "ml",
        model_path: Optional[str] = None,
        facade_feature_id_dictionary: Optional[Dict[str, int]] = None,
        threshold: float = 0.6
    ) -> None:
        self.method: str = method
        self.model_path: Optional[str] = model_path
        self.facade_feature_id_dictionary:Optional[Dict[str,int]] = facade_feature_id_dictionary
        self.threshold: float = threshold

        self.model: Optional[YOLO] = None

        # Do not raise error yet — let user call `download_model()` later
        loaded = False
        if self.method == "ml" and model_path and facade_feature_id_dictionary is not None:
            self.model = YOLO(model_path)
            loaded = True
        if not loaded:
            print("Warning: Model not loaded. Please call download_model() to download the model.")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Ensures the image matches the Roboflow training environment:
        1. Fixes Orientation (Auto-Orient)
        2. Ensures RGB color space (Roboflow standard)
        """
        # If the input is from OpenCV (BGR), convert to RGB
        # If it's already RGB, this is still safe to ensure consistency
        if isinstance(image, np.ndarray):
            # Assuming input is BGR from cv2.imread or GSV collector
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(image_rgb)
        else:
            pil_img = Image.open(image)

        # Apply Auto-Orient (Critical for mobile/GSV metadata)
        pil_img = ImageOps.exif_transpose(pil_img)
        
        return np.array(pil_img)

    def download_model(
        self,
        save_path = "facades_model.pt",
        verbose: bool = True
    )->None:
    
        """
        Downloads the façade characteristics model. Note that the
        model works best with views directed towards building façades or sidewalk views. The results of the segmentation 
        will return also the confidence of the detections. 

        Parameters
        ----------
        save_path
            Path where the .pt file will be downloaded.
        """
        #Non-failure case
        
        URL = "https://huggingface.co/walup/facades_segmentation/resolve/main/weights.pt"
        #We will only download if the model is not already present
        
        if not os.path.exists(save_path):
            response = requests.get(URL, stream=True)
            if response.status_code == 200:
                with open(save_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("Model downloaded.")
            else:
                raise Exception("Failed to download the model.")
            
            self.model = YOLO(save_path)
            self.model_path = save_path

            self.facade_feature_id_dictionary = {
            name: idx for idx, name in self.model.names.items()
            }

            if verbose:
                print("Facade feature ID dictionary:")
                for name, idx in self.facade_feature_id_dictionary.items():
                    print(f"{name}: {idx}")

        else:
            self.model = YOLO(save_path)
            self.model_path = save_path

            self.facade_feature_id_dictionary = {
            name: idx for idx, name in self.model.names.items()
            }
            if verbose:
                print("Model already exists. Loaded existing model.")
                print("Facade feature ID dictionary:")
                for name, idx in self.facade_feature_id_dictionary.items():
                    print(f"{name}: {idx}")
    

    def get_masks(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.6,
        class_name = "",
        verbose: bool = False,
        overlap_mask: bool = True
    ) ->Tuple[List[np.ndarray], List[float]]:
        """
        Returns a list of array masks for the specified class. If segmentation is well conducted 
        different masks correspond to disconnected trees or vegetation indexes.

        Parameters
        ----------
        image
            input image
        
        confidence_threshold
            threshold that will be used to discriminate which detected masks correspond
            to the requested class instances. 

        Returns
        ------------
        masks
            list of masks, where each mask is a binary matrix. 
        """

        if self.method == "ml":
            if(self.facade_feature_id_dictionary is None or self.facade_feature_id_dictionary.get(class_name, None) is None):
                raise RuntimeError("A valid class dictionary must be provided to extract the masks.")
            if self.model is None:
                raise RuntimeError("Model is not loaded. Cannot extract masks.")
            

            processed_image = self.preprocess_image(image)
            
            results = self.model(
                processed_image, 
                imgsz=640, 
                conf=confidence_threshold,
                verbose=verbose, 
                overlap_mask=overlap_mask
            )
            requested_id = self.facade_feature_id_dictionary.get(class_name, None)

            masks: List[np.ndarray] = []
            final_confidences: List[float] = []
            if results[0].masks is not None:
                confidences = results[0].boxes.conf.numpy()
                class_ids = results[0].boxes.cls.numpy()
                for mask, class_id, conf in zip(results[0].masks.data, class_ids, confidences):
                    if conf >= confidence_threshold and int(class_id) == requested_id:
                        masks.append(np.array(mask))
                        final_confidences.append(conf)

            return masks, final_confidences

        else:
            raise ValueError(f"Unsupported method '{self.method}'")
    


    def compute(
    self,
    input_feature_names: List[str],
    image: np.ndarray,
    confidence_threshold: float = 0.6,
    operation: Optional[Callable[..., Any]] = None, 
    ) -> Any:
        """
        Computes the façade features for the given image. If `operation` is provided, it will be called with the
        `FacadeFeatureExtractor` instance, the image, the input feature names, and the confidence threshold.
        If `operation` is None, it will return a dictionary with the instance counts and metadata.

        Parameters
        ----------
        input_feature_names
            List of feature names used to compute the result. Each of them must correspond to a name in the
            facade_feature_id_dictionary.
        image
            Input image as a numpy array.
        
        confidence_threshold
            Threshold for filtering the masks based on confidence scores
        
        operation
            Optional callable operation that takes the extractor, image, input feature names, and confidence threshold.
            If provided, it will be used to compute the result instead of the default behavior, which obtains instance
            counts and metadata.
        """
        if operation is None:

            result = {}
            for class_name in input_feature_names:
                masks, metadata = self.get_masks(image, class_name=class_name, confidence_threshold=confidence_threshold)
                result[class_name] = (len(masks), metadata)
            return result
        else:
            return operation(self, image, input_feature_names, confidence_threshold)










    
