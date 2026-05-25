from .street_view_feature import StreetViewFeatureExtractor
from typing import Optional, List
from ultralytics import YOLO
import os
import requests
import cv2
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

class StreetStallSegmentator(StreetViewFeatureExtractor):

    DOWNLOAD_URL = "https://huggingface.co/walup/street-stall/resolve/main/exp-5.pt"
    DEFAULT_MODEL_CLASS_NAME = "retail-stall"

    def __init__(
            self,
            model_path: Optional[str] = None, 
            street_stall_class_id = None,
            threshold = 0.5,
            save_path:str = "./models/retail_stalls.pt",
            verbose:bool = True
            ):
        
        self.threshold = threshold
        self.model_path = model_path
        self.street_stall_class_id = street_stall_class_id

        self.model:Optional[YOLO] = None
        if(self.model_path is not None and self.street_stall_class_id is not None):
            self.model = YOLO(model_path)
        
        else:
            #If the model path is not provided we download the default model.
            self.download_model(save_path = save_path, verbose = verbose)


    def download_model(
            self,
            save_path:str = None,
            verbose:bool = True):
        

        if(not os.path.exists(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            response = requests.get(self.DOWNLOAD_URL, stream = True)
            if(response.status_code == 200):
                with open(save_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                raise("The model could not be downloaded.")
            
            if(verbose):
                print(f"Model downloaded to {save_path}")

            #Load the model
            self.model = YOLO(save_path)
            self.id_dictionary = {
            name: idx for idx, name in self.model.names.items()
            }
            self.street_stall_class_id = self.id_dictionary.get(self.DEFAULT_MODEL_CLASS_NAME, None)

            if(verbose):
                print(f"Informal retail class Id: {self.street_stall_class_id}")

        #Case where you already have a model downloaded. 
        else:
            self.model = YOLO(save_path)
            self.id_dictionary = {
                name: idx for idx, name in self.model.names.items()
            }
            self.street_stall_class_id = self.id_dictionary.get(self.DEFAULT_MODEL_CLASS_NAME, None)

            if(verbose):
                print(f"Informal retail class Id: {self.street_stall_class_id}")


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
    


        
    

    def get_masks(
            self, 
            image:np.ndarray, 
            confidence_threshold:float = 0.5):
        
        if self.model is None:
            raise RuntimeError("Model is not loaded. Cannot extract masks.")
            
        preprocessed_image = self.preprocess_image(image)
        results = self.model(preprocessed_image, verbose=False, overlap_mask=True)

        masks: List[np.ndarray] = []
        final_confidences: List[float] = []
        if results[0].masks is not None:
            confidences = results[0].boxes.conf.numpy()
            class_ids = results[0].boxes.cls.numpy()
            for mask, class_id, conf in zip(results[0].masks.data, class_ids, confidences):
                if conf >= confidence_threshold and int(class_id) == self.street_stall_class_id:
                    masks.append(np.array(mask))
                    final_confidences.append(conf)

        return masks, final_confidences
    
    
    def compute(self,
                image:np.ndarray,
                confidence_threshold:float = 0.5, 
                debug:bool = False):
        
        masks, confidences = self.get_masks(
            image = image,
            confidence_threshold=confidence_threshold
        )

        combined_mask: np.ndarray = np.logical_or.reduce(np.array(masks)).astype(np.uint8)
        #Let's add a boolean that draws the segmented
        if debug:
            segmented_image = (combined_mask[:, :, np.newaxis] * image).astype(np.uint8)
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Original Image")
            plt.imshow(image)
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.title("Segmented Stall")
            plt.imshow(segmented_image)
            plt.axis("off")
            plt.show()

        #Return the number of detected instances. 
        return len(masks)
    


