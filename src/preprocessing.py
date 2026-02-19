"""
Image Preprocessing Module
Handles all image preprocessing operations for defect detection
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Handles image preprocessing operations including:
    - Loading and validation
    - Noise reduction
    - Contrast enhancement
    - Edge detection
    - Morphological operations
    """
    
    def __init__(self, target_size: Tuple[int, int] = (640, 480)):
        """
        Initialize preprocessor
        
        Args:
            target_size: Target image dimensions (width, height)
        """
        self.target_size = target_size
        logger.info(f"ImagePreprocessor initialized with target size: {target_size}")
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load and validate image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Loaded image as numpy array or None if failed
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            logger.info(f"Image loaded successfully: {image.shape}")
            return image
        
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size
        
        Args:
            image: Input image
            
        Returns:
            Resized image
        """
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
    
    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale
        
        Args:
            image: Input BGR image
            
        Returns:
            Grayscale image
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def denoise(self, image: np.ndarray, method: str = 'gaussian') -> np.ndarray:
        """
        Apply denoising to image
        
        Args:
            image: Input image
            method: Denoising method ('gaussian', 'bilateral', 'median')
            
        Returns:
            Denoised image
        """
        if method == 'gaussian':
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif method == 'bilateral':
            return cv2.bilateralFilter(image, 9, 75, 75)
        elif method == 'median':
            return cv2.medianBlur(image, 5)
        else:
            logger.warning(f"Unknown denoising method: {method}, using Gaussian")
            return cv2.GaussianBlur(image, (5, 5), 0)
    
    def enhance_contrast(self, image: np.ndarray, method: str = 'clahe') -> np.ndarray:
        """
        Enhance image contrast
        
        Args:
            image: Input grayscale image
            method: Enhancement method ('clahe', 'histogram')
            
        Returns:
            Contrast-enhanced image
        """
        if method == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
        elif method == 'histogram':
            return cv2.equalizeHist(image)
        else:
            return image
    
    def detect_edges(self, image: np.ndarray, method: str = 'canny') -> np.ndarray:
        """
        Detect edges in image
        
        Args:
            image: Input grayscale image
            method: Edge detection method ('canny', 'sobel', 'laplacian')
            
        Returns:
            Edge-detected image
        """
        if method == 'canny':
            return cv2.Canny(image, 50, 150)
        elif method == 'sobel':
            sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            return np.uint8(np.sqrt(sobelx**2 + sobely**2))
        elif method == 'laplacian':
            return cv2.Laplacian(image, cv2.CV_64F)
        else:
            return cv2.Canny(image, 50, 150)
    
    def apply_morphology(self, image: np.ndarray, operation: str = 'closing', 
                        kernel_size: int = 5) -> np.ndarray:
        """
        Apply morphological operations
        
        Args:
            image: Input binary image
            operation: Operation type ('erosion', 'dilation', 'opening', 'closing')
            kernel_size: Size of morphological kernel
            
        Returns:
            Processed image
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        if operation == 'erosion':
            return cv2.erode(image, kernel, iterations=1)
        elif operation == 'dilation':
            return cv2.dilate(image, kernel, iterations=1)
        elif operation == 'opening':
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif operation == 'closing':
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        else:
            return image
    
    def threshold_image(self, image: np.ndarray, method: str = 'otsu') -> np.ndarray:
        """
        Apply thresholding to image
        
        Args:
            image: Input grayscale image
            method: Thresholding method ('otsu', 'adaptive', 'binary')
            
        Returns:
            Thresholded binary image
        """
        if method == 'otsu':
            _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return thresh
        elif method == 'adaptive':
            return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
        elif method == 'binary':
            _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            return thresh
        else:
            _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return thresh
    
    def preprocess_pipeline(self, image_path: str) -> dict:
        """
        Complete preprocessing pipeline
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary containing all preprocessed versions
        """
        # Load image
        original = self.load_image(image_path)
        if original is None:
            return {"error": "Failed to load image"}
        
        # Resize
        resized = self.resize_image(original)
        
        # Convert to grayscale
        gray = self.convert_to_grayscale(resized)
        
        # Denoise
        denoised = self.denoise(gray, method='gaussian')
        
        # Enhance contrast
        enhanced = self.enhance_contrast(denoised, method='clahe')
        
        # Edge detection
        edges = self.detect_edges(enhanced, method='canny')
        
        # Threshold
        thresh = self.threshold_image(enhanced, method='otsu')
        
        # Morphological operations
        morphed = self.apply_morphology(thresh, operation='closing')
        
        return {
            "original": original,
            "resized": resized,
            "grayscale": gray,
            "denoised": denoised,
            "enhanced": enhanced,
            "edges": edges,
            "threshold": thresh,
            "morphological": morphed
        }


# Example usage
if __name__ == "__main__":
    preprocessor = ImagePreprocessor(target_size=(640, 480))
    
    # Example: Process an image
    # results = preprocessor.preprocess_pipeline("path/to/image.jpg")
    # cv2.imshow("Original", results["original"])
    # cv2.imshow("Processed", results["morphological"])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    print("ImagePreprocessor module ready!")