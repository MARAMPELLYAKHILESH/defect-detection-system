"""
Defect Detection Module
Implements various defect detection algorithms
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DefectDetector:
    """
    Detects defects in preprocessed images using various methods:
    - Contour-based detection
    - Template matching
    - Color-based detection
    - Blob detection
    """
    
    def __init__(self, min_defect_area: int = 100, max_defect_area: int = 10000):
        """
        Initialize defect detector
        
        Args:
            min_defect_area: Minimum area for a defect (pixels)
            max_defect_area: Maximum area for a defect (pixels)
        """
        self.min_defect_area = min_defect_area
        self.max_defect_area = max_defect_area
        logger.info(f"DefectDetector initialized (min: {min_defect_area}, max: {max_defect_area})")
    
    def detect_contours(self, binary_image: np.ndarray) -> List[np.ndarray]:
        """
        Detect contours in binary image
        
        Args:
            binary_image: Binary image (thresholded)
            
        Returns:
            List of detected contours
        """
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by area
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_defect_area <= area <= self.max_defect_area:
                valid_contours.append(contour)
        
        logger.info(f"Found {len(valid_contours)} valid contours")
        return valid_contours
    
    def analyze_contour(self, contour: np.ndarray) -> Dict:
        """
        Analyze a single contour for defect characteristics
        
        Args:
            contour: Contour to analyze
            
        Returns:
            Dictionary with contour properties
        """
        # Calculate properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Aspect ratio
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Circularity
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Moments for centroid
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00']) if M['m00'] != 0 else x + w // 2
        cy = int(M['m01'] / M['m00']) if M['m00'] != 0 else y + h // 2
        
        return {
            "area": area,
            "perimeter": perimeter,
            "bbox": (x, y, w, h),
            "centroid": (cx, cy),
            "aspect_ratio": aspect_ratio,
            "circularity": circularity
        }
    
    def classify_defect(self, properties: Dict) -> str:
        """
        Classify defect type based on properties
        
        Args:
            properties: Defect properties from analyze_contour
            
        Returns:
            Defect type classification
        """
        area = properties["area"]
        circularity = properties["circularity"]
        aspect_ratio = properties["aspect_ratio"]
        
        # Classification logic
        if circularity > 0.8:
            return "Circular Defect (Bubble/Hole)"
        elif aspect_ratio > 3:
            return "Linear Defect (Scratch)"
        elif area > 5000:
            return "Large Defect (Major Damage)"
        elif area < 500:
            return "Small Defect (Minor Scratch)"
        else:
            return "Irregular Defect"
    
    def detect_defects(self, original_image: np.ndarray, 
                      binary_image: np.ndarray) -> Dict:
        """
        Complete defect detection pipeline
        
        Args:
            original_image: Original color image
            binary_image: Preprocessed binary image
            
        Returns:
            Detection results with annotated image
        """
        # Detect contours
        contours = self.detect_contours(binary_image)
        
        # Analyze each contour
        defects = []
        annotated_image = original_image.copy()
        
        for idx, contour in enumerate(contours):
            # Analyze properties
            properties = self.analyze_contour(contour)
            
            # Classify defect
            defect_type = self.classify_defect(properties)
            
            # Draw on image
            cv2.drawContours(annotated_image, [contour], -1, (0, 0, 255), 2)
            
            # Draw bounding box
            x, y, w, h = properties["bbox"]
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add label
            cv2.putText(annotated_image, f"Defect {idx + 1}", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 0, 0), 2)
            
            # Store defect info
            defects.append({
                "id": idx + 1,
                "type": defect_type,
                "properties": properties
            })
        
        # Calculate overall quality score
        total_defect_area = sum(d["properties"]["area"] for d in defects)
        image_area = original_image.shape[0] * original_image.shape[1]
        defect_percentage = (total_defect_area / image_area) * 100
        
        # Quality assessment
        if defect_percentage < 1:
            quality = "PASS - Excellent"
        elif defect_percentage < 3:
            quality = "PASS - Good"
        elif defect_percentage < 5:
            quality = "WARNING - Acceptable"
        else:
            quality = "FAIL - Reject"
        
        return {
            "num_defects": len(defects),
            "defects": defects,
            "defect_percentage": defect_percentage,
            "quality_assessment": quality,
            "annotated_image": annotated_image
        }
    
    def detect_color_defects(self, image: np.ndarray, 
                            target_color: Tuple[int, int, int],
                            tolerance: int = 30) -> np.ndarray:
        """
        Detect defects based on color deviation
        
        Args:
            image: Input BGR image
            target_color: Target color in BGR
            tolerance: Color tolerance range
            
        Returns:
            Binary mask of color defects
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color range
        lower_bound = np.array([max(0, target_color[0] - tolerance),
                               max(0, target_color[1] - tolerance),
                               max(0, target_color[2] - tolerance)])
        upper_bound = np.array([min(255, target_color[0] + tolerance),
                               min(255, target_color[1] + tolerance),
                               min(255, target_color[2] + tolerance)])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Invert mask (defects are areas NOT matching target color)
        defect_mask = cv2.bitwise_not(mask)
        
        return defect_mask
    
    def detect_blobs(self, image: np.ndarray) -> List[cv2.KeyPoint]:
        """
        Detect blob-like defects using SimpleBlobDetector
        
        Args:
            image: Grayscale image
            
        Returns:
            List of detected keypoints (blobs)
        """
        # Setup blob detector parameters
        params = cv2.SimpleBlobDetector_Params()
        
        # Filter by area
        params.filterByArea = True
        params.minArea = self.min_defect_area
        params.maxArea = self.max_defect_area
        
        # Filter by circularity
        params.filterByCircularity = True
        params.minCircularity = 0.1
        
        # Filter by convexity
        params.filterByConvexity = True
        params.minConvexity = 0.5
        
        # Create detector
        detector = cv2.SimpleBlobDetector_create(params)
        
        # Detect blobs
        keypoints = detector.detect(image)
        
        logger.info(f"Detected {len(keypoints)} blobs")
        return keypoints


# Example usage
if __name__ == "__main__":
    detector = DefectDetector(min_defect_area=100, max_defect_area=10000)
    
    # Example: Detect defects in an image
    # original = cv2.imread("path/to/image.jpg")
    # binary = cv2.imread("path/to/preprocessed.jpg", cv2.IMREAD_GRAYSCALE)
    # results = detector.detect_defects(original, binary)
    # print(f"Quality: {results['quality_assessment']}")
    # print(f"Defects found: {results['num_defects']}")
    # cv2.imshow("Defects", results["annotated_image"])
    # cv2.waitKey(0)
    
    print("DefectDetector module ready!")