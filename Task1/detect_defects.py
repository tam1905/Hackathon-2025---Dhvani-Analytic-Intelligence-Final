from unittest import result
import cv2
from matplotlib import contour
import numpy as np
import json
from datetime import datetime
from typing import List, Dict, Tuple
from pathlib import Path


class DefectDetector:
    def __init__(
        self,
        circularity_threshold=0.80,
        defect_depth_threshold=15
    ):
        """
        Initialize the defect detector.

        Args:
            circularity_threshold (float): Minimum circularity for good
                parts (0-1, where 1.0 is a perfect circle)
            defect_depth_threshold (float): Minimum defect depth in pixels
                to be considered a defect
        """
        self.circularity_threshold = circularity_threshold
        self.defect_depth_threshold = defect_depth_threshold

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to binary.

        Args:
            image (np.ndarray): Input image (grayscale or color)

        Returns:
            np.ndarray: Binary image with objects in white
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Threshold to binary (invert if needed)
        _, binary = cv2.threshold(
            gray, 127, 255, cv2.THRESH_BINARY_INV
        )

        return binary

    def calculate_circularity(self, contour: np.ndarray) -> float:
        """
        Calculate circularity score.

        Args:
            contour (np.ndarray): Contour points

        Returns:
            float: Circularity score (1.0 = perfect circle, lower = defect)
        """
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            return 0

        circularity = 4 * np.pi * area / (perimeter ** 2)
        return circularity

    def detect_convexity_defects(
        self,
        contour: np.ndarray
    ) -> List[Dict]:
        """
        Detect notches and chips using convexity defects.

        Args:
            contour (np.ndarray): Contour points

        Returns:
            List[Dict]: List of detected defects with their properties
        """
        defects_list = []

        hull = cv2.convexHull(contour, returnPoints=False)

        if len(hull) > 3 and len(contour) > 3:
            try:
                defects = cv2.convexityDefects(contour, hull)

                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        depth = d / 256.0  # Convert to pixels

                        if depth > self.defect_depth_threshold:
                            start = tuple(contour[s][0])
                            end = tuple(contour[e][0])
                            far = tuple(contour[f][0])

                            defects_list.append({
                                'type': 'convexity_defect',
                                'depth': depth,
                                'start': start,
                                'end': end,
                                'farthest_point': far
                            })
            except cv2.error:
                pass

        return defects_list

    def detect_defects(
        self,
        image: np.ndarray,
        visualize: bool = False
    ) -> Dict:
        # Preprocess
        binary = self.preprocess(image)

        # Find contours
        contours, hierarchy = cv2.findContours(
            binary,
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_NONE
        )

        results = []
        vis_image = None
        if visualize:
            vis_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        for idx, contour in enumerate(contours):
            if cv2.contourArea(contour) < 100:
                continue
            circularity = self.calculate_circularity(contour)
            convex_defects = self.detect_convexity_defects(contour)
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            parent = hierarchy[0][idx][3]
            edge_type = "outer" if parent == -1 else "inner"
            has_circularity_issue = (circularity < self.circularity_threshold)
            has_convex_defects = len(convex_defects) > 0
            status = "DEFECT" if (has_circularity_issue or has_convex_defects) else "PASS"
            result = {
                'ring_id': idx,
                'edge': edge_type,   
                'status': status,
                'circularity': circularity,
                'center': center,
                'radius': radius,
                'convex_defects': convex_defects
            }
            results.append(result)
            
            # Visualization
            if visualize and vis_image is not None:
                color = (0, 0, 255) if status == "DEFECT" else (0, 255, 0)
                cv2.drawContours(vis_image, [contour], -1, color, 2)
                cv2.circle(vis_image, center, radius, color, 2)

                # Mark defect points
                for defect in convex_defects:
                    cv2.circle(
                        vis_image,
                        defect['farthest_point'],
                        8,
                        (255, 0, 255),
                        -1
                    )

                # Add text
                text = f"Ring {idx}: {status}"
                text_pos = (center[0] - 50, center[1] - radius - 10)
                cv2.putText(
                    vis_image,
                    text,
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

                circ_text = f"C: {circularity:.3f}"
                circ_pos = (center[0] - 50, center[1] - radius - 30)
                cv2.putText(
                    vis_image,
                    circ_text,
                    circ_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

        return {
            'results': results,
            'visualization': vis_image
        }

    def process_batch(
        self,
        input_folder: str,
        output_folder: str,
        save_visualization: bool = True,
        save_json: bool = True,
        image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')
    ) -> Dict:
        # Create output folder structure
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        if save_visualization:
            vis_folder = output_path / "visualizations"
            vis_folder.mkdir(exist_ok=True)

        if save_json:
            json_folder = output_path / "json_reports"
            json_folder.mkdir(exist_ok=True)

        # Get all image files
        input_path = Path(input_folder)
        image_files = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        if not image_files:
            print(f"No images found in {input_folder}")
            return {'total': 0, 'processed': 0, 'failed': 0}

        # Process images
        batch_results = []
        failed_images = []

        print(f"\nProcessing {len(image_files)} images...")

        for idx, image_file in enumerate(image_files, 1):
            try:
                # Load image
                image = cv2.imread(str(image_file))

                if image is None:
                    print(f"[{idx}/{len(image_files)}] Failed: "
                          f"{image_file.name}")
                    failed_images.append(image_file.name)
                    continue

                # Detect defects
                output = self.detect_defects(
                    image,
                    visualize=save_visualization
                )

                # Prepare result data
                image_result = {
                    'filename': image_file.name,
                    'timestamp': datetime.now().isoformat(),
                    'image_shape': image.shape,
                    'rings_detected': len(output['results']),
                    'results': []
                }

                # Convert results for JSON serialization
                for result in output['results']:
                    json_result = {
                        'ring_id': result['ring_id'],
                        'status': result['status'],
                        'circularity': float(result['circularity']),
                        'center': {
                            'x': int(result['center'][0]),
                            'y': int(result['center'][1])
                        },
                        'radius': int(result['radius']),
                        'defects': []
                    }

                    for defect in result['convex_defects']:
                        json_result['defects'].append({
                            'type': defect['type'],
                            'depth': float(defect['depth']),
                            'location': {
                                'x': int(defect['farthest_point'][0]),
                                'y': int(defect['farthest_point'][1])
                            }
                        })

                    image_result['results'].append(json_result)

                batch_results.append(image_result)

                # Save visualization
                if save_visualization and output['visualization'] is not None:
                    vis_filename = vis_folder / f"vis_{image_file.name}"
                    cv2.imwrite(str(vis_filename), output['visualization'])

                # Save JSON
                if save_json:
                    json_filename = json_folder / f"{image_file.stem}.json"
                    with open(json_filename, 'w') as f:
                        json.dump(image_result, f, indent=2)

                # Print progress
                status_summary = [r['status'] for r in output['results']]
                defect_count = status_summary.count('DEFECT')
                pass_count = status_summary.count('PASS')

                print(f"[{idx}/{len(image_files)}] {image_file.name}: "
                      f"{defect_count} DEFECT, {pass_count} PASS")

            except Exception as e:
                print(f"[{idx}/{len(image_files)}] Error processing "
                      f"{image_file.name}: {str(e)}")
                failed_images.append(image_file.name)

        # Save batch summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_images': len(image_files),
            'processed_successfully': len(batch_results),
            'failed': len(failed_images),
            'failed_images': failed_images,
            'detection_summary': {
                'total_rings': sum(
                    r['rings_detected'] for r in batch_results
                ),
                'defective_rings': sum(
                    sum(1 for ring in r['results']
                        if ring['status'] == 'DEFECT')
                    for r in batch_results
                ),
                'pass_rings': sum(
                    sum(1 for ring in r['results']
                        if ring['status'] == 'PASS')
                    for r in batch_results
                )
            },
            'results': batch_results
        }

        # Save summary JSON
        summary_file = output_path / "batch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*50}")
        print(f"Batch Processing Complete!")
        print(f"{'='*50}")
        print(f"Total images: {summary['total_images']}")
        print(f"Processed: {summary['processed_successfully']}")
        print(f"Failed: {summary['failed']}")
        print(f"\nDetection Summary:")
        print(f"Total rings: {summary['detection_summary']['total_rings']}")
        print(f"Defective: "
              f"{summary['detection_summary']['defective_rings']}")
        print(f"Pass: {summary['detection_summary']['pass_rings']}")
        print(f"\nResults saved to: {output_folder}")
        print(f"{'='*50}\n")

        return summary
    
def main():
    """Run defect detection on batch of images only."""
    detector = DefectDetector(
        circularity_threshold=0.80,
        defect_depth_threshold=5
    )

    current_dir = Path(__file__).absolute().parent
    input_folder = current_dir / "data"
    output_folder = current_dir / "results"

    if not input_folder.exists():
        print(f"Error: Input folder not found at {input_folder}")
        return

    # Always run batch processing
    summary = detector.process_batch(
        input_folder=str(input_folder),
        output_folder=str(output_folder),
        save_visualization=True,
        save_json=True,
        image_extensions=('.png', '.jpg', '.jpeg')
    )
    
    # Print summary
    print("\n=== Processing Summary ===")
    print(f"Total images found: {summary['total_images']}")
    print(f"Successfully processed: {summary['processed_successfully']}")
    print(f"Failed: {summary['failed']}")

if __name__ == "__main__":
    main()
