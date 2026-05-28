import cv2
import numpy as np

def nothing(x):
    """Empty function for trackbar callback."""
    pass

def run_tri_visualizer(image_path):
    # 1. Load the image
    original_bgr = cv2.imread(image_path)
    if original_bgr is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Convert to float32 for high-precision math
    img_float = original_bgr.astype(np.float32) / 255.0

    # 2. Setup Window
    window_name = "DaYa-YOLO: RGB vs XYZ vs LAB Filtering"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 850)
    
    # 3. Create Sliders
    # Toggle: 0 = XYZ, 1 = RGB, 2 = LAB
    cv2.createTrackbar('MODE: 0=XYZ, 1=RGB, 2=LAB', window_name, 0, 2, nothing)
    
    # XYZ Sliders
    cv2.createTrackbar('X (Chroma-Red)', window_name, 100, 200, nothing)
    cv2.createTrackbar('Y (Luminance)',  window_name, 100, 200, nothing)
    cv2.createTrackbar('Z (Chroma-Blue)', window_name, 100, 200, nothing)
    
    # RGB Sliders
    cv2.createTrackbar('R (Red Channel)',   window_name, 100, 200, nothing)
    cv2.createTrackbar('G (Green Channel)', window_name, 100, 200, nothing)
    cv2.createTrackbar('B (Blue Channel)',  window_name, 100, 200, nothing)

    # LAB Sliders
    cv2.createTrackbar('L (Lightness)',     window_name, 100, 200, nothing)
    cv2.createTrackbar('A (Green <-> Red)', window_name, 100, 200, nothing)
    cv2.createTrackbar('B (Blue <-> Yel)',  window_name, 100, 200, nothing)

    print("\n--- Tri-Mode Explorer Active ---")
    print("0: XYZ Mode (DaYa-YOLO Aux Branch Physics)")
    print("1: RGB Mode (Standard YOLO Hardware Vision)")
    print("2: LAB Mode (Human Perceptual Vision)")
    print("Press 'q' to exit.")

    while True:
        # Get Mode
        mode = cv2.getTrackbarPos('MODE: 0=XYZ, 1=RGB, 2=LAB', window_name)
        
        if mode == 0:  # XYZ MODE
            x_s = cv2.getTrackbarPos('X (Chroma-Red)', window_name) / 100.0
            y_s = cv2.getTrackbarPos('Y (Luminance)',  window_name) / 100.0
            z_s = cv2.getTrackbarPos('Z (Chroma-Blue)', window_name) / 100.0
            
            # Convert BGR -> XYZ
            temp = cv2.cvtColor(img_float, cv2.COLOR_BGR2XYZ)
            temp[:, :, 0] *= x_s
            temp[:, :, 1] *= y_s
            temp[:, :, 2] *= z_s
            # Convert XYZ -> BGR
            output = cv2.cvtColor(temp, cv2.COLOR_XYZ2BGR)
            label = f"MODE: XYZ | X:{x_s:.1f} Y:{y_s:.1f} Z:{z_s:.1f}"
            
        elif mode == 1:  # RGB MODE
            r_s = cv2.getTrackbarPos('R (Red Channel)',   window_name) / 100.0
            g_s = cv2.getTrackbarPos('G (Green Channel)', window_name) / 100.0
            b_s = cv2.getTrackbarPos('B (Blue Channel)',  window_name) / 100.0
            
            # Scale BGR channels directly
            output = img_float.copy()
            output[:, :, 0] *= b_s # Blue
            output[:, :, 1] *= g_s # Green
            output[:, :, 2] *= r_s # Red
            label = f"MODE: RGB | R:{r_s:.1f} G:{g_s:.1f} B:{b_s:.1f}"

        else:  # LAB MODE
            l_s = cv2.getTrackbarPos('L (Lightness)',     window_name) / 100.0
            a_s = cv2.getTrackbarPos('A (Green <-> Red)', window_name) / 100.0
            b_s = cv2.getTrackbarPos('B (Blue <-> Yel)',  window_name) / 100.0

            # Convert BGR -> LAB
            # In OpenCV float32, L is [0, 100], A and B are centered around 0 (roughly -100 to +100)
            temp = cv2.cvtColor(img_float, cv2.COLOR_BGR2Lab)
            
            temp[:, :, 0] *= l_s  # Scale pure lightness
            temp[:, :, 1] *= a_s  # Scale intensity of Green/Red
            temp[:, :, 2] *= b_s  # Scale intensity of Blue/Yellow
            
            # Convert LAB -> BGR
            output = cv2.cvtColor(temp, cv2.COLOR_Lab2BGR)
            label = f"MODE: LAB | L:{l_s:.1f} A:{a_s:.1f} B:{b_s:.1f}"

        # Clip and convert back to uint8 for display
        output = np.clip(output * 255, 0, 255).astype(np.uint8)
        
        # Add Text Background for readability
        cv2.rectangle(output, (10, 10), (550, 60), (0, 0, 0), -1)
        cv2.putText(output, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow(window_name, output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_tri_visualizer('img0858.jpg')