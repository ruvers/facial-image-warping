import cv2
import numpy as np

def get_head_pose(landmarks, img_width, img_height):
    # Standart bir 3D insan kafasının uzaydaki anatomik koordinatları
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Burun ucu
        (0.0, -330.0, -65.0),        # Çene
        (-225.0, 170.0, -135.0),     # Sol göz sol köşe
        (225.0, 170.0, -135.0),      # Sağ göz sağ köşe
        (-150.0, -150.0, -125.0),    # Sol dudak köşesi
        (150.0, -150.0, -125.0)      # Sağ dudak köşesi
    ])

    # MediaPipe'tan gelen karşılık noktaların 2D piksel karşılıkları
    image_points = np.array([
        (landmarks[1]['x'] * img_width, landmarks[1]['y'] * img_height),     # Burun ucu
        (landmarks[152]['x'] * img_width, landmarks[152]['y'] * img_height), # Çene
        (landmarks[33]['x'] * img_width, landmarks[33]['y'] * img_height),   # Sol göz dış
        (landmarks[263]['x'] * img_width, landmarks[263]['y'] * img_height), # Sağ göz dış
        (landmarks[61]['x'] * img_width, landmarks[61]['y'] * img_height),   # Sol dudak
        (landmarks[291]['x'] * img_width, landmarks[291]['y'] * img_height)  # Sağ dudak
    ], dtype="double")

    # Kamera odak uzaklığı (focal length) simülasyonu
    focal_length = img_width
    center = (img_width/2, img_height/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    # Rotasyon ve Çeviri (Rotation & Translation) vektörlerini hesapla
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, np.zeros((4,1), dtype=float), flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    return rotation_vector, translation_vector