import cv2
import numpy as np
import os
import urllib.request
from backend.face_parsing import parse_face, get_mask, feather_mask

def download_real_asset(path):
    print("[*] İnternetten gerçekçi, şeffaf kolye asset'i indiriliyor...")
    url = "https://creazilla-store.fra1.digitaloceanspaces.com/cliparts/7815250/necklace-clipart-md.png"
    try:
        # Site bot sanıp engellemesin diye kendimizi Chrome tarayıcı gibi gösteriyoruz
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        with urllib.request.urlopen(req) as response, open(path, 'wb') as out_file:
            out_file.write(response.read())
        print("[+] Kaliteli asset başarıyla indirildi!")
        return True
    except Exception as e:
        print(f"[-] İndirme hatası: {e}")
        return False

def create_dummy_necklace(save_path):
    # İnternet engellerse sistemin çökmemesi için acil durum yedek kolyesi
    size = 400
    necklace = np.zeros((size, size, 4), dtype=np.uint8)
    cv2.ellipse(necklace, (200, 180), (140, 100), 0, 0, 360, (0, 215, 255, 255), 8)
    pts = np.array([[200, 280], [230, 320], [200, 360], [170, 320]], np.int32)
    cv2.fillPoly(necklace, [pts], (0, 150, 255, 255))
    cv2.polylines(necklace, [pts], True, (255, 255, 255, 255), 2)
    cv2.imwrite(save_path, necklace)
    print(f"[+] Yedek (çizim) kolye asset'i oluşturuldu.")

def overlay_with_occlusion(image_rgb, overlay_rgba, x, y, parsing_map):
    h_img, w_img = image_rgb.shape[:2]
    h_over, w_over = overlay_rgba.shape[:2]

    y1, y2 = max(0, y), min(h_img, y + h_over)
    x1, x2 = max(0, x), min(w_img, x + w_over)

    over_y1, over_y2 = max(0, -y), h_over - max(0, (y + h_over) - h_img)
    over_x1, over_x2 = max(0, -x), w_over - max(0, (x + w_over) - w_img)

    if y1 >= y2 or x1 >= x2: return image_rgb

    overlay_crop = overlay_rgba[over_y1:over_y2, over_x1:over_x2]
    img_crop = image_rgb[y1:y2, x1:x2]
    
    alpha_overlay = overlay_crop[:, :, 3].astype(np.float32) / 255.0
    
    # Kolyeyi YÜZÜN(1) ve SAÇIN(17) arkasında, BOYUN(14) üstünde tutan maske
    occlusion_mask_full = get_mask(parsing_map, [1, 2, 3, 4, 5, 10, 11, 12, 13, 17])
    occlusion_roi = occlusion_mask_full[y1:y2, x1:x2]
    
    occlusion_roi_feathered = feather_mask(occlusion_roi, blur=15)
    
    final_alpha = alpha_overlay * (1.0 - occlusion_roi_feathered)

    result_crop = np.zeros_like(img_crop, dtype=np.float32)
    for c in range(3):
        result_crop[:, :, c] = (final_alpha * overlay_crop[:, :, c] +
                                (1.0 - final_alpha) * img_crop[:, :, c])

    result_image = image_rgb.copy()
    result_image[y1:y2, x1:x2] = result_crop.astype(np.uint8)
    return result_image

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_path = os.path.join(BASE_DIR, "test.jpg")
    
    if not os.path.exists(img_path):
        print(f"[-] HATA: {img_path} bulunamadı!")
        return
        
    img = cv2.imread(img_path)
    if img is None:
        print("[-] HATA: OpenCV resmi okuyamadı.")
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    necklace_path = os.path.join(BASE_DIR, "real_necklace.png")
    if not os.path.exists(necklace_path):
        success = download_real_asset(necklace_path)
        if not success:
            print("[!] Site engelledi. Yedek çizim kolye kullanılıyor...")
            create_dummy_necklace(necklace_path)
            
    necklace_rgba = cv2.imread(necklace_path, cv2.IMREAD_UNCHANGED)
    if necklace_rgba is None:
        print("[-] Kolye asseti yüklenemedi.")
        return

    print("[+] BiSeNet ile anatomik analiz yapılıyor...")
    parsing_map = parse_face(img_rgb)
    
    neck_mask = get_mask(parsing_map, [14])
    y_idx, x_idx = np.where(neck_mask > 0)
    
    if len(y_idx) > 0:
        lowest_y = np.max(y_idx)
        bottom_pixels = x_idx[y_idx > lowest_y - 20]
        cX = int(np.mean(bottom_pixels))
        cY = lowest_y
        print(f"[+] Boyun tabanı bulundu: X:{cX}, Y:{cY}")
    else:
        print("[-] Boyun bulunamadı!")
        return

    face_mask = get_mask(parsing_map, [1])
    fx, fy, fw, fh = cv2.boundingRect(face_mask)
    
    target_w = int(fw * 1.3)
    scale = target_w / necklace_rgba.shape[1]
    new_w, new_h = int(necklace_rgba.shape[1] * scale), int(necklace_rgba.shape[0] * scale)
    necklace_resized = cv2.resize(necklace_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)

    start_x = cX - (new_w // 2)
    start_y = cY - int(new_h * 0.15)

    print("[+] Derinlik (Z-Buffer) uygulanarak kolye oturtuluyor...")
    result_rgb = overlay_with_occlusion(img_rgb, necklace_resized, start_x, start_y, parsing_map)

    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    save_path = os.path.join(BASE_DIR, "ar_test_result.jpg")
    cv2.imwrite(save_path, result_bgr)
    print(f"[+] İşlem Başarılı! Sonuç kaydedildi:\n    -> {save_path}")

if __name__ == "__main__":
    main()