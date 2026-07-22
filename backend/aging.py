import cv2
import numpy as np
from typing import Any

# ── MediaPipe Landmark İndeksleri ─────────────────────────────────────────────

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]
LEFT_EYE   = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
MOUTH      = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40]
LEFT_BROW  = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
RIGHT_BROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]


# ── Landmark Yardımcısı ───────────────────────────────────────────────────────


def _extract_pt(lm) -> tuple[int, int]:
    """Dict veya dizi formatındaki landmark noktasını (x, y) piksel olarak döndürür."""
    if isinstance(lm, dict):
        return int(lm.get("x", 0)), int(lm.get("y", 0))
    return int(lm[0]), int(lm[1])


# ── Maske Oluşturucular ───────────────────────────────────────────────────────


def create_skin_mask(image: np.ndarray, landmarks: list) -> np.ndarray:
    """
    Yalnızca cilt dokusunu kapsayan yumuşatılmış float32 [0, 1] maske üretir.
    Yüz ovali doldurulur; gözler, ağız ve kaşlar siyaha boyanır.
    Erosion + Gaussian blur ile kenar geçişleri pürüzsüzleştirilir.
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    try:
        oval_pts = np.array(
            [_extract_pt(landmarks[i]) for i in FACE_OVAL], dtype=np.int32
        )
        cv2.fillPoly(mask, [oval_pts], 255)

        for indices in (LEFT_EYE, RIGHT_EYE, MOUTH, LEFT_BROW, RIGHT_BROW):
            pts = np.array(
                [_extract_pt(landmarks[i]) for i in indices], dtype=np.int32
            )
            cv2.fillPoly(mask, [pts], 0)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
    except Exception as exc:
        print(f"[FaceWarp] Cilt maskesi hatası: {exc}")

    return mask.astype(np.float32) / 255.0



# ── Frekans Filtre Oluşturucuları ─────────────────────────────────────────────


def _build_hfe_filter(shape: tuple, cutoff_ratio: float, boost: float) -> np.ndarray:
    """
    Gaussian High-Frequency Emphasis filtresi:
        H(u,v) = 1 + boost × (1 − G_lp(u,v))

    DC merkezine yakın düşük frekanslar ≈ 1.0 kalır;
    kenar yüksek frekansları (1 + boost) kat amplifikasyona uğrar.

    cutoff_ratio : kesim frekansının görüntü boyutuna oranı (örn. 0.07)
    boost        : yüksek frekans güçlendirme katsayısı
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows, dtype=np.float64) - crow
    v = np.arange(cols, dtype=np.float64) - ccol
    V, U = np.meshgrid(v, u)
    D2 = U ** 2 + V ** 2
    sigma = cutoff_ratio * min(rows, cols) / 2.0
    G_lp = np.exp(-D2 / (2.0 * sigma ** 2 + 1e-10))
    return 1.0 + boost * (1.0 - G_lp)


def _build_gaussian_lp_filter(shape: tuple, sigma: float) -> np.ndarray:
    """
    Merkezi DC'de olan Gaussian alçak geçiş filtresi.
    Gençleştirme (de-aging) için yüksek frekansları bastırmakta kullanılır.
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows, dtype=np.float64) - crow
    v = np.arange(cols, dtype=np.float64) - ccol
    V, U = np.meshgrid(v, u)
    D2 = U ** 2 + V ** 2
    return np.exp(-D2 / (2.0 * sigma ** 2 + 1e-10))


# ── Tek Kanal FFT Filtreleme ──────────────────────────────────────────────────


def _fft_filter_channel(channel: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Tek kanallı (float) görüntüye FFT → filtre uygulama → IFFT işlem zinciri:

    1. np.fft.fft2     — uzamsal → frekans dönüşümü
    2. np.fft.fftshift — DC bileşenini merkeze taşı
    3. Fshift × H      — frekans domeninde filtre çarpımı
    4. np.fft.ifftshift + np.fft.ifft2 — uzamsal alana geri döndür
    5. np.real         — hayali artıkları at

    Dönüş: input ile aynı değer aralığında float32 kanal.
    """
    F = np.fft.fft2(channel.astype(np.float64))
    Fshift = np.fft.fftshift(F)
    Fshift_filtered = Fshift * H
    F_back = np.fft.ifftshift(Fshift_filtered)
    result = np.real(np.fft.ifft2(F_back))
    return result.astype(np.float32)


# ── Ana Fonksiyon ─────────────────────────────────────────────────────────────


def apply_aging_simulation(
    image: np.ndarray,
    intensity: float,
    landmarks: list = None,
    target_age: float | None = None,
) -> np.ndarray:
    """
    FFT Tabanlı Hibrit Yüz Yaşlandırma / Gençleştirme Simülasyonu.

    İşlem Sırası:
        [A] FFT Kırışıklık Katmanı
            1. RGB → CIE-Lab: parlaklık (L) kanalını renk kanallarından ayır.
            2. FFT: L'yi 2D Fourier dönüşümüyle frekans domenine taşı.
            3. Filtre:
               · intensity > 1.0 → Gaussian HFE ile yüksek frekansları yükselt
                 (kırışıklık / ince doku detayları).
               · intensity < 1.0 → Gaussian LP ile yüksek frekansları bastır
                 (gençleştirme / pürüzsüzleştirme).
            4. IFFT: filtrelenmiş spektrumu uzamsal alana geri döndür.
            5. Unsharp masking (sadece aging): doku detaylarını pekiştir.
            6. Lab → RGB.

        [B] Maske Tabanlı Bölgesel Uygulama (landmarks mevcutsa)
            7. Cilt maskesi: FFT kırışık görüntüyü sadece cilt bölgesine uygula;
               arka plan ve gözler orijinal kalır.
            8. Saç beyazlatma: hairline üstündeki bölgede HSV ile doygunluk
               düşürülür, parlaklık artırılır → intensity orantılı beyazlaşma.
            9. Sakal beyazlatma: jawline altında aynı HSV işlemi uygulanır.

    Args:
        image     : RGB uint8 NumPy dizisi (512×512 beklenir).
        intensity : 0.0–<1.0 → gençleştirme | ≈1.0 → değişmez | >1.0 → yaşlandırma.
        landmarks : MediaPipe landmark dict/list listesi (opsiyonel, en az 468 nokta).
        target_age: 7-85 yaş aralığı (opsiyonel). Belirtilirse intensity bu yaşa göre ezilir.

    Returns:
        RGB uint8 NumPy dizisi.
    """
    if target_age is not None:
        target_age = float(np.clip(target_age, 7.0, 85.0))
        if target_age <= 35.0:
            intensity = 0.2 + 0.8 * (target_age - 7.0) / 28.0
        else:
            intensity = 1.0 + 1.0 * (target_age - 35.0) / 50.0

    if abs(intensity - 1.0) < 0.01:
        return image.copy()

    # ── [A] FFT Kırışıklık Katmanı ────────────────────────────────────────────

    # 1. RGB → CIE-Lab  (float32 [0,1] girdi; L: 0-100, A/B: -127…127)
    img_01 = image.astype(np.float32) / 255.0
    lab = cv2.cvtColor(img_01, cv2.COLOR_RGB2Lab)
    L, A, B = cv2.split(lab)

    # 2 & 3. Frekans filtresi tasarımı
    if intensity > 1.0:
        strength = min(intensity - 1.0, 2.0)
        cutoff   = 0.07            # DC merkezli kesim oranı
        boost    = strength * 3.0  # yüksek frekans güçlendirme katsayısı
        H_filt = _build_hfe_filter(L.shape, cutoff, boost)
    else:
        strength = 1.0 - intensity
        sigma    = 20.0 + strength * 60.0
        H_filt = _build_gaussian_lp_filter(L.shape, sigma)

    # 4. FFT → filtre → IFFT
    L_filtered = _fft_filter_channel(L, H_filt)

    # 5. Karışım & ton düzeltme
    if intensity > 1.0:
        blend_alpha = min(0.35 + strength * 0.30, 0.80)
        L_out = blend_alpha * L_filtered + (1.0 - blend_alpha) * L
        L_out = L_out - strength * 2.5

        # Unsharp masking: ince kırışıklık/doku detaylarını pekiştir
        L_clamped = np.clip(L_out, 0.0, 100.0)
        L_blur    = cv2.GaussianBlur(L_clamped, (0, 0), sigmaX=1.2)
        detail    = L_clamped - L_blur
        L_out     = L_clamped + detail * strength
    else:
        blend_alpha = min(0.30 + strength * 0.50, 0.85)
        L_out = blend_alpha * L_filtered + (1.0 - blend_alpha) * L
        L_out = L_out + strength * 2.0  # genç cilt hafif daha parlak

    L_out = np.clip(L_out, 0.0, 100.0).astype(np.float32)

    # 6. Lab → RGB
    lab_out    = cv2.merge([L_out, A, B])
    rgb_01     = cv2.cvtColor(lab_out, cv2.COLOR_Lab2RGB)
    aged_uint8 = np.clip(rgb_01 * 255.0, 0, 255).astype(np.uint8)

    # ── [B] Maske Tabanlı Bölgesel Uygulama ──────────────────────────────────

    if landmarks is None or len(landmarks) < 468:
        return aged_uint8

    # 7. Cilt maskesi: FFT kırışık görüntüyü sadece yüz derisine uygula
    skin_mask     = create_skin_mask(image, landmarks)
    skin_mask_3ch = np.stack([skin_mask, skin_mask, skin_mask], axis=-1)

    orig_f  = image.astype(np.float32)
    aged_f  = aged_uint8.astype(np.float32)
    result  = aged_f * skin_mask_3ch + orig_f * (1.0 - skin_mask_3ch)
    result  = np.clip(result, 0, 255).astype(np.uint8)

    return result


# ---------------------------------------------------------------------------
# Regional frequency aging implementation
#
# This section intentionally lives after the original implementation so older
# helper functions remain available while public calls use the safer regional
# version below.

NOSE_TIP_IDX = 1
LEFT_CHEEK_CENTER_IDX = 50
RIGHT_CHEEK_CENTER_IDX = 280

FOREHEAD_CAP_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377,
]
GLABELLA_IDX = [9, 8, 168, 6, 197, 195, 5, 151, 107, 336]
NASOLABIAL_LEFT = [205, 187, 207, 216, 212, 202]
NASOLABIAL_RIGHT = [425, 411, 427, 436, 432, 422]
JAWLINE_DISPLACE_IDX = [
    234, 132, 58, 172, 136, 150, 149, 176, 148, 152,
    377, 400, 378, 379, 365, 397, 288, 361, 323, 454,
]
UNDER_EYE_LEFT_IDX = [374, 381, 382, 362, 398, 384, 385, 386]
UNDER_EYE_RIGHT_IDX = [145, 144, 163, 7, 33, 154, 155, 153]


def _extract_pt(lm) -> tuple[float, float]:
    if isinstance(lm, dict):
        return float(lm.get("x", 0)), float(lm.get("y", 0))
    if hasattr(lm, "x") and hasattr(lm, "y"):
        return float(lm.x), float(lm.y)
    return float(lm[0]), float(lm[1])


def _landmarks_2d_from_list(landmarks: list | None) -> np.ndarray | None:
    if landmarks is None or len(landmarks) < 400:
        return None
    n = len(landmarks)
    arr = np.zeros((n, 2), dtype=np.float32)
    for i, lm in enumerate(landmarks):
        if isinstance(lm, dict) and lm.get("index") is not None:
            idx = int(lm["index"])
            if 0 <= idx < n:
                arr[idx, 0], arr[idx, 1] = _extract_pt(lm)
        else:
            arr[i, 0], arr[i, 1] = _extract_pt(lm)
    return arr


def _resolve_landmarks_2d(
    ctx: dict | None,
    landmarks: list | None,
) -> np.ndarray | None:
    if ctx:
        lm = ctx.get("landmarks_2d")
        if isinstance(lm, np.ndarray) and lm.shape[0] >= 400 and lm.shape[1] >= 2:
            return lm.astype(np.float32, copy=False)
    return _landmarks_2d_from_list(landmarks)


def _pt_indices(pt: np.ndarray, indices: list[int]) -> np.ndarray:
    rows = [pt[i] for i in indices if i < pt.shape[0]]
    if not rows:
        return np.zeros((0, 2), dtype=np.float32)
    return np.stack(rows, axis=0).astype(np.float32)


def _face_scale_px(pt: np.ndarray) -> tuple[float, float, float]:
    iod = float(np.linalg.norm(pt[33] - pt[263])) if pt.shape[0] > 263 else 80.0
    if pt.shape[0] > 152:
        chin_y = float(pt[152][1])
        top_y = float(np.min([pt[10][1], pt[151][1]]))
        height = max(chin_y - top_y, 1.0)
    else:
        top_y = 0.0
        height = 200.0
    return max(iod, 20.0), height, top_y


def _gaussian_blob_mask(
    h: int,
    w: int,
    cx: float,
    cy: float,
    sigma_x: float,
    sigma_y: float,
    angle_deg: float = 0.0,
) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x0 = xx - cx
    y0 = yy - cy
    if abs(angle_deg) > 1e-3:
        rad = np.deg2rad(angle_deg)
        c, s = np.cos(rad), np.sin(rad)
        xr = c * x0 + s * y0
        yr = -s * x0 + c * y0
    else:
        xr, yr = x0, y0
    sx = max(float(sigma_x), 1.0)
    sy = max(float(sigma_y), 1.0)
    return np.exp(-0.5 * (xr / sx) ** 2 - 0.5 * (yr / sy) ** 2).astype(np.float32)


def _poly_fill_soft(h: int, w: int, pts: np.ndarray, blur_ksize: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    if pts.shape[0] >= 3:
        hull = cv2.convexHull(pts.astype(np.float32))
        cv2.fillConvexPoly(mask, hull.astype(np.int32), 255, lineType=cv2.LINE_AA)
    k = max(3, int(blur_ksize)) | 1
    return np.clip(cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (k, k), 0), 0.0, 1.0)


def _polyline_ribbon_mask(
    h: int,
    w: int,
    pts: np.ndarray,
    thickness: int,
    blur_ksize: int,
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    if pts.shape[0] >= 2:
        cv2.polylines(
            mask,
            [pts.astype(np.int32)],
            isClosed=False,
            color=255,
            thickness=max(1, int(thickness)),
            lineType=cv2.LINE_AA,
        )
    k = max(3, int(blur_ksize)) | 1
    return np.clip(cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (k, k), 0), 0.0, 1.0)


def _build_forehead_mask(h: int, w: int, pt: np.ndarray) -> np.ndarray:
    top_pts = _pt_indices(pt, FOREHEAD_CAP_IDX)
    glab = _pt_indices(pt, GLABELLA_IDX)
    brows = _pt_indices(pt, LEFT_BROW + RIGHT_BROW)
    if top_pts.shape[0] < 4 or glab.shape[0] < 3 or brows.shape[0] < 3:
        return np.zeros((h, w), dtype=np.float32)
    brow_y = float(np.mean(brows[:, 1]))
    upper = top_pts[top_pts[:, 1] < brow_y + 0.02 * h]
    if upper.shape[0] < 3:
        upper = top_pts
    return _poly_fill_soft(h, w, np.vstack([upper, glab]), blur_ksize=int(max(21, h // 18)))


def _build_under_eye_masks(h: int, w: int, pt: np.ndarray, iod: float) -> tuple[np.ndarray, np.ndarray]:
    masks = []
    for indices in (UNDER_EYE_LEFT_IDX, UNDER_EYE_RIGHT_IDX):
        pts = _pt_indices(pt, indices)
        if pts.shape[0] == 0:
            masks.append(np.zeros((h, w), dtype=np.float32))
            continue
        c = np.mean(pts, axis=0)
        c[1] += 0.035 * iod
        masks.append(_gaussian_blob_mask(h, w, float(c[0]), float(c[1]), iod * 0.22, iod * 0.12))
    return masks[0], masks[1]


def _eye_center(pt: np.ndarray, indices: list[int]) -> np.ndarray:
    pts = _pt_indices(pt, indices)
    return np.mean(pts, axis=0) if pts.shape[0] else pt[0]


def _build_crow_feet_masks(h: int, w: int, pt: np.ndarray, iod: float) -> tuple[np.ndarray, np.ndarray]:
    if pt.shape[0] <= 263:
        z = np.zeros((h, w), dtype=np.float32)
        return z, z
    lc = _eye_center(pt, LEFT_EYE)
    rc = _eye_center(pt, RIGHT_EYE)
    left_outer = pt[263]
    right_outer = pt[33]
    dl = left_outer - lc
    dr = right_outer - rc
    dl = dl / (float(np.linalg.norm(dl)) + 1e-6)
    dr = dr / (float(np.linalg.norm(dr)) + 1e-6)
    cl = left_outer + dl * (iod * 0.35)
    cr = right_outer + dr * (iod * 0.35)
    return (
        _gaussian_blob_mask(h, w, float(cl[0]), float(cl[1]), iod * 0.18, iod * 0.10, 25.0),
        _gaussian_blob_mask(h, w, float(cr[0]), float(cr[1]), iod * 0.18, iod * 0.10, -25.0),
    )


def _build_nasolabial_masks(h: int, w: int, pt: np.ndarray, iod: float) -> tuple[np.ndarray, np.ndarray]:
    thick = max(3, int(iod * 0.07))
    blur = int(max(15, iod * 0.16))
    pl = _pt_indices(pt, NASOLABIAL_LEFT)
    pr = _pt_indices(pt, NASOLABIAL_RIGHT)
    ml = _polyline_ribbon_mask(h, w, pl, thick, blur) if pl.shape[0] >= 2 else np.zeros((h, w), np.float32)
    mr = _polyline_ribbon_mask(h, w, pr, thick, blur) if pr.shape[0] >= 2 else np.zeros((h, w), np.float32)
    return ml, mr


def _build_exclusion_masks(h: int, w: int, pt: np.ndarray, iod: float) -> np.ndarray:
    ex = np.zeros((h, w), dtype=np.float32)
    if NOSE_TIP_IDX < pt.shape[0]:
        p = pt[NOSE_TIP_IDX]
        ex = np.maximum(ex, _gaussian_blob_mask(h, w, float(p[0]), float(p[1]), iod * 0.12, iod * 0.11))
    for idx in (LEFT_CHEEK_CENTER_IDX, RIGHT_CHEEK_CENTER_IDX):
        if idx < pt.shape[0]:
            p = pt[idx]
            ex = np.maximum(ex, _gaussian_blob_mask(h, w, float(p[0]), float(p[1]), iod * 0.28, iod * 0.88 * 0.28))
    for idx in LEFT_EYE + RIGHT_EYE:
        if idx < pt.shape[0]:
            p = pt[idx]
            ex = np.maximum(ex, _gaussian_blob_mask(h, w, float(p[0]), float(p[1]), iod * 0.14, iod * 0.10))
    mouth_pts = _pt_indices(pt, [13, 14, 78, 308, 312, 82, 87, 178])
    if mouth_pts.shape[0] >= 3:
        ex = np.maximum(ex, _poly_fill_soft(h, w, mouth_pts, int(max(13, h // 40))))
    return np.clip(ex, 0.0, 1.0)


def build_wrinkle_prone_mask(image_shape: tuple[int, ...], pt: np.ndarray) -> np.ndarray:
    h, w = int(image_shape[0]), int(image_shape[1])
    iod, _, _ = _face_scale_px(pt)
    m_fore = _build_forehead_mask(h, w, pt)
    ul, ur = _build_under_eye_masks(h, w, pt, iod)
    cl, cr = _build_crow_feet_masks(h, w, pt, iod)
    nl, nr = _build_nasolabial_masks(h, w, pt, iod)
    combined = np.maximum.reduce([m_fore, ul, ur, cl, cr, nl, nr])
    combined *= 1.0 - np.clip(_build_exclusion_masks(h, w, pt, iod) * 1.15, 0.0, 1.0)
    smooth_k = int(max(151, min(h, w) // 3)) | 1
    return cv2.GaussianBlur(np.clip(combined, 0.0, 1.0), (smooth_k, smooth_k), smooth_k * 0.3).astype(np.float32)


def _extract_wrinkles_gabor_morphology(L_norm: np.ndarray, iod: float, t: float) -> np.ndarray:
    bh_ksize = int(max(7, iod * 0.1)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bh_ksize, bh_ksize))
    lines = cv2.morphologyEx(L_norm.astype(np.float32), cv2.MORPH_BLACKHAT, kernel)
    responses = []
    for theta_deg in (0, 45, 90, 135):
        gabor = cv2.getGaborKernel(
            (int(max(5, iod * 0.12)) | 1, int(max(5, iod * 0.12)) | 1),
            max(iod * 0.02, 1.0),
            np.deg2rad(theta_deg),
            max(iod * 0.05, 2.0),
            0.5,
            0,
            ktype=cv2.CV_32F,
        )
        responses.append(np.maximum(cv2.filter2D(lines, cv2.CV_32F, gabor), 0.0))
    if responses:
        lines = np.maximum(lines, np.max(np.stack(responses, axis=0), axis=0))
    lines = cv2.GaussianBlur(lines, (5, 5), 0)
    return np.maximum(lines - 0.1, 0.0).astype(np.float32)


def _generate_directional_lines(h: int, w: int, iod: float, angle_deg: float) -> np.ndarray:
    noise = np.random.randn(h, w).astype(np.float32)
    ksize = int(max(5.0, iod * 0.35)) | 1
    center = ksize // 2
    k = np.zeros((ksize, ksize), dtype=np.float32)
    k[center, :] = 1.0
    m = cv2.getRotationMatrix2D((center, center), angle_deg, 1.0)
    k = cv2.warpAffine(k, m, (ksize, ksize))
    k /= float(np.sum(k) + 1e-6)
    blurred = cv2.filter2D(noise, -1, k)
    normalized = (blurred - float(np.mean(blurred))) / (float(np.std(blurred)) + 1e-6)
    return np.clip(np.maximum(normalized - 1.8, 0.0) * 2.5, 0.0, 1.0)


def _generate_age_spots(h: int, w: int) -> np.ndarray:
    noise1 = cv2.GaussianBlur(np.random.randn(h, w).astype(np.float32), (3, 3), 0.5)
    noise2 = np.random.randn(max(1, h // 2), max(1, w // 2)).astype(np.float32)
    noise2 = cv2.resize(cv2.GaussianBlur(noise2, (3, 3), 0), (w, h))
    noise = noise1 * 0.7 + noise2 * 0.3
    noise = (noise - float(np.mean(noise))) / (float(np.std(noise)) + 1e-6)
    return np.clip(cv2.GaussianBlur(np.maximum(noise - 2.0, 0.0), (3, 3), 0) * 3.0, 0.0, 1.0)


def _build_hair_mask(h: int, w: int, pt: np.ndarray, iod: float) -> np.ndarray:
    top_pts = _pt_indices(pt, FOREHEAD_CAP_IDX)
    if top_pts.shape[0] < 5:
        return np.zeros((h, w), dtype=np.float32)
    dy = iod * 1.4
    hair_pts = np.vstack([top_pts, np.flip(top_pts, axis=0) + np.array([0.0, -dy], dtype=np.float32)])
    return _poly_fill_soft(h, w, hair_pts, blur_ksize=int(max(51, iod * 0.5)))


def _build_sideburn_mask(h: int, w: int, pt: np.ndarray, iod: float) -> np.ndarray:
    l_sb = _pt_indices(pt, [162, 127, 234, 93])
    r_sb = _pt_indices(pt, [389, 356, 454, 323])
    thick = int(max(15, iod * 0.45))
    blur = int(max(21, iod * 0.4))
    ml = _polyline_ribbon_mask(h, w, l_sb, thick, blur) if l_sb.shape[0] >= 2 else np.zeros((h, w), np.float32)
    mr = _polyline_ribbon_mask(h, w, r_sb, thick, blur) if r_sb.shape[0] >= 2 else np.zeros((h, w), np.float32)
    return np.clip(ml + mr, 0.0, 1.0)


def _build_eyebrow_mask(h: int, w: int, pt: np.ndarray, iod: float) -> np.ndarray:
    ml = _poly_fill_soft(h, w, _pt_indices(pt, LEFT_BROW), int(max(9, iod * 0.1)))
    mr = _poly_fill_soft(h, w, _pt_indices(pt, RIGHT_BROW), int(max(9, iod * 0.1)))
    return np.maximum(ml, mr)


def _build_full_face_mask(h: int, w: int, pt: np.ndarray, iod: float) -> np.ndarray:
    oval_pts = _pt_indices(pt, FACE_OVAL)
    if oval_pts.shape[0] < 10:
        return np.zeros((h, w), dtype=np.float32)
    face_mask = _poly_fill_soft(h, w, oval_pts, blur_ksize=int(max(21, iod * 0.2)))
    ex = np.clip(
        _poly_fill_soft(h, w, _pt_indices(pt, LEFT_EYE), int(max(11, iod * 0.1)))
        + _poly_fill_soft(h, w, _pt_indices(pt, RIGHT_EYE), int(max(11, iod * 0.1)))
        + _poly_fill_soft(h, w, _pt_indices(pt, MOUTH), int(max(11, iod * 0.1))),
        0.0,
        1.0,
    )
    return np.clip(face_mask - ex, 0.0, 1.0)


def _build_glabella_mask(h: int, w: int, pt: np.ndarray, iod: float) -> np.ndarray:
    pts = _pt_indices(pt, GLABELLA_IDX)
    if pts.shape[0] == 0:
        return np.zeros((h, w), dtype=np.float32)
    c = np.mean(pts, axis=0)
    return _gaussian_blob_mask(h, w, float(c[0]), float(c[1]), iod * 0.13, iod * 0.16)


def _build_cheek_texture_mask(h: int, w: int, pt: np.ndarray, iod: float) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.float32)
    for idx in (LEFT_CHEEK_CENTER_IDX, RIGHT_CHEEK_CENTER_IDX):
        if idx < pt.shape[0]:
            p = pt[idx]
            mask = np.maximum(
                mask,
                _gaussian_blob_mask(
                    h,
                    w,
                    float(p[0]),
                    float(p[1] + iod * 0.05),
                    iod * 0.24,
                    iod * 0.20,
                ),
            )
    return np.clip(mask, 0.0, 1.0)


def _build_marionette_masks(h: int, w: int, pt: np.ndarray, iod: float) -> tuple[np.ndarray, np.ndarray]:
    left = _pt_indices(pt, [61, 146, 91, 181, 84])
    right = _pt_indices(pt, [291, 375, 321, 405, 314])
    thick = max(3, int(iod * 0.045))
    blur = int(max(13, iod * 0.13))
    ml = _polyline_ribbon_mask(h, w, left, thick, blur) if left.shape[0] >= 2 else np.zeros((h, w), np.float32)
    mr = _polyline_ribbon_mask(h, w, right, thick, blur) if right.shape[0] >= 2 else np.zeros((h, w), np.float32)
    return ml, mr


def _build_jaw_softening_mask(h: int, w: int, pt: np.ndarray, iod: float) -> np.ndarray:
    jaw = _pt_indices(pt, JAWLINE_DISPLACE_IDX)
    if jaw.shape[0] < 4:
        return np.zeros((h, w), dtype=np.float32)
    return _polyline_ribbon_mask(
        h,
        w,
        jaw,
        thickness=max(5, int(iod * 0.12)),
        blur_ksize=int(max(19, iod * 0.24)),
    )


def _build_pore_texture(h: int, w: int, iod: float) -> np.ndarray:
    noise = np.random.randn(h, w).astype(np.float32)
    fine = cv2.GaussianBlur(noise, (3, 3), 0.5)
    coarse = cv2.GaussianBlur(noise, (0, 0), max(1.0, iod * 0.045))
    texture = fine - coarse
    texture = (texture - float(np.mean(texture))) / (float(np.std(texture)) + 1e-6)
    texture = np.maximum(texture - 0.45, 0.0)
    return np.clip(texture * 0.55, 0.0, 1.0).astype(np.float32)


def _region_debug(mask: np.ndarray) -> dict[str, Any]:
    pixels = int(np.count_nonzero(mask > 0.04))
    return {
        "applied": pixels > 0,
        "mask_pixels": pixels,
    }


def _safe_parse_face(rgb: np.ndarray) -> np.ndarray | None:
    try:
        from .face_parsing import parse_face

        parsing = parse_face(np.clip(rgb, 0, 255).astype(np.uint8))
        if isinstance(parsing, np.ndarray) and parsing.size:
            return parsing
    except Exception as exc:
        print(f"[FaceWarp] Face parsing aging fallback: {exc}")
    return None


def _safe_label_mask(parsing: np.ndarray | None, labels: list[int]) -> np.ndarray | None:
    if parsing is None:
        return None
    try:
        from .face_parsing import feather_mask, get_mask

        return feather_mask(get_mask(parsing, labels), blur=21).astype(np.float32)
    except Exception as exc:
        print(f"[FaceWarp] Face parsing mask fallback: {exc}")
        return None


def create_skin_mask(image: np.ndarray, landmarks: list) -> np.ndarray:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    try:
        pt = _landmarks_2d_from_list(landmarks)
        if pt is None:
            return mask.astype(np.float32)
        oval_pts = _pt_indices(pt, FACE_OVAL).astype(np.int32)
        cv2.fillPoly(mask, [oval_pts], 255)
        for indices in (LEFT_EYE, RIGHT_EYE, MOUTH, LEFT_BROW, RIGHT_BROW):
            pts = _pt_indices(pt, indices)
            if pts.shape[0] >= 3:
                cv2.fillPoly(mask, [pts.astype(np.int32)], 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
    except Exception as exc:
        print(f"[FaceWarp] Skin mask error: {exc}")
    return mask.astype(np.float32) / 255.0



def apply_aging_simulation(
    image: np.ndarray,
    intensity: float,
    landmarks: list | None = None,
    target_age: float | None = None,
) -> np.ndarray:
    """
    Frequency-based regional aging/de-aging. RGB uint8 in/out.

    Landmark-free aging keeps the broad frequency fallback. Landmark-based aging
    uses feathered facial zones so wrinkles do not appear uniformly or as hard
    random spikes across the full face.
    """
    if image is None:
        raise ValueError("image is None")

    if target_age is not None:
        target_age = float(np.clip(target_age, 7.0, 85.0))
        if target_age <= 35.0:
            intensity = 0.2 + 0.8 * (target_age - 7.0) / 28.0
        else:
            intensity = 1.0 + 1.0 * (target_age - 35.0) / 50.0

    value = float(intensity)
    if abs(value - 1.0) < 0.01:
        return image.copy()

    rgb = image.astype(np.float32).copy()
    h, w = rgb.shape[:2]
    t = float(np.clip(abs(value - 1.0), 0.0, 1.0))
    pt = _landmarks_2d_from_list(landmarks)

    img_01 = np.clip(rgb / 255.0, 0.0, 1.0)
    lab = cv2.cvtColor(img_01, cv2.COLOR_RGB2Lab)
    L, A, Bch = cv2.split(lab)
    L_norm = np.clip(L.astype(np.float32) / 100.0, 0.0, 1.0)
    A_new = A.astype(np.float32).copy()
    B_new = Bch.astype(np.float32).copy()

    if value < 1.0:
        smooth = cv2.bilateralFilter(L_norm, 9, 0.25 * t, 18.0 * t + 1.0)
        blend = np.clip(1.35 * t, 0.0, 0.85)
        L_out = (L_norm * (1.0 - blend) + smooth * blend) * 100.0
        lab_out = cv2.merge([np.clip(L_out, 1.0, 99.5).astype(np.float32), A_new, B_new])
        return np.clip(cv2.cvtColor(lab_out, cv2.COLOR_Lab2RGB) * 255.0, 0, 255).astype(np.uint8)

    if pt is None:
        # Landmark-free fallback: preserve legacy broad frequency behavior.
        strength = t
        h_filt = _build_hfe_filter(L.shape, 0.07, strength * 3.0)
        L_filtered = _fft_filter_channel(L, h_filt)
        L_out = np.clip(
            (0.35 + strength * 0.30) * L_filtered + (0.65 - strength * 0.30) * L,
            0.0,
            100.0,
        )
        lab_out = cv2.merge([L_out.astype(np.float32), A_new, B_new])
        return np.clip(cv2.cvtColor(lab_out, cv2.COLOR_Lab2RGB) * 255.0, 0, 255).astype(np.uint8)

    def feather(mask: np.ndarray) -> np.ndarray:
        mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
        return np.clip(cv2.GaussianBlur(mask, (31, 31), 0), 0.0, 1.0)

    def blob(cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        return np.exp(-0.5 * ((xx - cx) / max(rx, 1.0)) ** 2 - 0.5 * ((yy - cy) / max(ry, 1.0)) ** 2).astype(np.float32)

    def apply_luminance(effect_l: np.ndarray, alpha: np.ndarray) -> None:
        nonlocal L_norm
        a = np.clip(alpha, 0.0, 1.0)
        L_norm = np.clip(effect_l * a + L_norm * (1.0 - a), 0.0, 1.0)

    face_top = float(min(pt[10][1], pt[151][1])) if pt.shape[0] > 151 else h * 0.24
    face_bottom = float(pt[152][1]) if pt.shape[0] > 152 else h * 0.72
    face_height = max(face_bottom - face_top, 1.0)
    face_width = max(float(abs(pt[454][0] - pt[234][0])) if pt.shape[0] > 454 else w * 0.35, 1.0)
    factor = np.clip((value - 1.0) / 0.75, 0.0, 1.0)

    protected = np.zeros((h, w), dtype=np.float32)
    for indices in (LEFT_EYE, RIGHT_EYE, MOUTH):
        pts = _pt_indices(pt, indices)
        if pts.shape[0] >= 3:
            protected = np.maximum(protected, _poly_fill_soft(h, w, pts, int(max(9, face_width * 0.04))))
    protected = np.clip(protected * 1.25, 0.0, 1.0)

    # Forehead: mild parallel horizontal lines between head top and eyebrow center.
    forehead = np.zeros((h, w), dtype=np.float32)
    if pt.shape[0] > 454:
        left_x = int(np.clip(pt[234][0], 0, w - 1))
        right_x = int(np.clip(pt[454][0], 0, w - 1))
        top_y = int(np.clip(face_top - face_height * 0.05, 0, h - 1))
        bottom_y = int(np.clip(pt[9][1] if pt.shape[0] > 9 else face_top + face_height * 0.22, 0, h - 1))
        if right_x > left_x and bottom_y > top_y:
            cv2.rectangle(forehead, (left_x, top_y), (right_x, bottom_y), 1.0, -1)
            lines = np.zeros_like(forehead)
            line_count = 6
            for i in range(line_count):
                y = int(top_y + (i + 1) * (bottom_y - top_y) / (line_count + 1))
                cv2.line(lines, (left_x + 8, y), (right_x - 8, y + (i % 2)), 1.0, 1, cv2.LINE_AA)
            lines = cv2.GaussianBlur(lines, (5, 5), 0)
            alpha = feather(forehead * (1.0 - protected)) * (0.85 * factor)
            effect = np.clip(L_norm - lines * 0.22 * factor, 0.0, 1.0)
            apply_luminance(effect, alpha)

    # Eye corners: crow's feet radiate outward from outer eye landmarks.
    eye_lines = np.zeros((h, w), dtype=np.float32)
    for idx, direction in ((33, -1), (263, 1)):
        if idx < pt.shape[0]:
            cx, cy = float(pt[idx][0]), float(pt[idx][1])
            for dy in (-12, 0, 12):
                end = (int(cx + direction * 34), int(cy + dy))
                cv2.line(eye_lines, (int(cx), int(cy)), end, 1.0, 1, cv2.LINE_AA)
    eye_zone = np.zeros((h, w), dtype=np.float32)
    if pt.shape[0] > 263:
        eye_zone = np.maximum(blob(float(pt[33][0]), float(pt[33][1]), 25.0, 25.0), blob(float(pt[263][0]), float(pt[263][1]), 25.0, 25.0))
    eye_lines = cv2.GaussianBlur(eye_lines, (5, 5), 0)
    alpha = feather(eye_zone * (1.0 - protected)) * (1.2 * factor)
    effect = np.clip(L_norm - eye_lines * 0.22 * factor, 0.0, 1.0)
    apply_luminance(effect, alpha)

    # Nasolabial folds: soft vertical/diagonal fold from nose side toward mouth corner.
    naso_lines = np.zeros((h, w), dtype=np.float32)
    for indices in ([49, 50, 92, 206], [279, 280, 322, 426]):
        pts = _pt_indices(pt, indices)
        if pts.shape[0] >= 2:
            cv2.polylines(naso_lines, [pts.astype(np.int32)], False, 1.0, 2, cv2.LINE_AA)
    naso_lines = cv2.GaussianBlur(naso_lines, (7, 7), 0)
    naso_zone = feather(naso_lines) * (1.1 * factor) * (1.0 - protected)
    effect = np.clip(L_norm - naso_lines * 0.28 * factor, 0.0, 1.0)
    apply_luminance(effect, naso_zone)

    # Mouth corners: subtle downward marionette creases.
    marionette = np.zeros((h, w), dtype=np.float32)
    for indices in ([61, 146, 91, 181, 84], [291, 375, 321, 405, 314]):
        pts = _pt_indices(pt, indices)
        if pts.shape[0] >= 2:
            cv2.polylines(marionette, [pts.astype(np.int32)], False, 1.0, 1, cv2.LINE_AA)
    marionette = cv2.GaussianBlur(marionette, (7, 7), 0)
    effect = np.clip(L_norm - marionette * 0.10 * factor, 0.0, 1.0)
    apply_luminance(effect, feather(marionette) * (0.45 * factor) * (1.0 - protected))

    # Cheeks: mild pore/laxity texture, not line scratches.
    rng = np.random.default_rng(12345)
    noise = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    pore = cv2.GaussianBlur(noise, (3, 3), 0.5) - cv2.GaussianBlur(noise, (0, 0), max(1.0, face_width * 0.025))
    pore = (pore - float(np.mean(pore))) / (float(np.std(pore)) + 1e-6)
    pore = np.clip(np.maximum(pore - 0.35, 0.0) * 0.35, 0.0, 1.0)
    cheek_zone = np.zeros((h, w), dtype=np.float32)
    for indices in ([116, 117, 118, 119, 120], [345, 346, 347, 348, 349]):
        pts = _pt_indices(pt, indices)
        if pts.shape[0] > 0:
            c = np.mean(pts, axis=0)
            cheek_zone = np.maximum(cheek_zone, blob(float(c[0]), float(c[1]), 40.0, 40.0))
    cheek_alpha = feather(cheek_zone * (1.0 - protected)) * (0.85 * factor)
    effect = np.clip(L_norm - pore * 0.18 * factor, 0.0, 1.0)
    apply_luminance(effect, cheek_alpha)
    A_new *= 1.0 - cheek_alpha * 0.020 * factor
    B_new *= 1.0 - cheek_alpha * 0.025 * factor

    # Jaw: slight softening/laxity along lower contour.
    jaw = _pt_indices(pt, [172, 136, 150, 149, 176, 148, 152])
    if jaw.shape[0] >= 2:
        jaw_mask = _polyline_ribbon_mask(h, w, jaw, max(5, int(face_width * 0.05)), int(max(21, face_width * 0.12)))
        jaw_alpha = feather(jaw_mask * (1.0 - protected)) * (0.75 * factor)
        softened = cv2.GaussianBlur(L_norm, (0, 0), max(1.0, face_width * 0.018))
        effect = np.clip(softened - jaw_mask * 0.065 * factor, 0.0, 1.0)
        apply_luminance(effect, jaw_alpha)

    hair_mask_f = None
    try:
        from backend.face_parsing import parse_face, get_mask, feather_mask

        parsing = parse_face(image.astype(np.uint8))
        hair_raw = get_mask(parsing, [17])
        hair_mask_f = feather_mask(hair_raw, blur=41).astype(np.float32)
        if float(np.max(hair_mask_f)) <= 1e-4:
            hair_mask_f = None
    except Exception as exc:
        print(f"[Aging] Hair parsing failed: {exc}")
        hair_mask_f = None

    if hair_mask_f is None and pt is not None:
        top_pts = _pt_indices(pt, FOREHEAD_CAP_IDX)
        if top_pts.shape[0] >= 5:
            hair_zone = np.zeros((h, w), dtype=np.float32)
            face_top_y = int(np.clip(np.min(top_pts[:, 1]), 0, h - 1))
            hair_top_y = max(0, face_top_y - int(face_height * 0.35))
            face_left_x = int(np.clip(np.min(top_pts[:, 0]), 0, w - 1))
            face_right_x = int(np.clip(np.max(top_pts[:, 0]), 0, w - 1))
            face_left_x = max(0, face_left_x - int(face_width * 0.15))
            face_right_x = min(w, face_right_x + int(face_width * 0.15))
            if face_right_x > face_left_x and face_top_y > hair_top_y:
                hair_zone[hair_top_y:face_top_y, face_left_x:face_right_x] = 1.0
                hair_mask_f = cv2.GaussianBlur(hair_zone, (51, 51), 0)

    if hair_mask_f is not None:
        hair_mask_f = np.clip(hair_mask_f.astype(np.float32), 0.0, 1.0)
        hair_strength = float(np.clip(factor * 0.85, 0.0, 1.0))
        if hair_strength > 0.05:
            img_bgr_temp = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(img_bgr_temp, cv2.COLOR_BGR2HSV).astype(np.float32)
            H_ch, S_ch, V_ch = cv2.split(hsv)

            S_new = S_ch * (1.0 - hair_mask_f * hair_strength * 0.92)
            V_target = 210.0
            V_new = V_ch + (V_target - V_ch) * hair_mask_f * hair_strength * 0.75
            V_new = np.clip(V_new, 0, 255)
            S_new = np.clip(S_new, 0, 255)

            hsv_new = cv2.merge([
                H_ch.astype(np.float32),
                S_new.astype(np.float32),
                V_new.astype(np.float32),
            ])
            hair_bgr = cv2.cvtColor(hsv_new.astype(np.uint8), cv2.COLOR_HSV2BGR)
            hair_rgb = cv2.cvtColor(hair_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            hair_rgb_01 = np.clip(hair_rgb / 255.0, 0.0, 1.0).astype(np.float32)
            hair_lab = cv2.cvtColor(hair_rgb_01, cv2.COLOR_RGB2Lab)
            H_hair, A_hair, B_hair = cv2.split(hair_lab)

            hair_alpha = hair_mask_f * hair_strength
            L_norm_updated = (
                L_norm * (1.0 - hair_alpha * 0.7)
                + (H_hair / 100.0) * hair_alpha * 0.7
            )
            L_norm = np.clip(L_norm_updated, 0.0, 1.0)
            A_new = (
                A_new * (1.0 - hair_alpha * 0.85)
                + A_hair.astype(np.float32) * hair_alpha * 0.15
            )
            B_new = (
                B_new * (1.0 - hair_alpha * 0.85)
                + B_hair.astype(np.float32) * hair_alpha * 0.15
            )

    lab_out = cv2.merge([
        np.clip(L_norm * 100.0, 1.0, 99.5).astype(np.float32),
        A_new.astype(np.float32),
        B_new.astype(np.float32),
    ])
    return np.clip(cv2.cvtColor(lab_out, cv2.COLOR_Lab2RGB) * 255.0, 0, 255).astype(np.uint8)

def apply_frequency_aging_effect(
    image_bgr: np.ndarray,
    ctx: dict,
    params: dict,
) -> np.ndarray:
    """
    effect_engine contract: BGR in/out.
    Prefer ctx['landmarks_2d']; fallback to params['landmarks'].
    """
    p = params or {}
    intensity = float(np.clip(float(p.get("intensity", 1.0)), 0.0, 2.0))
    landmarks = p.get("landmarks")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    out_rgb = apply_aging_simulation(
        rgb,
        intensity,
        landmarks=landmarks,
    )
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
