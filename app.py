import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from skimage.color import rgb2lab, deltaE_ciede2000
import plotly.graph_objects as go
import plotly.express as px
import json
import io
import base64
from datetime import date

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Lab Wash V6 - 洗浄性評価アプリ",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
def init_session_state():
    if "metadata" not in st.session_state:
        st.session_state.metadata = {
            "exp_id": "EXP-2026-001",
            "sample_name": "Standard Cotton Gauze",
            "operator": "Tester A",
            "condition": "40°C Standard Wash 30min",
            "date": str(date.today())
        }
    if "images" not in st.session_state:
        st.session_state.images = {"base": None, "before": None, "after": None}
    if "weights" not in st.session_state:
        st.session_state.weights = {"base": 0.0, "before": 0.0, "after": 0.0}
    if "calib_enabled" not in st.session_state:
        st.session_state.calib_enabled = False
    if "calib_points" not in st.session_state:
        st.session_state.calib_points = {
            "base": {"black": [15, 15, 15], "white": [240, 240, 240]},
            "before": {"black": [15, 15, 15], "white": [240, 240, 240]},
            "after": {"black": [15, 15, 15], "white": [240, 240, 240]}
        }
    if "roi" not in st.session_state:
        st.session_state.roi = {
            "use_common": True,
            "common": {"x": 10, "y": 10, "w": 80, "h": 80}, # percentage
            "base": {"x": 10, "y": 10, "w": 80, "h": 80},
            "before": {"x": 10, "y": 10, "w": 80, "h": 80},
            "after": {"x": 10, "y": 10, "w": 80, "h": 80}
        }
    if "gauze_params" not in st.session_state:
        st.session_state.gauze_params = {
            "blur_kernel": 5,
            "thresh_offset": 0,
            "morph_kernel": 5,
            "min_area_pct": 5.0,
            "binarize_method": "Otsu",
            "invert_mask": False
        }

init_session_state()

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS & COLOR SCIENCE MATH
# -----------------------------------------------------------------------------

def calibrate_image(img_np, black_rgb, white_rgb):
    """
    2-Point Linear Calibration on RGB image array
    """
    if img_np is None:
        return None
    
    img_float = img_np.astype(np.float32)
    c_black = np.array(black_rgb, dtype=np.float32)
    c_white = np.array(white_rgb, dtype=np.float32)
    
    # Avoid division by zero
    diff = c_white - c_black
    diff = np.where(diff == 0, 1e-5, diff)
    
    calibrated = (img_float - c_black) / diff * 255.0
    calibrated = np.clip(calibrated, 0, 255).astype(np.uint8)
    return calibrated

def get_roi_crop(img_np, roi_pct):
    """
    Crops image according to percentage ROI bounds (x, y, w, h)
    """
    if img_np is None:
        return None
    h_img, w_img = img_np.shape[:2]
    x = int((roi_pct["x"] / 100.0) * w_img)
    y = int((roi_pct["y"] / 100.0) * h_img)
    w = int((roi_pct["w"] / 100.0) * w_img)
    h = int((roi_pct["h"] / 100.0) * h_img)
    
    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    w = max(1, min(w, w_img - x))
    h = max(1, min(h, h_img - y))
    
    return img_np[y:y+h, x:x+w], (x, y, w, h)

def create_gauze_mask(roi_np, params):
    """
    Generates binary mask for gauze area within ROI using OpenCV
    """
    if roi_np is None:
        return None, None
    
    gray = cv2.cvtColor(roi_np, cv2.COLOR_RGB2GRAY)
    
    # 1. Blur
    ksize = params["blur_kernel"]
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    
    # 2. Binarization
    if params["binarize_method"] == "Otsu":
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else: # Adaptive
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
    
    if params.get("invert_mask", False):
        thresh = cv2.bitwise_not(thresh)
        
    # Apply Threshold Offset if needed
    offset = params.get("thresh_offset", 0)
    if offset != 0:
        thresh = np.clip(thresh.astype(np.int16) + offset, 0, 255).astype(np.uint8)

    # 3. Morphology
    m_ksize = max(1, params["morph_kernel"])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (m_ksize, m_ksize))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel)
    
    # 4. Contour filtering
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    
    roi_area = roi_np.shape[0] * roi_np.shape[1]
    min_area = (params["min_area_pct"] / 100.0) * roi_area
    
    valid_contours = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            valid_contours.append(c)
            
    if valid_contours:
        cv2.drawContours(mask, valid_contours, -1, 255, thickness=cv2.FILLED)
    else:
        # Fallback to entire ROI if no valid contour found
        mask[:, :] = 255

    # Visual result with green contour outline
    contour_overlay = roi_np.copy()
    cv2.drawContours(contour_overlay, valid_contours, -1, (0, 255, 0), 2)
    
    return mask, contour_overlay

def compute_metrics(img_np, mask):
    """
    Computes mean L*a*b*, WI, K/S values for pixels within mask
    """
    if img_np is None or mask is None:
        return None
    
    # Filter masked pixels
    masked_pixels = img_np[mask > 0]
    if len(masked_pixels) == 0:
        masked_pixels = img_np.reshape(-1, 3)
        
    # RGB 0..1 float
    rgb_float = masked_pixels.astype(np.float32) / 255.0
    
    # CIE L*a*b* calculation
    lab_pixels = rgb_lab = rgb2lab(rgb_float.reshape(-1, 1, 3)).reshape(-1, 3)
    mean_lab = np.mean(lab_pixels, axis=0)
    L, a, b = mean_lab[0], mean_lab[1], mean_lab[2]
    
    # Whiteness Index (WI, ASTM E313 approx)
    wi = 100.0 - np.sqrt((100.0 - L)**2 + a**2 + b**2)
    
    # Kubelka-Munk K/S calculation
    # Clip R to avoid 0 division
    r_clipped = np.clip(rgb_float, 1e-6, 1.0)
    ks_channels = ((1.0 - r_clipped) ** 2) / (2.0 * r_clipped)
    mean_ks_rgb = np.mean(ks_channels, axis=0)
    
    # Luminance Y
    y = 0.2126 * r_clipped[:, 0] + 0.7152 * r_clipped[:, 1] + 0.0722 * r_clipped[:, 2]
    y_clipped = np.clip(y, 1e-6, 1.0)
    ks_y = np.mean(((1.0 - y_clipped) ** 2) / (2.0 * y_clipped))
    
    return {
        "L": L, "a": a, "b": b,
        "WI": wi,
        "KS_R": mean_ks_rgb[0], "KS_G": mean_ks_rgb[1], "KS_B": mean_ks_rgb[2],
        "KS_Y": ks_y,
        "mean_rgb": np.mean(masked_pixels, axis=0)
    }

# -----------------------------------------------------------------------------
# SIDEBAR: METADATA & CONTROL
# -----------------------------------------------------------------------------
st.sidebar.title("🧪 Lab Wash V6")
st.sidebar.markdown("**試験布・ガーゼ洗浄性評価システム**")
st.sidebar.divider()

st.sidebar.subheader("📋 実験メタデータ")
st.session_state.metadata["exp_id"] = st.sidebar.text_input("実験 ID", st.session_state.metadata["exp_id"])
st.session_state.metadata["sample_name"] = st.sidebar.text_input("試料名 / 布種", st.session_state.metadata["sample_name"])
st.session_state.metadata["operator"] = st.sidebar.text_input("担当者名", st.session_state.metadata["operator"])
st.session_state.metadata["condition"] = st.sidebar.text_area("洗浄条件", st.session_state.metadata["condition"], height=70)
st.session_state.metadata["date"] = st.sidebar.date_input("実施日", date.today()).strftime("%Y-%m-%d")

st.sidebar.divider()
st.sidebar.info("💡 Tab 1から順に処理を進行してください。")

# -----------------------------------------------------------------------------
# MAIN APP: TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "① 画像・重量入力", 
    "② 黒白校正", 
    "③ ROI・ガーゼ抽出", 
    "④ 解析結果・評価", 
    "⑤ レポート・エクスポート"
])

# -----------------------------------------------------------------------------
# TAB 1: IMAGES & WEIGHTS
# -----------------------------------------------------------------------------
with tab1:
    st.header("① 画像読み込み & サンプル重量入力")
    st.markdown("素地布（未汚染）、洗浄前汚染布、洗浄後布の3種類の画像と重量（g）を入力します。")
    
    cols = st.columns(3)
    stages = [
        ("base", "1. 素地布 (Base)", "標準未汚染布"),
        ("before", "2. 洗浄前汚染布 (Before)", "人工汚染布"),
        ("after", "3. 洗洗浄後布 (After)", "洗浄処理後の布")
    ]
    
    for i, (key, title, desc) in enumerate(stages):
        with cols[i]:
            st.subheader(title)
            st.caption(desc)
            
            # Weight input
            st.session_state.weights[key] = st.number_input(
                f"重量 (g) - {key.capitalize()}",
                min_value=0.0,
                value=float(st.session_state.weights[key]),
                format="%.4f",
                key=f"weight_input_{key}"
            )
            
            # File Uploader
            uploaded_file = st.file_uploader(
                f"画像アップロード ({key})",
                type=["png", "jpg", "jpeg", "tif", "tiff"],
                key=f"uploader_{key}"
            )
            
            if uploaded_file is not None:
                img_pil = Image.open(uploaded_file).convert("RGB")
                st.session_state.images[key] = np.array(img_pil)
                
            if st.session_state.images[key] is not None:
                st.image(st.session_state.images[key], caption=f"{title} プレビュー", use_container_width=True)
                st.success(f"画像読み込み完了 ({st.session_state.images[key].shape[1]}x{st.session_state.images[key].shape[0]} px)")
            else:
                st.warning("画像が未選択です")

# Check if images are uploaded helper
def check_images_uploaded():
    return all(st.session_state.images[k] is not None for k in ["base", "before", "after"])

# -----------------------------------------------------------------------------
# TAB 2: BLACK & WHITE CALIBRATION
# -----------------------------------------------------------------------------
with tab2:
    st.header("② 黒白キャリブレーション（2点色彩校正）")
    
    if not check_images_uploaded():
        st.warning("⚠️ 先に【Tab 1: 画像・重量入力】で3種類すべての画像をアップロードしてください。")
    else:
        st.session_state.calib_enabled = st.checkbox(
            "色校正（2点補正）を有効にする", 
            value=st.session_state.calib_enabled
        )
        st.markdown("各画像上の「黒点（ブラックレベル）」および「白点（ホワイトレベル）」の参照RGB値を設定します。")
        
        calib_cols = st.columns(3)
        for i, (key, title, _) in enumerate(stages):
            with calib_cols[i]:
                st.subheader(f"補正設定: {title}")
                img = st.session_state.images[key]
                h_img, w_img = img.shape[:2]
                
                st.markdown("**黒点 (Black Point) 座標**")
                bx = st.slider(f"X (px) - 黒 - {key}", 0, w_img-1, int(w_img*0.05), key=f"bx_{key}")
                by = st.slider(f"Y (px) - 黒 - {key}", 0, h_img-1, int(h_img*0.05), key=f"by_{key}")
                
                # 5x5 region average
                bx_min, bx_max = max(0, bx-2), min(w_img, bx+3)
                by_min, by_max = max(0, by-2), min(h_img, by+3)
                black_rgb = np.mean(img[by_min:by_max, bx_min:bx_max], axis=(0, 1)).astype(int).tolist()
                st.session_state.calib_points[key]["black"] = black_rgb
                st.caption(f"検出黒RGB: `{black_rgb}`")
                
                st.markdown("**白点 (White Point) 座標**")
                wx = st.slider(f"X (px) - 白 - {key}", 0, w_img-1, int(w_img*0.95), key=f"wx_{key}")
                wy = st.slider(f"Y (px) - 白 - {key}", 0, h_img-1, int(h_img*0.95), key=f"wy_{key}")
                
                wx_min, wx_max = max(0, wx-2), min(w_img, wx+3)
                wy_min, wy_max = max(0, wy-2), min(h_img, wy+3)
                white_rgb = np.mean(img[wy_min:wy_max, wx_min:wx_max], axis=(0, 1)).astype(int).tolist()
                st.session_state.calib_points[key]["white"] = white_rgb
                st.caption(f"検出白RGB: `{white_rgb}`")
                
                # Visual preview with markers
                preview_mark = img.copy()
                cv2.rectangle(preview_mark, (bx-5, by-5), (bx+5, by+5), (255, 0, 0), 2)
                cv2.rectangle(preview_mark, (wx-5, wy-5), (wx+5, wy+5), (0, 0, 255), 2)
                
                if st.session_state.calib_enabled:
                    calibrated = calibrate_image(img, black_rgb, white_rgb)
                    st.image(calibrated, caption=f"{key.capitalize()} 補正後プレビュー", use_container_width=True)
                else:
                    st.image(preview_mark, caption=f"{key.capitalize()} 参照点マーク（赤:黒点, 青:白点）", use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: ROI & GAUZE EXTRACTION
# -----------------------------------------------------------------------------
with tab3:
    st.header("③ ROI指定 & OpenCVガーゼ領域自動抽出")
    
    if not check_images_uploaded():
        st.warning("⚠️ 先に【Tab 1: 画像・重量入力】で3種類すべての画像をアップロードしてください。")
    else:
        st.subheader("1. 関心領域 (ROI) 設定")
        st.session_state.roi["use_common"] = st.checkbox(
            "全画像に共通のROI領域を適用する", 
            value=st.session_state.roi["use_common"]
        )
        
        if st.session_state.roi["use_common"]:
            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            st.session_state.roi["common"]["x"] = col_r1.slider("ROI Offset X (%)", 0, 90, st.session_state.roi["common"]["x"])
            st.session_state.roi["common"]["y"] = col_r2.slider("ROI Offset Y (%)", 0, 90, st.session_state.roi["common"]["y"])
            st.session_state.roi["common"]["w"] = col_r3.slider("ROI 幅 Width (%)", 10, 100, st.session_state.roi["common"]["w"])
            st.session_state.roi["common"]["h"] = col_r4.slider("ROI 高さ Height (%)", 10, 100, st.session_state.roi["common"]["h"])
            
            for k in ["base", "before", "after"]:
                st.session_state.roi[k] = st.session_state.roi["common"].copy()
        
        st.divider()
        st.subheader("2. OpenCV ガーゼ抽出 パラメータ設定")
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        
        st.session_state.gauze_params["blur_kernel"] = p_col1.slider("ガウシアンフィルタ Kernel", 1, 15, st.session_state.gauze_params["blur_kernel"], step=2)
        st.session_state.gauze_params["morph_kernel"] = p_col2.slider("モルフォロジー Kernel", 1, 15, st.session_state.gauze_params["morph_kernel"])
        st.session_state.gauze_params["min_area_pct"] = p_col3.slider("最小輪郭面積フィルタ (%)", 0.1, 50.0, st.session_state.gauze_params["min_area_pct"])
        st.session_state.gauze_params["binarize_method"] = p_col4.selectbox("二値化手法", ["Otsu", "Adaptive"], index=0)
        
        st.session_state.gauze_params["invert_mask"] = st.checkbox("二値化マスクを反転する", value=st.session_state.gauze_params["invert_mask"])
        st.session_state.gauze_params["thresh_offset"] = st.slider("二値化閾値オフセット", -50, 50, st.session_state.gauze_params["thresh_offset"])
        
        st.divider()
        st.subheader("3. 抽出結果プレビュー")
        
        roi_cols = st.columns(3)
        st.session_state.gauze_masks = {}
        
        for i, (key, title, _) in enumerate(stages):
            with roi_cols[i]:
                st.write(f"**{title}**")
                
                raw_img = st.session_state.images[key]
                if st.session_state.calib_enabled:
                    raw_img = calibrate_image(raw_img, st.session_state.calib_points[key]["black"], st.session_state.calib_points[key]["white"])
                
                roi_img, _ = get_roi_crop(raw_img, st.session_state.roi[key])
                mask, overlay = create_gauze_mask(roi_img, st.session_state.gauze_params)
                
                st.session_state.gauze_masks[key] = mask
                
                st.image(overlay, caption=f"{key.capitalize()} 抽出領域（緑枠）", use_container_width=True)
                st.image(mask, caption=f"{key.capitalize()} 2値マスク", use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: ANALYSIS & EVALUATION
# -----------------------------------------------------------------------------
with tab4:
    st.header("④ 解析結果 & 洗浄率総合評価")
    
    if not check_images_uploaded():
        st.warning("⚠️ 先に【Tab 1: 画像・重量入力】で3種類すべての画像をアップロードしてください。")
    else:
        # A. Weight Washability Calculation
        w_base = st.session_state.weights["base"]
        w_before = st.session_state.weights["before"]
        w_after = st.session_state.weights["after"]
        
        m_stain = w_before - w_base
        m_removed = w_before - w_after
        
        if m_stain > 0:
            w_weight = (m_removed / m_stain) * 100.0
        else:
            w_weight = 0.0
            
        # B. Color & Optical Computations
        stage_metrics = {}
        for key in ["base", "before", "after"]:
            raw_img = st.session_state.images[key]
            if st.session_state.calib_enabled:
                raw_img = calibrate_image(raw_img, st.session_state.calib_points[key]["black"], st.session_state.calib_points[key]["white"])
            
            roi_img, _ = get_roi_crop(raw_img, st.session_state.roi[key])
            mask = st.session_state.gauze_masks.get(key)
            if mask is None or mask.shape != roi_img.shape[:2]:
                mask, _ = create_gauze_mask(roi_img, st.session_state.gauze_params)
                
            stage_metrics[key] = compute_metrics(roi_img, mask)
            
        # Washability WL (L* based)
        L_base = stage_metrics["base"]["L"]
        L_before = stage_metrics["before"]["L"]
        L_after = stage_metrics["after"]["L"]
        
        L_denom = L_base - L_before
        if abs(L_denom) > 1e-5:
            w_L = ((L_after - L_before) / L_denom) * 100.0
        else:
            w_L = 0.0
            
        # Color difference deltaE 1976 & CIEDE2000
        lab_base = np.array([stage_metrics["base"]["L"], stage_metrics["base"]["a"], stage_metrics["base"]["b"]])
        lab_before = np.array([stage_metrics["before"]["L"], stage_metrics["before"]["a"], stage_metrics["before"]["b"]])
        lab_after = np.array([stage_metrics["after"]["L"], stage_metrics["after"]["a"], stage_metrics["after"]["b"]])
        
        dE76_stain = np.linalg.norm(lab_before - lab_base)
        dE76_wash = np.linalg.norm(lab_after - lab_before)
        dE76_residual = np.linalg.norm(lab_after - lab_base)
        
        dE00_stain = deltaE_ciede2000(lab_base.reshape(1,1,3), lab_before.reshape(1,1,3))[0,0]
        dE00_wash = deltaE_ciede2000(lab_before.reshape(1,1,3), lab_after.reshape(1,1,3))[0,0]
        dE00_residual = deltaE_ciede2000(lab_after.reshape(1,1,3), lab_base.reshape(1,1,3))[0,0]
        
        # K/S reduction rate
        ks_before = stage_metrics["before"]["KS_Y"]
        ks_after = stage_metrics["after"]["KS_Y"]
        ks_base = stage_metrics["base"]["KS_Y"]
        
        ks_denom = ks_before - ks_base
        if abs(ks_denom) > 1e-5:
            w_ks = ((ks_before - ks_after) / ks_denom) * 100.0
        else:
            w_ks = 0.0
            
        # Key Summary Metrics Display
        st.subheader("📊 主要評価指標 Summary")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        
        m_col1.metric("重量基準洗浄率 (W_w)", f"{w_weight:.2f} %", delta=f"除去量: {m_removed:.4f}g")
        m_col2.metric("L*基準洗浄率 (W_L)", f"{w_L:.2f} %", delta=f"ΔL*: {L_after - L_before:+.2f}")
        m_col3.metric("K/S低減率 (輝度Y)", f"{w_ks:.2f} %", delta=f"ΔK/S: {ks_before - ks_after:+.4f}")
        m_col4.metric("色差 CIEDE2000 (対素地)", f"{dE00_residual:.2f}", delta=f"汚染時: {dE00_stain:.2f}", delta_color="inverse")
        
        st.divider()
        st.subheader("📋 詳細解析データ一覧")
        
        data_df = pd.DataFrame({
            "指標": ["L* (明度)", "a* (赤-緑)", "b* (黄-青)", "白さ指数 (WI)", "K/S (R)", "K/S (G)", "K/S (B)", "K/S (Y輝度)"],
            "素地 (Base)": [
                f"{stage_metrics['base']['L']:.2f}", f"{stage_metrics['base']['a']:.2f}", f"{stage_metrics['base']['b']:.2f}",
                f"{stage_metrics['base']['WI']:.2f}", f"{stage_metrics['base']['KS_R']:.4f}", f"{stage_metrics['base']['KS_G']:.4f}",
                f"{stage_metrics['base']['KS_B']:.4f}", f"{stage_metrics['base']['KS_Y']:.4f}"
            ],
            "洗浄前 (Before)": [
                f"{stage_metrics['before']['L']:.2f}", f"{stage_metrics['before']['a']:.2f}", f"{stage_metrics['before']['b']:.2f}",
                f"{stage_metrics['before']['WI']:.2f}", f"{stage_metrics['before']['KS_R']:.4f}", f"{stage_metrics['before']['KS_G']:.4f}",
                f"{stage_metrics['before']['KS_B']:.4f}", f"{stage_metrics['before']['KS_Y']:.4f}"
            ],
            "洗浄後 (After)": [
                f"{stage_metrics['after']['L']:.2f}", f"{stage_metrics['after']['a']:.2f}", f"{stage_metrics['after']['b']:.2f}",
                f"{stage_metrics['after']['WI']:.2f}", f"{stage_metrics['after']['KS_R']:.4f}", f"{stage_metrics['after']['KS_G']:.4f}",
                f"{stage_metrics['after']['KS_B']:.4f}", f"{stage_metrics['after']['KS_Y']:.4f}"
            ]
        })
        st.dataframe(data_df, use_container_width=True)
        
        # Charts
        st.divider()
        st.subheader("📈 視覚的比較グラフ")
        
        g_col1, g_col2 = st.columns(2)
        
        with g_col1:
            fig_lab = go.Figure(data=[
                go.Bar(name='Base', x=['L*', 'a*', 'b*'], y=[lab_base[0], lab_base[1], lab_base[2]]),
                go.Bar(name='Before', x=['L*', 'a*', 'b*'], y=[lab_before[0], lab_before[1], lab_before[2]]),
                go.Bar(name='After', x=['L*', 'a*', 'b*'], y=[lab_after[0], lab_after[1], lab_after[2]])
            ])
            fig_lab.update_layout(title="CIE L*a*b* 表色系 比較", barmode='group', template="plotly_white")
            st.plotly_chart(fig_lab, use_container_width=True)
            
        with g_col2:
            fig_ks = go.Figure(data=[
                go.Bar(name='Base', x=['Red', 'Green', 'Blue', 'Y_Luminance'], y=[stage_metrics['base']['KS_R'], stage_metrics['base']['KS_G'], stage_metrics['base']['KS_B'], stage_metrics['base']['KS_Y']]),
                go.Bar(name='Before', x=['Red', 'Green', 'Blue', 'Y_Luminance'], y=[stage_metrics['before']['KS_R'], stage_metrics['before']['KS_G'], stage_metrics['before']['KS_B'], stage_metrics['before']['KS_Y']]),
                go.Bar(name='After', x=['Red', 'Green', 'Blue', 'Y_Luminance'], y=[stage_metrics['after']['KS_R'], stage_metrics['after']['KS_G'], stage_metrics['after']['KS_B'], stage_metrics['after']['KS_Y']])
            ])
            fig_ks.update_layout(title="K/S値 (Kubelka-Munk) 比較", barmode='group', template="plotly_white")
            st.plotly_chart(fig_ks, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 5: REPORT GENERATION & DATA EXPORT
# -----------------------------------------------------------------------------
with tab5:
    st.header("⑤ レポート自動生成 & データエクスポート")
    
    if not check_images_uploaded():
        st.warning("⚠️ 先に【Tab 1: 画像・重量入力】で3種類すべての画像をアップロードしてください。")
    else:
        st.subheader("📄 PDFレポート発行 (ReportLab)")
        
        def generate_pdf_report():
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
            story = []
            
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'TitleStyle',
                parent=styles['Heading1'],
                fontSize=20,
                leading=24,
                textColor=colors.HexColor("#1E3A8A"),
                spaceAfter=15
            )
            h2_style = ParagraphStyle(
                'H2Style',
                parent=styles['Heading2'],
                fontSize=14,
                leading=18,
                textColor=colors.HexColor("#1E40AF"),
                spaceBefore=12,
                spaceAfter=6
            )
            body_style = styles['Normal']
            
            # Title
            story.append(Paragraph("Lab Wash V6 - 洗浄性評価試験報告書", title_style))
            story.append(Spacer(1, 10))
            
            # Meta Table
            meta_data = [
                [Paragraph("<b>実験ID:</b>", body_style), st.session_state.metadata["exp_id"], Paragraph("<b>実施日:</b>", body_style), st.session_state.metadata["date"]],
                [Paragraph("<b>試料名:</b>", body_style), st.session_state.metadata["sample_name"], Paragraph("<b>担当者:</b>", body_style), st.session_state.metadata["operator"]],
                [Paragraph("<b>洗浄条件:</b>", body_style), st.session_state.metadata["condition"], "", ""]
            ]
            t_meta = Table(meta_data, colWidths=[80, 170, 80, 170])
            t_meta.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F3F4F6")),
                ('TEXTCOLOR', (0,0), (-1,-1), colors.black),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#D1D5DB")),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('SPAN', (1,2), (3,2)),
            ]))
            story.append(t_meta)
            story.append(Spacer(1, 15))
            
            # Summary Table
            story.append(Paragraph("1. 主要洗浄性評価結果", h2_style))
            summary_data = [
                ["評価項目", "計算値", "単位 / 備考"],
                ["重量基準洗浄率 (W_w)", f"{w_weight:.2f}", "%"],
                ["明度基準洗浄率 (W_L)", f"{w_L:.2f}", "%"],
                ["K/S低減率 (Y輝度)", f"{w_ks:.2f}", "%"],
                ["残留色差 ΔE*00 (対素地)", f"{dE00_residual:.2f}", "CIEDE2000"],
                ["汚染時色差 ΔE*00 (汚染-素地)", f"{dE00_stain:.2f}", "CIEDE2000"]
            ]
            t_sum = Table(summary_data, colWidths=[200, 100, 200])
            t_sum.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E40AF")),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#9CA3AF")),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F9FAFB")]),
            ]))
            story.append(t_sum)
            story.append(Spacer(1, 15))
            
            # Detailed Optics Table
            story.append(Paragraph("2. 詳細色彩・光学データ", h2_style))
            opt_data = [
                ["段階", "L*", "a*", "b*", "WI", "K/S (Y)"],
                ["素地 (Base)", f"{stage_metrics['base']['L']:.2f}", f"{stage_metrics['base']['a']:.2f}", f"{stage_metrics['base']['b']:.2f}", f"{stage_metrics['base']['WI']:.2f}", f"{stage_metrics['base']['KS_Y']:.4f}"],
                ["洗浄前 (Before)", f"{stage_metrics['before']['L']:.2f}", f"{stage_metrics['before']['a']:.2f}", f"{stage_metrics['before']['b']:.2f}", f"{stage_metrics['before']['WI']:.2f}", f"{stage_metrics['before']['KS_Y']:.4f}"],
                ["洗浄後 (After)", f"{stage_metrics['after']['L']:.2f}", f"{stage_metrics['after']['a']:.2f}", f"{stage_metrics['after']['b']:.2f}", f"{stage_metrics['after']['WI']:.2f}", f"{stage_metrics['after']['KS_Y']:.4f}"]
            ]
            t_opt = Table(opt_data, colWidths=[100, 80, 80, 80, 80, 80])
            t_opt.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#3B82F6")),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#9CA3AF")),
            ]))
            story.append(t_opt)
            
            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()

        pdf_bytes = generate_pdf_report()
        st.download_button(
            label="📥 A4 PDFレポートをダウンロード",
            data=pdf_bytes,
            file_name=f"lab_wash_report_{st.session_state.metadata['exp_id']}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        
        st.divider()
        st.subheader("💾 データ エクスポート & インポート")
        
        col_ex1, col_ex2 = st.columns(2)
        
        # CSV Export
        export_df = pd.DataFrame([{
            **st.session_state.metadata,
            "W_weight_%": w_weight,
            "W_L_%": w_L,
            "W_KS_%": w_ks,
            "dE00_residual": dE00_residual,
            "dE00_stain": dE00_stain,
            "weight_base_g": w_base,
            "weight_before_g": w_before,
            "weight_after_g": w_after,
            "L_base": stage_metrics["base"]["L"],
            "L_before": stage_metrics["before"]["L"],
            "L_after": stage_metrics["after"]["L"],
        }])
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        
        col_ex1.download_button(
            label="📄 解析データ (CSV) ダウンロード",
            data=csv_bytes,
            file_name=f"lab_wash_data_{st.session_state.metadata['exp_id']}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # JSON Export
        json_export_data = {
            "metadata": st.session_state.metadata,
            "weights": st.session_state.weights,
            "calib_points": st.session_state.calib_points,
            "roi": st.session_state.roi,
            "gauze_params": st.session_state.gauze_params,
            "results": {
                "W_weight": w_weight,
                "W_L": w_L,
                "W_KS": w_ks,
                "dE00_residual": dE00_residual,
                "stage_metrics": {
                    k: {m: float(v) if isinstance(v, (np.floating, float)) else v.tolist() if isinstance(v, np.ndarray) else v for m, v in stage_metrics[k].items()}
                    for k in stage_metrics
                }
            }
        }
        json_bytes = json.dumps(json_export_data, indent=2, ensure_ascii=False).encode("utf-8")
        
        col_ex2.download_button(
            label="🌐 フル設定・データ (JSON) ダウンロード",
            data=json_bytes,
            file_name=f"lab_wash_config_{st.session_state.metadata['exp_id']}.json",
            mime="application/json",
            use_container_width=True
        )
