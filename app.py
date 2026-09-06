import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000
import plotly.graph_objects as go
import json
import io
from datetime import date

# 外部UIコンポーネント
from streamlit_cropper import st_cropper
from streamlit_image_coordinates import streamlit_image_coordinates

# ReportLab imports (PDF生成用)
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
# 日本語フォント対応 (CIDFont)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

# Streamlit Page Configuration
st.set_page_config(
    page_title="Lab Wash V6 - 洗浄性評価アプリ",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 1. CUSTOM JSON ENCODER (TypeError 修正)
# -----------------------------------------------------------------------------
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, (pd.Timestamp, date)):
            return str(obj)
        if isinstance(obj, Image.Image):
            return "PIL.Image Object"
        return super(NumpyEncoder, self).default(obj)


# -----------------------------------------------------------------------------
# 2. SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
def init_session_state():
    if "metadata" not in st.session_state:
        st.session_state.metadata = {
            "exp_id": "EXP-2026-001",
            "sample_name": "Standard Cotton Gauze",
            "operator": "Tester A",
            "condition": "40°C Standard Wash 30min",
            "date": date.today()
        }
    if "images" not in st.session_state:
        st.session_state.images = {"base": None, "before": None, "after": None}
    if "weights" not in st.session_state:
        st.session_state.weights = {"base": 0.0, "before": 0.0, "after": 0.0}
    if "calib_enabled" not in st.session_state:
        st.session_state.calib_enabled = False
    
    # 2点校正用の座標・RGB値
    if "calib_points" not in st.session_state:
        st.session_state.calib_points = {k: {"black": [15,15,15], "white": [240,240,240], "b_pos": None, "w_pos": None} for k in ["base", "before", "after"]}
    
    # ROI指定情報 (streamlit-cropperのbox座標)
    if "roi_box" not in st.session_state:
        st.session_state.roi_box = None
        
    # ガーゼ抽出パラメータ
    if "gauze_params" not in st.session_state:
        st.session_state.gauze_params = {
            "blur_kernel": 5, "morph_kernel": 7, "min_area_pct": 5.0,
            "binarize_method": "Otsu", "use_convex_hull": True
        }

init_session_state()


# -----------------------------------------------------------------------------
# 3. CORE IMAGE PROCESSING & MATH FUNCTIONS
# -----------------------------------------------------------------------------
def calibrate_image(img_np, black_rgb, white_rgb):
    """ 2-Point Linear Calibration """
    if img_np is None: return None
    img_float = img_np.astype(np.float32)
    c_black = np.array(black_rgb, dtype=np.float32)
    c_white = np.array(white_rgb, dtype=np.float32)
    
    diff = c_white - c_black
    diff = np.where(diff == 0, 1e-5, diff)  # Zero division prevention
    
    calibrated = (img_float - c_black) / diff * 255.0
    return np.clip(calibrated, 0, 255).astype(np.uint8)

def get_roi_crop(img_np, box):
    """ streamlit-cropper の box {'left', 'top', 'width', 'height'} を用いてクロップ """
    if img_np is None or box is None: return img_np
    h_img, w_img = img_np.shape[:2]
    left = max(0, min(box['left'], w_img - 1))
    top = max(0, min(box['top'], h_img - 1))
    width = max(1, min(box['width'], w_img - left))
    height = max(1, min(box['height'], h_img - top))
    return img_np[top:top+height, left:left+width]

def create_gauze_mask(roi_np, params):
    """ OpenCVによるいびつなガーゼ輪郭自動抽出 (Convex Hull対応) """
    if roi_np is None: return None, None
    gray = cv2.cvtColor(roi_np, cv2.COLOR_RGB2GRAY)
    
    # 1. Blur
    ksize = params["blur_kernel"] if params["blur_kernel"] % 2 != 0 else params["blur_kernel"] + 1
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    
    # 2. Binarization
    if params["binarize_method"] == "Otsu":
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # 3. Morphology (Closing -> Opening による網目やノイズ補正)
    m_ksize = max(1, params["morph_kernel"])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (m_ksize, m_ksize))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel)
    
    # 4. Contour & Convex Hull (いびつな形状対応)
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    
    roi_area = roi_np.shape[0] * roi_np.shape[1]
    min_area = (params["min_area_pct"] / 100.0) * roi_area
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    
    contour_overlay = roi_np.copy()
    
    if valid_contours:
        # 最大の輪郭を取得
        largest_contour = max(valid_contours, key=cv2.contourArea)
        
        if params.get("use_convex_hull", True):
            # 凸包 (Convex Hull) を計算してスムージング
            hull = cv2.convexHull(largest_contour)
            cv2.drawContours(mask, [hull], -1, 255, thickness=cv2.FILLED)
            cv2.drawContours(contour_overlay, [hull], -1, (0, 255, 0), 3) # 緑の枠
        else:
            cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            cv2.drawContours(contour_overlay, [largest_contour], -1, (0, 255, 0), 3)
    else:
        mask[:, :] = 255 # フォールバック

    return mask, contour_overlay

def compute_metrics(img_np, mask):
    """ L*a*b*, WI, K/S 計算 """
    if img_np is None or mask is None: return None
    masked_pixels = img_np[mask > 0]
    if len(masked_pixels) == 0: masked_pixels = img_np.reshape(-1, 3)
        
    rgb_float = masked_pixels.astype(np.float32) / 255.0
    lab_pixels = rgb2lab(rgb_float.reshape(-1, 1, 3)).reshape(-1, 3)
    mean_lab = np.mean(lab_pixels, axis=0)
    L, a, b = mean_lab[0], mean_lab[1], mean_lab[2]
    
    wi = 100.0 - np.sqrt((100.0 - L)**2 + a**2 + b**2)
    
    r_clipped = np.clip(rgb_float, 1e-6, 1.0)
    ks_channels = ((1.0 - r_clipped) ** 2) / (2.0 * r_clipped)
    mean_ks_rgb = np.mean(ks_channels, axis=0)
    
    y = 0.2126 * r_clipped[:, 0] + 0.7152 * r_clipped[:, 1] + 0.0722 * r_clipped[:, 2]
    y_clipped = np.clip(y, 1e-6, 1.0)
    ks_y = np.mean(((1.0 - y_clipped) ** 2) / (2.0 * y_clipped))
    
    return {
        "L": L, "a": a, "b": b, "WI": wi,
        "KS_R": mean_ks_rgb[0], "KS_G": mean_ks_rgb[1], "KS_B": mean_ks_rgb[2], "KS_Y": ks_y
    }

# -----------------------------------------------------------------------------
# 4. UI: SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.title("🧪 Lab Wash V6")
st.sidebar.markdown("**試験布・ガーゼ洗浄性評価システム**")
st.sidebar.divider()

st.sidebar.subheader("📋 実験メタデータ")
st.session_state.metadata["exp_id"] = st.sidebar.text_input("実験 ID", st.session_state.metadata["exp_id"])
st.session_state.metadata["sample_name"] = st.sidebar.text_input("試料名 / 布種", st.session_state.metadata["sample_name"])
st.session_state.metadata["operator"] = st.sidebar.text_input("担当者名", st.session_state.metadata["operator"])
st.session_state.metadata["condition"] = st.sidebar.text_area("洗浄条件", st.session_state.metadata["condition"], height=70)
st.session_state.metadata["date"] = st.sidebar.date_input("実施日", st.session_state.metadata["date"])

def check_images_uploaded():
    return all(st.session_state.images[k] is not None for k in ["base", "before", "after"])

# -----------------------------------------------------------------------------
# 5. UI: MAIN TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "① 画像・重量", "② 黒白校正 (タップ選択)", "③ ROI (ドラッグ)・ガーゼ抽出", "④ 解析結果・評価", "⑤ レポート・エクスポート"
])
stages = [("base", "素地布 (Base)"), ("before", "洗浄前汚染布 (Before)"), ("after", "洗浄後布 (After)")]

# --- TAB 1: IMAGES & WEIGHTS ---
with tab1:
    cols = st.columns(3)
    for i, (key, title) in enumerate(stages):
        with cols[i]:
            st.subheader(title)
            st.session_state.weights[key] = st.number_input(f"重量 (g) - {key}", min_value=0.0, value=float(st.session_state.weights[key]), format="%.4f")
            uploaded_file = st.file_uploader(f"画像アップロード ({key})", type=["png", "jpg", "jpeg", "tif"])
            if uploaded_file is not None:
                img_pil = Image.open(uploaded_file).convert("RGB")
                st.session_state.images[key] = np.array(img_pil)
            if st.session_state.images[key] is not None:
                st.image(st.session_state.images[key], use_container_width=True)

# --- TAB 2: BLACK & WHITE CALIBRATION (TAP TO SELECT) ---
with tab2:
    if not check_images_uploaded():
        st.warning("⚠️ 先に Tab 1 で3種類すべての画像をアップロードしてください。")
    else:
        st.session_state.calib_enabled = st.checkbox("✅ 色校正（2点補正）を有効にする", value=st.session_state.calib_enabled)
        st.info("画像上を直接タップ（クリック）して基準点を取得します。ラジオボタンで「黒点」または「白点」を選んでからタップしてください。")
        
        calib_cols = st.columns(3)
        for i, (key, title) in enumerate(stages):
            with calib_cols[i]:
                st.write(f"**{title}**")
                mode = st.radio(f"取得モード ({key})", ["黒点 (Black) を取得", "白点 (White) を取得"], horizontal=True, key=f"radio_{key}")
                
                # インタラクティブ画像タップUI
                img = st.session_state.images[key]
                coord = streamlit_image_coordinates(img, key=f"coord_{key}", use_column_width=True)
                
                h, w = img.shape[:2]
                if coord is not None:
                    cx, cy = coord["x"], coord["y"]
                    # 5x5 平均を取得
                    x_min, x_max = max(0, cx-2), min(w, cx+3)
                    y_min, y_max = max(0, cy-2), min(h, cy+3)
                    avg_rgb = np.mean(img[y_min:y_max, x_min:x_max], axis=(0, 1)).astype(int).tolist()
                    
                    if "黒点" in mode:
                        st.session_state.calib_points[key]["black"] = avg_rgb
                        st.session_state.calib_points[key]["b_pos"] = (cx, cy)
                    else:
                        st.session_state.calib_points[key]["white"] = avg_rgb
                        st.session_state.calib_points[key]["w_pos"] = (cx, cy)

                # 現在の設定値を表示
                st.caption(f"⚫ 黒RGB: {st.session_state.calib_points[key]['black']}")
                st.caption(f"⚪ 白RGB: {st.session_state.calib_points[key]['white']}")
                
                if st.session_state.calib_enabled:
                    calibrated = calibrate_image(img, st.session_state.calib_points[key]["black"], st.session_state.calib_points[key]["white"])
                    st.image(calibrated, caption="補正後プレビュー", use_container_width=True)

# --- TAB 3: ROI (DRAG) & GAUZE EXTRACTION ---
with tab3:
    if not check_images_uploaded():
        st.warning("⚠️ 先に Tab 1 で3種類すべての画像をアップロードしてください。")
    else:
        st.subheader("1. 関心領域 (ROI) の指定（ドラッグ操作）")
        st.markdown("素地布 (Base) の画像上で青い枠をドラッグして、全画像共通の解析エリアを指定してください。")
        
        # ドラッグ用 Cropper (代表としてBase画像を表示)
        base_img = st.session_state.images["base"]
        if st.session_state.calib_enabled:
             base_img = calibrate_image(base_img, st.session_state.calib_points["base"]["black"], st.session_state.calib_points["base"]["white"])
        base_pil = Image.fromarray(base_img)
        
        # Cropper (box座標のみ取得)
        box = st_cropper(base_pil, realtime_update=True, box_color='blue', return_type='box')
        st.session_state.roi_box = box
        
        st.divider()
        st.subheader("2. 自動ガーゼ検出設定 (いびつな形状対応)")
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        st.session_state.gauze_params["blur_kernel"] = p_col1.slider("ガウシアンフィルタ Kernel", 1, 15, st.session_state.gauze_params["blur_kernel"], step=2)
        st.session_state.gauze_params["morph_kernel"] = p_col2.slider("モルフォロジー補正 Kernel", 1, 15, st.session_state.gauze_params["morph_kernel"])
        st.session_state.gauze_params["min_area_pct"] = p_col3.slider("最小面積フィルタ (%)", 0.1, 50.0, st.session_state.gauze_params["min_area_pct"])
        st.session_state.gauze_params["use_convex_hull"] = p_col4.checkbox("凸包 (Convex Hull) スムージングを適用", value=st.session_state.gauze_params["use_convex_hull"])
        
        st.divider()
        st.subheader("3. 抽出結果プレビュー")
        roi_cols = st.columns(3)
        st.session_state.gauze_masks = {}
        st.session_state.roi_images = {}
        
        for i, (key, title) in enumerate(stages):
            with roi_cols[i]:
                st.write(f"**{title}**")
                raw_img = st.session_state.images[key]
                if st.session_state.calib_enabled:
                    raw_img = calibrate_image(raw_img, st.session_state.calib_points[key]["black"], st.session_state.calib_points[key]["white"])
                
                roi_img = get_roi_crop(raw_img, st.session_state.roi_box)
                st.session_state.roi_images[key] = roi_img
                
                mask, overlay = create_gauze_mask(roi_img, st.session_state.gauze_params)
                st.session_state.gauze_masks[key] = mask
                
                st.image(overlay, caption="抽出領域（緑枠）", use_container_width=True)

# --- TAB 4: ANALYSIS & EVALUATION ---
with tab4:
    if not check_images_uploaded() or st.session_state.roi_box is None:
        st.warning("⚠️ Tab 1 で画像入力、Tab 3 でROI設定を完了してください。")
    else:
        w_base = st.session_state.weights["base"]
        w_before = st.session_state.weights["before"]
        w_after = st.session_state.weights["after"]
        
        m_stain = w_before - w_base
        m_removed = w_before - w_after
        w_weight = (m_removed / m_stain * 100.0) if m_stain > 0 else 0.0
            
        stage_metrics = {}
        for key in ["base", "before", "after"]:
            stage_metrics[key] = compute_metrics(st.session_state.roi_images[key], st.session_state.gauze_masks[key])
            
        L_denom = stage_metrics["base"]["L"] - stage_metrics["before"]["L"]
        w_L = ((stage_metrics["after"]["L"] - stage_metrics["before"]["L"]) / L_denom * 100.0) if abs(L_denom) > 1e-5 else 0.0
            
        lab_base = np.array([stage_metrics["base"]["L"], stage_metrics["base"]["a"], stage_metrics["base"]["b"]])
        lab_before = np.array([stage_metrics["before"]["L"], stage_metrics["before"]["a"], stage_metrics["before"]["b"]])
        lab_after = np.array([stage_metrics["after"]["L"], stage_metrics["after"]["a"], stage_metrics["after"]["b"]])
        
        dE00_residual = deltaE_ciede2000(lab_after.reshape(1,1,3), lab_base.reshape(1,1,3))[0,0]
        dE00_stain = deltaE_ciede2000(lab_base.reshape(1,1,3), lab_before.reshape(1,1,3))[0,0]
        
        ks_denom = stage_metrics["before"]["KS_Y"] - stage_metrics["base"]["KS_Y"]
        w_ks = ((stage_metrics["before"]["KS_Y"] - stage_metrics["after"]["KS_Y"]) / ks_denom * 100.0) if abs(ks_denom) > 1e-5 else 0.0
            
        st.subheader("📊 主要評価指標 Summary")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("重量基準洗浄率 (W_w)", f"{w_weight:.2f} %")
        m_col2.metric("L*基準洗浄率 (W_L)", f"{w_L:.2f} %")
        m_col3.metric("K/S低減率 (輝度Y)", f"{w_ks:.2f} %")
        m_col4.metric("残留色差 ΔE*00", f"{dE00_residual:.2f}", delta_color="inverse")
        
        data_df = pd.DataFrame({
            "指標": ["L*", "a*", "b*", "WI", "K/S (Y輝度)"],
            "素地": [f"{stage_metrics['base']['L']:.2f}", f"{stage_metrics['base']['a']:.2f}", f"{stage_metrics['base']['b']:.2f}", f"{stage_metrics['base']['WI']:.2f}", f"{stage_metrics['base']['KS_Y']:.4f}"],
            "洗浄前": [f"{stage_metrics['before']['L']:.2f}", f"{stage_metrics['before']['a']:.2f}", f"{stage_metrics['before']['b']:.2f}", f"{stage_metrics['before']['WI']:.2f}", f"{stage_metrics['before']['KS_Y']:.4f}"],
            "洗浄後": [f"{stage_metrics['after']['L']:.2f}", f"{stage_metrics['after']['a']:.2f}", f"{stage_metrics['after']['b']:.2f}", f"{stage_metrics['after']['WI']:.2f}", f"{stage_metrics['after']['KS_Y']:.4f}"]
        })
        st.dataframe(data_df, use_container_width=True)

# --- TAB 5: REPORT & EXPORT ---
with tab5:
    if not check_images_uploaded() or st.session_state.roi_box is None:
        st.warning("⚠️ Tab 1~4の処理を完了してください。")
    else:
        st.subheader("📄 PDFレポート発行 (日本語・画像対応)")
        
        def np_to_rl_image(img_np, max_w=120):
            """ NumPy配列を ReportLab用画像に変換 """
            if img_np is None: return ""
            pil_img = Image.fromarray(img_np)
            buf = io.BytesIO()
            pil_img.save(buf, format='PNG')
            buf.seek(0)
            aspect = pil_img.height / pil_img.width
            return RLImage(buf, width=max_w, height=max_w * aspect)

        def generate_pdf_report():
            # ReportLabの標準CIDフォント(日本語)を登録
            pdfmetrics.registerFont(UnicodeCIDFont('HeiseiKakuGo-W5'))
            
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
            story = []
            
            styles = getSampleStyleSheet()
            # 日本語対応スタイルを作成
            title_style = ParagraphStyle('TitleJ', parent=styles['Heading1'], fontName='HeiseiKakuGo-W5', fontSize=18, textColor=colors.HexColor("#1E3A8A"), spaceAfter=15)
            h2_style = ParagraphStyle('H2J', parent=styles['Heading2'], fontName='HeiseiKakuGo-W5', fontSize=14, spaceBefore=12, spaceAfter=6)
            body_style = ParagraphStyle('BodyJ', parent=styles['Normal'], fontName='HeiseiKakuGo-W5', fontSize=10)
            
            story.append(Paragraph("Lab Wash V6 - 洗浄性評価試験報告書", title_style))
            story.append(Spacer(1, 10))
            
            # メタデータ表
            meta_data = [
                [Paragraph("実験ID:", body_style), Paragraph(st.session_state.metadata["exp_id"], body_style), Paragraph("実施日:", body_style), Paragraph(str(st.session_state.metadata["date"]), body_style)],
                [Paragraph("試料名:", body_style), Paragraph(st.session_state.metadata["sample_name"], body_style), Paragraph("担当者:", body_style), Paragraph(st.session_state.metadata["operator"], body_style)],
                [Paragraph("洗浄条件:", body_style), Paragraph(st.session_state.metadata["condition"], body_style), "", ""]
            ]
            t_meta = Table(meta_data, colWidths=[60, 190, 60, 190])
            t_meta.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F3F4F6")),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#D1D5DB")),
                ('SPAN', (1,2), (3,2)),
                ('FONTNAME', (0,0), (-1,-1), 'HeiseiKakuGo-W5')
            ]))
            story.append(t_meta)
            
            # 評価結果表
            story.append(Paragraph("1. 主要洗浄性評価結果", h2_style))
            summary_data = [
                [Paragraph("評価項目", body_style), Paragraph("計算値", body_style)],
                [Paragraph("重量基準洗浄率 (W_w) %", body_style), f"{w_weight:.2f}"],
                [Paragraph("明度基準洗浄率 (W_L) %", body_style), f"{w_L:.2f}"],
                [Paragraph("K/S低減率 (Y輝度) %", body_style), f"{w_ks:.2f}"],
                [Paragraph("残留色差 ΔE*00", body_style), f"{dE00_residual:.2f}"]
            ]
            t_sum = Table(summary_data, colWidths=[200, 150])
            t_sum.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('FONTNAME', (0,0), (-1,-1), 'HeiseiKakuGo-W5')]))
            story.append(t_sum)
            
            # ガーゼ画像一覧
            story.append(Paragraph("2. 抽出ガーゼ画像 (ROI内)", h2_style))
            img_data = [
                [Paragraph("素地 (Base)", body_style), Paragraph("洗浄前 (Before)", body_style), Paragraph("洗浄後 (After)", body_style)],
                [np_to_rl_image(st.session_state.roi_images["base"]), np_to_rl_image(st.session_state.roi_images["before"]), np_to_rl_image(st.session_state.roi_images["after"])]
            ]
            t_img = Table(img_data, colWidths=[160, 160, 160])
            t_img.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('FONTNAME', (0,0), (-1,-1), 'HeiseiKakuGo-W5')]))
            story.append(t_img)
            
            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()

        pdf_bytes = generate_pdf_report()
        st.download_button("📥 A4 PDFレポートをダウンロード", data=pdf_bytes, file_name=f"report_{st.session_state.metadata['exp_id']}.pdf", mime="application/pdf")
        
        st.divider()
        st.subheader("💾 データ エクスポート (JSON TypeError 解決済)")
        
        json_export_data = {
            "metadata": st.session_state.metadata,
            "weights": st.session_state.weights,
            "calib_points": st.session_state.calib_points,
            "roi_box": st.session_state.roi_box,
            "results": {
                "W_weight": w_weight, "W_L": w_L, "W_KS": w_ks, "dE00_residual": dE00_residual,
                "stage_metrics": stage_metrics
            }
        }
        
        # カスタムエンコーダー(NumpyEncoder)を用いて安全にJSON化
        json_str = json.dumps(json_export_data, cls=NumpyEncoder, indent=2, ensure_ascii=False)
        st.download_button("🌐 フルデータ (JSON) ダウンロード", data=json_str.encode("utf-8"), file_name=f"data_{st.session_state.metadata['exp_id']}.json", mime="application/json")
