# Lab Wash V6 - ガーゼ・布の洗浄評価・画像解析アプリ
# Streamlit + OpenCV + scikit-image 堅牢版
# 実行: pip install streamlit opencv-python-headless pillow scikit-image pandas reportlab plotly
#       streamlit run app.py

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import json
import io
import datetime
from typing import Dict, Tuple, Optional

# scikit-image
try:
    from skimage.color import rgb2lab, rgb2xyz
except ImportError:
    rgb2lab = None
    rgb2xyz = None

# reportlab
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
except ImportError:
    SimpleDocTemplate = None

# --- Page Config ---
st.set_page_config(
    page_title="Lab Wash V6",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Utils ---
def safe_divide(a, b, default=0.0):
    try:
        if b == 0 or b is None or np.abs(b) < 1e-9:
            return default
        return a / b
    except Exception:
        return default

def pil_to_cv(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def cv_to_pil(cv_img: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def pil_to_np(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img).astype(np.float32)

def apply_bw_calibration(pil_img: Image.Image, black_rgb: Tuple[int,int,int], white_rgb: Tuple[int,int,int]) -> Image.Image:
    """2点校正: (img - black) / (white - black) * 255"""
    try:
        img = pil_to_np(pil_img)  # HWC
        black = np.array(black_rgb, dtype=np.float32).reshape(1,1,3)
        white = np.array(white_rgb, dtype=np.float32).reshape(1,1,3)
        denom = white - black
        denom = np.where(np.abs(denom) < 1e-6, 1.0, denom)  # ゼロ除算回避
        corrected = (img - black) / denom * 255.0
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        return Image.fromarray(corrected)
    except Exception as e:
        st.warning(f"色補正失敗: {e}")
        return pil_img

def extract_bw_auto(pil_img: Image.Image, percentile: int = 2) -> Tuple[Tuple[int,int,int], Tuple[int,int,int]]:
    """画像から自動で黒点・白点を推定（パーセンタイル法）"""
    try:
        arr = np.array(pil_img).reshape(-1,3)
        # 下位・上位パーセンタイルを平均
        low = np.percentile(arr, percentile, axis=0)
        high = np.percentile(arr, 100-percentile, axis=0)
        # 極端な値から少しマージン
        black = tuple(np.clip(low, 0, 255).astype(int).tolist())
        white = tuple(np.clip(high, 0, 255).astype(int).tolist())
        return black, white
    except Exception:
        return (0,0,0), (255,255,255)

def detect_gauze_mask(roi_pil: Image.Image, blur_k: int=5, thresh: int=0, use_otsu: bool=True, morph_k: int=7, invert: bool=False, min_area_ratio: float=0.1) -> Tuple[np.ndarray, Image.Image, Image.Image]:
    """
    ROI内でガーゼのいびつな外形を検出してマスク作成
    Returns: mask (uint8 0/255), overlay PIL, binary PIL
    """
    try:
        cv_img = pil_to_cv(roi_pil)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        # blur
        k = max(1, blur_k)
        if k % 2 == 0:
            k += 1
        blurred = cv2.GaussianBlur(gray, (k,k), 0)

        # threshold
        if use_otsu:
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY)

        if invert:
            binary = cv2.bitwise_not(binary)

        # morphology: close -> open でコブ・裁断歪みを保持しつつノイズ除去
        mk = max(1, morph_k)
        if mk % 2 == 0:
            mk += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

        # 輪郭抽出
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            # フォールバック: 全面マスク
            mask = np.ones_like(gray) * 255
            overlay = roi_pil.copy()
            return mask, overlay, Image.fromarray(binary)

        # 面積でフィルタ
        h, w = gray.shape
        total_area = h * w
        filtered = [c for c in contours if cv2.contourArea(c) > total_area * min_area_ratio * 0.01]
        if not filtered:
            filtered = contours

        # 最大輪郭をメイン、他も含める（ガーゼが複数片の場合）
        filtered = sorted(filtered, key=cv2.contourArea, reverse=True)
        # 上位3つまで統合（いびつな形状対応）
        mask = np.zeros_like(gray)
        for c in filtered[:3]:
            cv2.drawContours(mask, [c], -1, 255, -1)
            # 凸包も少しブレンドしてコブ対応
            hull = cv2.convexHull(c)
            cv2.drawContours(mask, [hull], -1, 255, -1)

        # 最終的に少し膨張させて縁取りを確実に含む
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # オーバーレイ作成
        overlay_cv = cv_img.copy()
        colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        overlay_cv = cv2.addWeighted(overlay_cv, 0.7, colored_mask, 0.3, 0)

        # 輪郭線描画
        final_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_cv, final_contours, -1, (0,255,0), 2)

        return mask, cv_to_pil(overlay_cv), Image.fromarray(opened)
    except Exception as e:
        st.warning(f"マスク検出エラー: {e}")
        h, w = roi_pil.size[1], roi_pil.size[0]
        dummy_mask = np.ones((h, w), dtype=np.uint8) * 255
        return dummy_mask, roi_pil, roi_pil

def get_mean_rgb_lab_xyz(pil_img: Image.Image, mask: Optional[np.ndarray]=None) -> Dict:
    """領域平均の RGB, Lab, XYZ を算出"""
    try:
        if rgb2lab is None:
            raise ImportError("scikit-image not installed")
        img_np = np.array(pil_img)  # uint8
        if mask is not None:
            # mask: 0/255, サイズ合わせ
            if mask.shape[:2] != img_np.shape[:2]:
                mask_resized = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                mask_resized = mask
            valid = mask_resized > 127
            if np.sum(valid) < 10:
                valid = np.ones((img_np.shape[0], img_np.shape[1]), dtype=bool)
        else:
            valid = np.ones((img_np.shape[0], img_np.shape[1]), dtype=bool)

        # RGB mean
        mean_rgb = np.mean(img_np[valid], axis=0)  # R,G,B 0-255
        # Lab用に0-1正規化
        rgb_norm = img_np.astype(np.float32) / 255.0
        lab = rgb2lab(rgb_norm)  # L 0-100
        xyz = rgb2xyz(rgb_norm)  # 0-1

        mean_lab = np.mean(lab[valid], axis=0) if lab[valid].size else np.array([0,0,0])
        mean_xyz = np.mean(xyz[valid], axis=0) if xyz[valid].size else np.array([0,0,0])

        return {
            "R": float(mean_rgb[0]),
            "G": float(mean_rgb[1]),
            "B": float(mean_rgb[2]),
            "L": float(mean_lab[0]),
            "a": float(mean_lab[1]),
            "b": float(mean_lab[2]),
            "X": float(mean_xyz[0]*100),
            "Y": float(mean_xyz[1]*100),
            "Z": float(mean_xyz[2]*100),
            "valid_pixels": int(np.sum(valid))
        }
    except Exception as e:
        st.warning(f"色彩計算エラー: {e}")
        return {"R":0,"G":0,"B":0,"L":0,"a":0,"b":0,"X":0,"Y":0,"Z":0,"valid_pixels":0}

def calc_deltaE76(lab1: Dict, lab2: Dict) -> float:
    try:
        return float(np.sqrt((lab1["L"]-lab2["L"])**2 + (lab1["a"]-lab2["a"])**2 + (lab1["b"]-lab2["b"])**2))
    except:
        return 0.0

def calc_deltaE2000(lab1: Dict, lab2: Dict, kL=1, kC=1, kH=1) -> float:
    """CIEDE2000 実装 - Sharma et al. 2005"""
    try:
        L1, a1, b1 = lab1["L"], lab1["a"], lab1["b"]
        L2, a2, b2 = lab2["L"], lab2["a"], lab2["b"]

        C1 = np.sqrt(a1**2 + b1**2)
        C2 = np.sqrt(a2**2 + b2**2)
        C_bar = (C1 + C2) / 2.0

        G = 0.5 * (1 - np.sqrt((C_bar**7) / (C_bar**7 + 25**7 + 1e-12)))

        a1p = (1+G)*a1
        a2p = (1+G)*a2

        C1p = np.sqrt(a1p**2 + b1**2)
        C2p = np.sqrt(a2p**2 + b2**2)

        # h'
        def hp(ap,b):
            h = np.degrees(np.arctan2(b, ap))
            return h + 360 if h < 0 else h
        h1p = hp(a1p, b1)
        h2p = hp(a2p, b2)

        dLp = L2 - L1
        dCp = C2p - C1p

        # dhp
        dhp = 0.0
        if C1p*C2p < 1e-12:
            dhp = 0
        else:
            dh = h2p - h1p
            if np.abs(dh) <= 180:
                dhp = dh
            elif dh > 180:
                dhp = dh - 360
            else:
                dhp = dh + 360
        dHp = 2 * np.sqrt(C1p*C2p) * np.sin(np.radians(dhp/2))

        # 平均
        Lp_bar = (L1+L2)/2
        Cp_bar = (C1p+C2p)/2

        if C1p*C2p < 1e-12:
            hp_bar = h1p + h2p
        else:
            dh = np.abs(h1p - h2p)
            if dh <= 180:
                hp_bar = (h1p + h2p)/2
            else:
                if h1p + h2p < 360:
                    hp_bar = (h1p + h2p + 360)/2
                else:
                    hp_bar = (h1p + h2p - 360)/2

        # T
        T = 1 - 0.17*np.cos(np.radians(hp_bar-30)) + 0.24*np.cos(np.radians(2*hp_bar)) + 0.32*np.cos(np.radians(3*hp_bar+6)) - 0.20*np.cos(np.radians(4*hp_bar-63))

        # delta theta
        dtheta = 30 * np.exp(-((hp_bar-275)/25)**2)

        # RC
        RC = 2 * np.sqrt((Cp_bar**7)/(Cp_bar**7 + 25**7 + 1e-12))

        # SL, SC, SH
        SL = 1 + (0.015 * (Lp_bar - 50)**2) / np.sqrt(20 + (Lp_bar - 50)**2 + 1e-12)
        SC = 1 + 0.045 * Cp_bar
        SH = 1 + 0.015 * Cp_bar * T

        # RT
        RT = -np.sin(np.radians(2*dtheta)) * RC

        dE = np.sqrt(
            (dLp/(kL*SL))**2 +
            (dCp/(kC*SC))**2 +
            (dHp/(kH*SH))**2 +
            RT * (dCp/(kC*SC)) * (dHp/(kH*SH))
        )
        return float(dE)
    except Exception:
        return 0.0

def calc_KS(R: float) -> float:
    """K/S = (1-R)^2 / 2R, R: 0-1 reflectance. 入力は0-1想定"""
    try:
        R = np.clip(R, 0.001, 0.999)
        return float(((1-R)**2) / (2*R))
    except:
        return 0.0

def calc_WI_ASTM_E313(XYZ: Dict) -> float:
    """ASTM E313 WI 近似: 3.388*Z -3*Y, Y,Z 0-100"""
    try:
        Y = XYZ["Y"]
        Z = XYZ["Z"]
        return float(3.388 * Z - 3.0 * Y)
    except:
        return 0.0

def calc_WI_CIE(XYZ: Dict) -> float:
    """CIE Whiteness: WI = Y +800(xn - x)+1700(yn - y), D65 2deg xn=0.3127 yn=0.3290"""
    try:
        X, Y, Z = XYZ["X"], XYZ["Y"], XYZ["Z"]
        s = X + Y + Z + 1e-12
        x = X / s
        y = Y / s
        xn, yn = 0.3127, 0.3290
        WI = Y + 800*(xn - x) + 1700*(yn - y)
        return float(WI)
    except:
        return 0.0

def generate_pdf_buffer(metadata, weights, bw_points, results_df, lab_data, wash_rates, images_dict, calibrated_dict):
    """ReportLabでPDF生成、BytesIOを返す"""
    try:
        if SimpleDocTemplate is None:
            st.error("reportlabがインストールされていません")
            return None
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=20*mm, leftMargin=20*mm, topMargin=15*mm, bottomMargin=15*mm)
        styles = getSampleStyleSheet()
        story = []

        title = Paragraph("<b>Lab Wash V6 洗浄評価レポート</b>", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 10*mm))

        # メタデータテーブル
        meta_rows = [
            ["実験ID", metadata.get("exp_id","") , "実施日", str(metadata.get("date",""))],
            ["試料名", metadata.get("sample_name",""), "担当者", metadata.get("operator","")],
            ["洗浄温度", f"{metadata.get('temp','')} ℃", "洗浄時間", f"{metadata.get('time','')} min"],
            ["洗剤濃度", f"{metadata.get('detergent','')} %", "", ""],
        ]
        t = Table(meta_rows, colWidths=[25*mm, 50*mm, 25*mm, 50*mm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#E8F0FE")),
            ('BACKGROUND', (2,0), (2,-1), colors.HexColor("#E8F0FE")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ]))
        story.append(Paragraph("<b>1. 実験条件</b>", styles['Heading2']))
        story.append(t)
        story.append(Spacer(1, 8*mm))

        # 重量・洗浄率
        story.append(Paragraph("<b>2. 重量評価</b>", styles['Heading2']))
        weight_rows = [
            ["項目", "素地", "洗浄前", "洗浄後"],
            ["重量(g)", f"{weights.get('素地',0):.4f}", f"{weights.get('洗浄前',0):.4f}", f"{weights.get('洗浄後',0):.4f}"],
            ["汚染量(g)", f"{wash_rates.get('contamination',0):.4f}", "", ""],
            ["除去量(g)", f"{wash_rates.get('removed',0):.4f}", "", ""],
            ["重量洗浄率(%)", f"{wash_rates.get('weight_rate',0):.2f}", "", ""],
        ]
        wt = Table(weight_rows, colWidths=[35*mm, 35*mm, 35*mm, 35*mm])
        wt.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#D0E0FF")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('FONTSIZE', (0,0), (-1,-1), 8),
        ]))
        story.append(wt)
        story.append(Spacer(1, 8*mm))

        # 色彩評価テーブル
        story.append(Paragraph("<b>3. 色彩・光学評価</b>", styles['Heading2']))
        if results_df is not None and not results_df.empty:
            # DataFrameをTableに変換
            df = results_df.copy()
            # 数値は丸める
            header = ["指標"] + list(df.columns)
            table_data = [header]
            for idx, row in df.iterrows():
                table_data.append([str(idx)] + [f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v) for v in row.values])
            ct = Table(table_data, colWidths=[35*mm] + [25*mm]*(len(header)-1))
            ct.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#FFE0B2")),
                ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#FFF3E0")),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('FONTSIZE', (0,0), (-1,-1), 7),
            ]))
            story.append(ct)
        story.append(Spacer(1, 5*mm))

        # Lab詳細
        if lab_data:
            lab_rows = [["試料", "L*", "a*", "b*", "ΔEab*", "ΔE00", "K/S", "WI_ASTM", "WI_CIE"]]
            for key in ["素地","洗浄前","洗浄後"]:
                d = lab_data.get(key, {})
                lab_rows.append([
                    key,
                    f"{d.get('L',0):.2f}", f"{d.get('a',0):.2f}", f"{d.get('b',0):.2f}",
                    f"{d.get('dE76',0):.2f}" if key!="素地" else "-",
                    f"{d.get('dE00',0):.2f}" if key!="素地" else "-",
                    f"{d.get('KS',0):.4f}",
                    f"{d.get('WI_ASTM',0):.2f}",
                    f"{d.get('WI_CIE',0):.2f}",
                ])
            lt = Table(lab_rows, colWidths=[15*mm] + [18*mm]*8)
            lt.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#C8E6C9")),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('FONTSIZE', (0,0), (-1,-1), 7),
            ]))
            story.append(Spacer(1, 3*mm))
            story.append(lt)

        story.append(Spacer(1, 8*mm))
        story.append(Paragraph("<b>4. 画像</b>", styles['Heading2']))

        # 画像配置 (最大3枚)
        try:
            from reportlab.lib.utils import ImageReader
            img_tables = []
            for key in ["素地","洗浄前","洗浄後"]:
                pil_img = calibrated_dict.get(key) or images_dict.get(key)
                if pil_img:
                    # リサイズしてバッファへ
                    tmp = pil_img.copy()
                    tmp.thumbnail((300,300))
                    b = io.BytesIO()
                    tmp.save(b, format="PNG")
                    b.seek(0)
                    rl_img = RLImage(b, width=45*mm, height=45*mm, kind='proportional')
                    img_tables.append([Paragraph(f"<b>{key}</b>", styles['Normal']), rl_img])
            if img_tables:
                # 3列に
                it = Table([ [row[1] for row in img_tables] ], colWidths=[50*mm]*len(img_tables))
                ht = Table([ [row[0] for row in img_tables] ], colWidths=[50*mm]*len(img_tables))
                story.append(ht)
                story.append(it)
        except Exception as e:
            story.append(Paragraph(f"画像埋め込み失敗: {e}", styles['Normal']))

        story.append(Spacer(1, 10*mm))
        story.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Lab Wash V6", styles['Normal']))

        doc.build(story)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"PDF生成エラー: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

# --- Session State Init ---
def init_state():
    defaults = {
        "metadata": {
            "exp_id": "EXP-001",
            "sample_name": "ガーゼ試料A",
            "operator": "",
            "temp": 40.0,
            "time": 10.0,
            "detergent": 0.1,
            "date": datetime.date.today()
        },
        "images": {"素地": None, "洗浄前": None, "洗浄後": None},
        "weights": {"素地": 1.0, "洗浄前": 1.5, "洗浄後": 1.1},
        "black_white_points": {
            "素地": {"black": (0,0,0), "white": (255,255,255)},
            "洗浄前": {"black": (0,0,0), "white": (255,255,255)},
            "洗浄後": {"black": (0,0,0), "white": (255,255,255)},
        },
        "roi": {
            "common": {"x": 5, "y": 5, "w": 90, "h": 90, "use_common": True},
            "素地": {"x": 5, "y": 5, "w": 90, "h": 90},
            "洗浄前": {"x": 5, "y": 5, "w": 90, "h": 90},
            "洗浄後": {"x": 5, "y": 5, "w": 90, "h": 90},
        },
        "calibrated_images": {"素地": None, "洗浄前": None, "洗浄後": None},
        "masks": {"素地": None, "洗浄前": None, "洗浄後": None},
        "results": None,
        "results_df": None,
        "lab_data": {},
        "wash_rates": {},
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# --- Sidebar ---
with st.sidebar:
    st.title("🧪 Lab Wash V6")
    st.caption("ガーゼ・布の洗浄評価")
    st.divider()
    st.markdown("**使い方**\n1. メタデータと画像を入力\n2. 黒白校正\n3. ROIと自動検出\n4. 解析実行\n5. レポート出力")
    st.divider()
    if st.button("🔄 全状態リセット", type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        init_state()
        st.rerun()
    st.info("対応: PC / iPad\n画像はドラッグ&ドロップ、貼り付け対応")

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "① メタデータ & 画像",
    "② 黒白校正",
    "③ ROI & 自動検出",
    "④ 解析 & 評価",
    "⑤ レポート出力"
])

# ===== Tab1 =====
with tab1:
    st.subheader("ステップ1: メタデータ入力 & 画像読み込み")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.metadata["exp_id"] = st.text_input("実験ID", value=st.session_state.metadata.get("exp_id",""))
        st.session_state.metadata["sample_name"] = st.text_input("試料名", value=st.session_state.metadata.get("sample_name",""))
        st.session_state.metadata["operator"] = st.text_input("担当者名", value=st.session_state.metadata.get("operator",""))
    with c2:
        st.session_state.metadata["temp"] = st.number_input("洗浄温度 (℃)", value=float(st.session_state.metadata.get("temp",40)), step=1.0)
        st.session_state.metadata["time"] = st.number_input("洗浄時間 (min)", value=float(st.session_state.metadata.get("time",10)), step=1.0)
        st.session_state.metadata["detergent"] = st.number_input("洗剤濃度 (%)", value=float(st.session_state.metadata.get("detergent",0.1)), step=0.05, format="%.3f")
        st.session_state.metadata["date"] = st.date_input("実施日", value=st.session_state.metadata.get("date", datetime.date.today()))

    st.divider()
    st.markdown("### 画像アップロード（素地 / 洗浄前 / 洗浄後）")
    st.caption("💡 各アップローダーにドラッグ&ドロップ、またはクリップボードからCtrl+Vで貼り付け可能（ブラウザ対応）")

    cols = st.columns(3)
    for idx, key in enumerate(["素地","洗浄前","洗浄後"]):
        with cols[idx]:
            st.markdown(f"**{key}**")
            uploaded = st.file_uploader(f"{key}画像", type=["png","jpg","jpeg","bmp","tiff"], key=f"upload_{key}", label_visibility="collapsed")
            if uploaded:
                try:
                    pil_img = Image.open(uploaded).convert("RGB")
                    st.session_state.images[key] = pil_img
                    st.session_state.calibrated_images[key] = None  # リセット
                    st.session_state.masks[key] = None
                except Exception as e:
                    st.error(f"画像読み込み失敗: {e}")

            # 既存画像表示
            if st.session_state.images[key] is not None:
                st.image(st.session_state.images[key], caption=f"{key} ({st.session_state.images[key].size[0]}x{st.session_state.images[key].size[1]})", use_column_width=True)
                # 重量入力
                st.session_state.weights[key] = st.number_input(f"{key} 重量 (g)", value=float(st.session_state.weights.get(key,0.0)), step=0.001, format="%.4f", key=f"weight_{key}")
            else:
                st.info(f"{key} 画像未登録")
                st.session_state.weights[key] = st.number_input(f"{key} 重量 (g)", value=float(st.session_state.weights.get(key,0.0)), step=0.001, format="%.4f", key=f"weight_{key}_empty")

    # JSONインポート（画像以外）
    with st.expander("📥 JSONからメタデータ・重量をインポート"):
        json_file = st.file_uploader("JSONファイル", type=["json"], key="json_import_meta")
        if json_file:
            try:
                data = json.load(json_file)
                if "metadata" in data:
                    md = data["metadata"]
                    # date は文字列→date変換
                    if "date" in md and isinstance(md["date"], str):
                        try:
                            md["date"] = datetime.datetime.strptime(md["date"], "%Y-%m-%d").date()
                        except:
                            md["date"] = datetime.date.today()
                    st.session_state.metadata.update(md)
                if "weights" in data:
                    st.session_state.weights.update(data["weights"])
                if "black_white_points" in data:
                    st.session_state.black_white_points.update(data["black_white_points"])
                if "roi" in data:
                    st.session_state.roi.update(data["roi"])
                st.success("JSONインポート成功。リロードしてください。")
                st.json(data)
            except Exception as e:
                st.error(f"JSON読込エラー: {e}")

# ===== Tab2 =====
with tab2:
    st.subheader("ステップ2: 黒白キャリブレーション（2点色補正）")
    st.caption("黒点=0、白点=255として線形補正。代表色の自動抽出または手動設定。")

    if all(v is None for v in st.session_state.images.values()):
        st.warning("まず①で画像を登録してください。")
    else:
        for key in ["素地","洗浄前","洗浄後"]:
            pil_img = st.session_state.images.get(key)
            if pil_img is None:
                continue
            st.markdown(f"#### {key}")
            c1, c2, c3 = st.columns([1,1,2])

            # 自動抽出
            with c1:
                if st.button(f"自動抽出 ({key})", key=f"auto_bw_{key}"):
                    b,w = extract_bw_auto(pil_img, percentile=2)
                    st.session_state.black_white_points[key]["black"] = b
                    st.session_state.black_white_points[key]["white"] = w
                    st.rerun()

                # 現在値取得
                cur_black = st.session_state.black_white_points[key]["black"]
                cur_white = st.session_state.black_white_points[key]["white"]

                st.markdown("**黒点 RGB**")
                # カラーピッカーと数値入力併用
                black_hex = "#%02x%02x%02x" % cur_black
                picked_black_hex = st.color_picker(f"黒点カラー {key}", value=black_hex, key=f"black_picker_{key}")
                # hex to rgb
                try:
                    r = int(picked_black_hex[1:3],16); g = int(picked_black_hex[3:5],16); b = int(picked_black_hex[5:7],16)
                    # 数値入力も同期
                    br = st.number_input(f"R_black {key}", 0,255, r, key=f"br_{key}")
                    bg = st.number_input(f"G_black {key}", 0,255, g, key=f"bg_{key}")
                    bb = st.number_input(f"B_black {key}", 0,255, b, key=f"bb_{key}")
                    st.session_state.black_white_points[key]["black"] = (br,bg,bb)
                except:
                    pass

            with c2:
                st.markdown("**白点 RGB**")
                cur_white = st.session_state.black_white_points[key]["white"]
                white_hex = "#%02x%02x%02x" % cur_white
                picked_white_hex = st.color_picker(f"白点カラー {key}", value=white_hex, key=f"white_picker_{key}")
                try:
                    r = int(picked_white_hex[1:3],16); g = int(picked_white_hex[3:5],16); b = int(picked_white_hex[5:7],16)
                    wr = st.number_input(f"R_white {key}", 0,255, r, key=f"wr_{key}")
                    wg = st.number_input(f"G_white {key}", 0,255, g, key=f"wg_{key}")
                    wb = st.number_input(f"B_white {key}", 0,255, b, key=f"wb_{key}")
                    st.session_state.black_white_points[key]["white"] = (wr,wg,wb)
                except:
                    pass

            with c3:
                black = st.session_state.black_white_points[key]["black"]
                white = st.session_state.black_white_points[key]["white"]
                try:
                    calibrated = apply_bw_calibration(pil_img, black, white)
                    st.session_state.calibrated_images[key] = calibrated
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.image(pil_img, caption="元画像", use_column_width=True)
                    with col_b:
                        st.image(calibrated, caption=f"補正後 B{black} W{white}", use_column_width=True)
                except Exception as e:
                    st.error(f"補正エラー: {e}")

            st.divider()

# ===== Tab3 =====
with tab3:
    st.subheader("ステップ3: ROI指定 & ガーゼ領域自動検出")
    st.caption("ROI内でのみ解析。OpenCVでいびつな外形・コブを高精度抽出。")

    has_image = any(v is not None for v in st.session_state.images.values())
    if not has_image:
        st.warning("画像がありません。①で登録してください。")
    else:
        # 共通ROI設定
        use_common = st.checkbox("共通ROIを使用（全画像で同じ領域）", value=st.session_state.roi["common"].get("use_common", True))
        st.session_state.roi["common"]["use_common"] = use_common

        # 検出パラメータ
        with st.expander("🔧 自動検出パラメータ", expanded=True):
            pc1, pc2, pc3, pc4 = st.columns(4)
            with pc1:
                blur_k = st.slider("Blur Kernel", 1, 21, 5, step=2, key="blur_k")
                morph_k = st.slider("Morph Kernel", 1, 31, 9, step=2, key="morph_k")
            with pc2:
                use_otsu = st.checkbox("Otsu自動閾値", value=True, key="use_otsu")
                thresh_val = st.slider("手動閾値", 0, 255, 127, key="thresh_val", disabled=use_otsu)
                invert_mask = st.checkbox("白黒反転", value=False, key="invert_mask", help="背景が白い場合にON")
            with pc3:
                min_area_ratio = st.slider("最小面積率 (%)", 1, 50, 5, key="min_area", help="ROIに対してこの%未満の小領域は除外")
            with pc4:
                st.info("いびつな裁断面やコブはモルフォロジーCloseで保持されます。感度はMorph Kernelで調整。")

        # ROI入力UI
        def roi_ui(label, default_dict, key_prefix):
            st.markdown(f"**{label} ROI (%)** - 画像サイズに対する相対位置")
            c1,c2,c3,c4 = st.columns(4)
            with c1:
                x = st.slider(f"x {label}", 0, 90, int(default_dict.get("x",5)), key=f"x_{key_prefix}")
            with c2:
                y = st.slider(f"y {label}", 0, 90, int(default_dict.get("y",5)), key=f"y_{key_prefix}")
            with c3:
                w = st.slider(f"w {label}", 10, 100, int(default_dict.get("w",90)), key=f"w_{key_prefix}")
            with c4:
                h = st.slider(f"h {label}", 10, 100, int(default_dict.get("h",90)), key=f"h_{key_prefix}")
            return {"x":x,"y":y,"w":w,"h":h}

        if use_common:
            common_roi = roi_ui("共通", st.session_state.roi["common"], "common")
            st.session_state.roi["common"].update(common_roi)
            # 全画像に適用して表示
            for key in ["素地","洗浄前","洗浄後"]:
                src_img = st.session_state.calibrated_images.get(key) or st.session_state.images.get(key)
                if src_img is None:
                    continue
                st.markdown(f"##### {key}")
                W,H = src_img.size
                rx = int(W * common_roi["x"]/100)
                ry = int(H * common_roi["y"]/100)
                rw = int(W * common_roi["w"]/100)
                rh = int(H * common_roi["h"]/100)
                # 安全クリップ
                rx = max(0, min(rx, W-10)); ry = max(0, min(ry, H-10))
                rw = max(10, min(rw, W-rx)); rh = max(10, min(rh, H-ry))

                # ROI crop
                roi_pil = src_img.crop((rx, ry, rx+rw, ry+rh))

                # 検出
                mask, overlay, binary = detect_gauze_mask(roi_pil, blur_k=blur_k, thresh=thresh_val, use_otsu=use_otsu, morph_k=morph_k, invert=invert_mask, min_area_ratio=min_area_ratio)

                # フルサイズマスク作成（元画像サイズに配置）
                full_mask = np.zeros((H,W), dtype=np.uint8)
                full_mask[ry:ry+rh, rx:rx+rw] = mask

                st.session_state.masks[key] = full_mask

                c1,c2,c3,c4 = st.columns(4)
                with c1:
                    # ROI枠描画
                    cv_full = pil_to_cv(src_img)
                    cv2.rectangle(cv_full, (rx,ry), (rx+rw, ry+rh), (0,255,0), 3)
                    st.image(cv_to_pil(cv_full), caption="ROI位置", use_column_width=True)
                with c2:
                    st.image(roi_pil, caption="ROIクロップ", use_column_width=True)
                with c3:
                    st.image(binary, caption="二値化", use_column_width=True)
                with c4:
                    st.image(overlay, caption="マスクオーバーレイ", use_column_width=True)

        else:
            # 画像ごとのROI
            for key in ["素地","洗浄前","洗浄後"]:
                src_img = st.session_state.calibrated_images.get(key) or st.session_state.images.get(key)
                if src_img is None:
                    continue
                st.markdown(f"##### {key}")
                default = st.session_state.roi.get(key, {"x":5,"y":5,"w":90,"h":90})
                per_roi = roi_ui(key, default, key)
                st.session_state.roi[key] = per_roi

                W,H = src_img.size
                rx = int(W * per_roi["x"]/100)
                ry = int(H * per_roi["y"]/100)
                rw = int(W * per_roi["w"]/100)
                rh = int(H * per_roi["h"]/100)
                rx = max(0, min(rx, W-10)); ry = max(0, min(ry, H-10))
                rw = max(10, min(rw, W-rx)); rh = max(10, min(rh, H-ry))

                roi_pil = src_img.crop((rx, ry, rx+rw, ry+rh))
                mask, overlay, binary = detect_gauze_mask(roi_pil, blur_k=blur_k, thresh=thresh_val, use_otsu=use_otsu, morph_k=morph_k, invert=invert_mask, min_area_ratio=min_area_ratio)
                full_mask = np.zeros((H,W), dtype=np.uint8)
                full_mask[ry:ry+rh, rx:rx+rw] = mask
                st.session_state.masks[key] = full_mask

                c1,c2,c3,c4 = st.columns(4)
                with c1:
                    cv_full = pil_to_cv(src_img)
                    cv2.rectangle(cv_full, (rx,ry), (rx+rw, ry+rh), (0,255,0), 3)
                    st.image(cv_to_pil(cv_full), caption="ROI位置", use_column_width=True)
                with c2:
                    st.image(roi_pil, caption="ROIクロップ", use_column_width=True)
                with c3:
                    st.image(binary, caption="二値化", use_column_width=True)
                with c4:
                    st.image(overlay, caption="マスクオーバーレイ", use_column_width=True)

# ===== Tab4 =====
with tab4:
    st.subheader("ステップ4: 解析・洗浄率評価")
    st.caption("重量・色彩・光学指標を統合評価。ゼロ除算は安全に回避。")

    if rgb2lab is None:
        st.error("scikit-imageが未インストールです。 pip install scikit-image")
    else:
        if st.button("🔬 解析実行", type="primary"):
            try:
                # 重量洗浄率
                w_soil_free = float(st.session_state.weights.get("素地",0))
                w_before = float(st.session_state.weights.get("洗浄前",0))
                w_after = float(st.session_state.weights.get("洗浄後",0))

                contamination = w_before - w_soil_free
                removed = w_before - w_after
                weight_rate = safe_divide(removed, contamination, 0.0) * 100.0

                wash_rates = {
                    "contamination": contamination,
                    "removed": removed,
                    "weight_rate": weight_rate,
                    "w_soil_free": w_soil_free,
                    "w_before": w_before,
                    "w_after": w_after,
                }

                # 色彩評価
                lab_data = {}
                for key in ["素地","洗浄前","洗浄後"]:
                    src_img = st.session_state.calibrated_images.get(key) or st.session_state.images.get(key)
                    if src_img is None:
                        lab_data[key] = {"L":0,"a":0,"b":0,"R":0,"G":0,"B":0,"X":0,"Y":0,"Z":0}
                        continue
                    mask = st.session_state.masks.get(key)
                    info = get_mean_rgb_lab_xyz(src_img, mask)
                    # K/S, WI
                    # 反射率RはY/100を反射率近似として使用（より正確にはRGB平均でも可）
                    R_ref = safe_divide(info["Y"], 100.0, 0.5)
                    # 0-1クリップ
                    R_ref = float(np.clip(R_ref, 0.001, 0.999))
                    # K/SはRから、別途RGB平均反射率も参考値として
                    R_rgb = float(np.mean([info["R"], info["G"], info["B"]]) / 255.0)
                    R_rgb = float(np.clip(R_rgb, 0.001, 0.999))
                    KS = calc_KS(R_ref)
                    KS_rgb = calc_KS(R_rgb)
                    WI_ASTM = calc_WI_ASTM_E313(info)
                    WI_CIE = calc_WI_CIE(info)

                    lab_data[key] = {
                        **info,
                        "KS": KS,
                        "KS_rgb": KS_rgb,
                        "WI_ASTM": WI_ASTM,
                        "WI_CIE": WI_CIE,
                        "R_ref": R_ref,
                    }

                # ΔE, L*洗浄率
                # 基準は素地と洗浄前
                Ls = lab_data.get("素地", {}).get("L",0)
                Lb = lab_data.get("洗浄前", {}).get("L",0)
                La = lab_data.get("洗浄後", {}).get("L",0)

                L_rate = safe_divide((La - Lb), (Ls - Lb), 0.0) * 100.0

                # 色差
                for key in ["洗浄前","洗浄後"]:
                    if key in lab_data and "素地" in lab_data:
                        d76 = calc_deltaE76(lab_data["素地"], lab_data[key])
                        d00 = calc_deltaE2000(lab_data["素地"], lab_data[key])
                        lab_data[key]["dE76_vs_soil"] = d76
                        lab_data[key]["dE00_vs_soil"] = d00
                # 洗浄前 vs 洗浄後
                if "洗浄前" in lab_data and "洗浄後" in lab_data:
                    lab_data["洗浄後"]["dE76_vs_before"] = calc_deltaE76(lab_data["洗浄前"], lab_data["洗浄後"])
                    lab_data["洗浄後"]["dE00_vs_before"] = calc_deltaE2000(lab_data["洗浄前"], lab_data["洗浄後"])
                    lab_data["洗浄前"]["dE76_vs_before"] = 0.0
                    lab_data["洗浄前"]["dE00_vs_before"] = 0.0
                    lab_data["素地"]["dE76_vs_before"] = calc_deltaE76(lab_data["洗浄前"], lab_data["素地"])
                    lab_data["素地"]["dE00_vs_before"] = calc_deltaE2000(lab_data["洗浄前"], lab_data["素地"])

                # 結果DataFrame作成
                rows = []
                for key in ["素地","洗浄前","洗浄後"]:
                    d = lab_data.get(key, {})
                    rows.append({
                        "試料": key,
                        "重量(g)": wash_rates.get(f"w_{'soil_free' if key=='素地' else 'before' if key=='洗浄前' else 'after'}", 0),
                        "L*": d.get("L",0),
                        "a*": d.get("a",0),
                        "b*": d.get("b",0),
                        "R(反射率)": d.get("R_ref",0),
                        "K/S": d.get("KS",0),
                        "WI_ASTM": d.get("WI_ASTM",0),
                        "WI_CIE": d.get("WI_CIE",0),
                        "ΔE*ab vs素地": d.get("dE76_vs_soil", 0) if key!="素地" else 0,
                        "ΔE00 vs素地": d.get("dE00_vs_soil", 0) if key!="素地" else 0,
                    })
                # 重量行追加
                df_main = pd.DataFrame(rows).set_index("試料")

                # 洗浄率サマリ
                summary = pd.DataFrame([
                    {"指標": "重量洗浄率 (%)", "値": weight_rate},
                    {"指標": "L*基準洗浄率 (%)", "値": L_rate},
                    {"指標": "汚染量 (g)", "値": contamination},
                    {"指標": "除去量 (g)", "値": removed},
                    {"指標": "ΔE*ab 洗浄前→洗浄後", "値": lab_data.get("洗浄後",{}).get("dE76_vs_before",0)},
                    {"指標": "ΔE00 洗浄前→洗浄後", "値": lab_data.get("洗浄後",{}).get("dE00_vs_before",0)},
                ]).set_index("指標")

                st.session_state.results_df = df_main
                st.session_state.lab_data = lab_data
                st.session_state.wash_rates = {**wash_rates, "L_rate": L_rate}
                st.session_state.results = {"df_main": df_main, "summary": summary}

                st.success("解析完了")

            except Exception as e:
                st.error(f"解析エラー: {e}")
                import traceback
                st.code(traceback.format_exc())

        # 結果表示
        if st.session_state.get("results_df") is not None:
            st.divider()
            st.markdown("### 📊 解析結果")

            c1,c2 = st.columns([2,1])
            with c1:
                st.dataframe(st.session_state.results_df.style.format("{:.3f}"), use_container_width=True)
            with c2:
                if "results" in st.session_state and st.session_state.results:
                    st.dataframe(st.session_state.results["summary"].style.format("{:.3f}"), use_container_width=True)

            # グラフ
            try:
                import plotly.express as px
                df_plot = st.session_state.results_df.reset_index()
                # L*比較
                fig1 = px.bar(df_plot, x="試料", y="L*", title="L* (明度) 比較", color="試料")
                st.plotly_chart(fig1, use_container_width=True)

                fig2 = px.bar(df_plot, x="試料", y="K/S", title="K/S (Kubelka-Munk) 比較", color="試料")
                st.plotly_chart(fig2, use_container_width=True)

                # 洗浄率
                rates = st.session_state.wash_rates
                rate_df = pd.DataFrame({
                    "指標": ["重量洗浄率","L*洗浄率"],
                    "洗浄率(%)": [rates.get("weight_rate",0), rates.get("L_rate",0)]
                })
                fig3 = px.bar(rate_df, x="指標", y="洗浄率(%)", title="洗浄率比較", color="指標")
                st.plotly_chart(fig3, use_container_width=True)

            except ImportError:
                st.bar_chart(st.session_state.results_df[["L*","K/S"]])
                st.bar_chart(pd.DataFrame({
                    "重量洗浄率": [st.session_state.wash_rates.get("weight_rate",0)],
                    "L*洗浄率": [st.session_state.wash_rates.get("L_rate",0)]
                }))

            # 画像比較
            st.markdown("#### 🖼️ 洗浄前後比較")
            cols = st.columns(3)
            for idx,key in enumerate(["素地","洗浄前","洗浄後"]):
                with cols[idx]:
                    img = st.session_state.calibrated_images.get(key) or st.session_state.images.get(key)
                    if img:
                        mask = st.session_state.masks.get(key)
                        if mask is not None:
                            # マスク適用プレビュー
                            cv_img = pil_to_cv(img)
                            mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_BONE)
                            blended = cv2.addWeighted(cv_img, 0.8, mask_color, 0.2, 0)
                            st.image(cv_to_pil(blended), caption=f"{key} +マスク", use_column_width=True)
                        else:
                            st.image(img, caption=key, use_column_width=True)

# ===== Tab5 =====
with tab5:
    st.subheader("ステップ5: レポート出力・データ保存")
    if st.session_state.get("results_df") is None:
        st.warning("先に④で解析を実行してください。")
    else:
        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown("**PDFレポート**")
            if st.button("📄 PDF生成", type="primary"):
                with st.spinner("PDF生成中..."):
                    pdf_buffer = generate_pdf_buffer(
                        metadata=st.session_state.metadata,
                        weights=st.session_state.weights,
                        bw_points=st.session_state.black_white_points,
                        results_df=st.session_state.results_df,
                        lab_data=st.session_state.lab_data,
                        wash_rates=st.session_state.wash_rates,
                        images_dict=st.session_state.images,
                        calibrated_dict=st.session_state.calibrated_images
                    )
                    if pdf_buffer:
                        st.session_state["pdf_buffer"] = pdf_buffer
                        st.success("PDF生成完了。下のボタンからダウンロード。")

            if "pdf_buffer" in st.session_state and st.session_state["pdf_buffer"]:
                st.download_button(
                    label="⬇️ PDFをダウンロード",
                    data=st.session_state["pdf_buffer"],
                    file_name=f"{st.session_state.metadata.get('exp_id','report')}_LabWashV6.pdf",
                    mime="application/pdf"
                )

        with c2:
            st.markdown("**CSV出力**")
            df = st.session_state.results_df
            csv = df.to_csv(encoding="utf-8-sig")
            st.download_button("⬇️ 結果CSV", data=csv, file_name=f"{st.session_state.metadata.get('exp_id','report')}_results.csv", mime="text/csv")

            # サマリCSV
            summary_csv = st.session_state.results["summary"].to_csv(encoding="utf-8-sig")
            st.download_button("⬇️ サマリCSV", data=summary_csv, file_name=f"{st.session_state.metadata.get('exp_id','report')}_summary.csv", mime="text/csv")

        with c3:
            st.markdown("**JSON出力（全データ）**")
            # JSONは日付を文字列化
            meta_copy = st.session_state.metadata.copy()
            if isinstance(meta_copy.get("date"), (datetime.date, datetime.datetime)):
                meta_copy["date"] = meta_copy["date"].isoformat()

            export_data = {
                "metadata": meta_copy,
                "weights": st.session_state.weights,
                "black_white_points": st.session_state.black_white_points,
                "roi": st.session_state.roi,
                "lab_data": st.session_state.lab_data,
                "wash_rates": st.session_state.wash_rates,
                "results_table": st.session_state.results_df.to_dict(orient="index"),
            }
            json_str = json.dumps(export_data, ensure_ascii=False, indent=2, default=str)
            st.download_button("⬇️ JSONエクスポート", data=json_str, file_name=f"{st.session_state.metadata.get('exp_id','report')}_LabWashV6.json", mime="application/json")

            st.markdown("**JSONインポート**")
            st.caption("①で既に実装済み、ここでもインポート可能")
            jf = st.file_uploader("JSONインポート（再解析用）", type=["json"], key="json_import_tab5")
            if jf:
                try:
                    data = json.load(jf)
                    if "metadata" in data:
                        md = data["metadata"]
                        if "date" in md and isinstance(md["date"], str):
                            try:
                                md["date"] = datetime.datetime.strptime(md["date"], "%Y-%m-%d").date()
                            except:
                                pass
                        st.session_state.metadata.update(md)
                    if "weights" in data:
                        st.session_state.weights.update(data["weights"])
                    if "black_white_points" in data:
                        st.session_state.black_white_points.update(data["black_white_points"])
                    if "roi" in data:
                        st.session_state.roi.update(data["roi"])
                    st.success("インポート成功。タブ①に戻って確認してください。")
                    st.json(data)
                except Exception as e:
                    st.error(f"JSONエラー: {e}")

        st.divider()
        st.markdown("### 📋 レポートプレビュー")
        st.dataframe(st.session_state.results_df, use_container_width=True)
        st.dataframe(st.session_state.results["summary"], use_container_width=True)

        # 実験情報表示
        st.json({
            "metadata": {k: str(v) if isinstance(v, (datetime.date, datetime.datetime)) else v for k,v in st.session_state.metadata.items()},
            "wash_rates": st.session_state.wash_rates
        })

# Footer
st.divider()
st.caption("Lab Wash V6 | Streamlit + OpenCV | 堅牢化: ゼロ除算回避, try-except, ROI安全クリップ, マスクフォールバック | © 2026")
