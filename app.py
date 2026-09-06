
# Lab Wash V7 - ROIドラッグ選択 & テクスチャ・色判定 & タップ式黒白校正
# 修正: 洗剤濃度→実験条件自由記述, ROIドラッグ, ガーゼ自動検出(テクスチャ+色), 黒白は画像タップで真っ黒・真っ白補正

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import json
import io
import datetime
import base64
from typing import Dict, Tuple, Optional

try:
    from skimage.color import rgb2lab, rgb2xyz
except ImportError:
    rgb2lab = None
    rgb2xyz = None

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.units import mm
except ImportError:
    SimpleDocTemplate = None

try:
    from streamlit_paste_button import paste_button
    HAS_PASTE = True
except ImportError:
    HAS_PASTE = False

try:
    from streamlit_drawable_canvas import st_canvas
    HAS_CANVAS = True
except ImportError:
    HAS_CANVAS = False

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    HAS_IMG_COORDS = True
except ImportError:
    HAS_IMG_COORDS = False

st.set_page_config(page_title="Lab Wash V7", page_icon="🧪", layout="wide")

def safe_divide(a,b,default=0.0):
    try:
        if b==0 or b is None or abs(b)<1e-9: return default
        return a/b
    except: return default

def pil_to_cv(pil_img): return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
def cv_to_pil(cv_img): return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

def sample_color_at(pil_img, x, y, size=7):
    """x,yは元画像座標、周辺平均でノイズ低減"""
    try:
        w,h = pil_img.size
        x = int(np.clip(x,0,w-1))
        y = int(np.clip(y,0,h-1))
        half = size//2
        x1 = max(0, x-half); y1 = max(0, y-half)
        x2 = min(w, x+half+1); y2 = min(h, y+half+1)
        crop = pil_img.crop((x1,y1,x2,y2))
        arr = np.array(crop)
        if arr.size==0: return (0,0,0)
        mean = arr.mean(axis=(0,1))
        return tuple(int(v) for v in mean[:3])
    except:
        return (0,0,0)

def apply_bw_calibration(pil_img, black_rgb, white_rgb):
    """ユーザーがタップした黒点を0,0,0 白点を255,255,255に正規化して条件を揃える"""
    try:
        img=np.array(pil_img).astype(np.float32)
        black=np.array(black_rgb,dtype=np.float32).reshape(1,1,3)
        white=np.array(white_rgb,dtype=np.float32).reshape(1,1,3)
        denom=white-black
        denom=np.where(np.abs(denom)<1e-6,1.0,denom)
        corrected=(img-black)/denom*255.0
        return Image.fromarray(np.clip(corrected,0,255).astype(np.uint8))
    except Exception as e:
        st.warning(f"色補正失敗:{e}")
        return pil_img

def extract_bw_auto(pil_img, percentile=2):
    try:
        arr=np.array(pil_img).reshape(-1,3)
        low=np.percentile(arr,percentile,axis=0)
        high=np.percentile(arr,100-percentile,axis=0)
        return tuple(np.clip(low,0,255).astype(int)), tuple(np.clip(high,0,255).astype(int))
    except: return (0,0,0),(255,255,255)

def detect_gauze_advanced(roi_pil, color_sens=0.5, texture_sens=0.5, morph_k=7, use_kmeans=False):
    """
    ROI内からガーゼ領域をテクスチャ+色で自動検出
    - 色: HSV S低 + V高 + Lab L高 = 白いガーゼ
    - テクスチャ: 局所標準偏差で織り目を検出
    - いびつな外形はapproxPolyDPで保持
    """
    try:
        cv_img=pil_to_cv(roi_pil)
        h,w = cv_img.shape[:2]
        # リサイズして高速化(最大800px)
        scale=1.0
        if max(h,w)>800:
            scale=800/max(h,w)
            cv_small=cv2.resize(cv_img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        else:
            cv_small=cv_img

        # --- 色特徴 ---
        hsv=cv2.cvtColor(cv_small, cv2.COLOR_BGR2HSV)
        H,S,V=cv2.split(hsv)
        # 感度で閾値可変
        # color_sens 0..1: 0=厳しめ(白のみ), 1=緩め(薄汚れもガーゼ扱い)
        s_thresh = int(60 + color_sens*60)  # 60-120: Sがこれ以下がガーゼ
        v_thresh = int(200 - color_sens*40) # 160-200: Vがこれ以上がガーゼ
        _, s_mask = cv2.threshold(S, s_thresh, 255, cv2.THRESH_BINARY_INV)
        _, v_mask = cv2.threshold(V, v_thresh, 255, cv2.THRESH_BINARY)
        color_mask = cv2.bitwise_and(s_mask, v_mask)

        # Lab L
        lab=cv2.cvtColor(cv_small, cv2.COLOR_BGR2LAB)
        L,A,B=cv2.split(lab)
        l_thresh = int(170 - color_sens*30) # 140-170
        _, l_mask = cv2.threshold(L, l_thresh, 255, cv2.THRESH_BINARY)
        color_mask = cv2.bitwise_and(color_mask, l_mask)

        # --- テクスチャ特徴: 局所標準偏差 ---
        gray=cv2.cvtColor(cv_small, cv2.COLOR_BGR2GRAY)
        ksize=15
        # 高速な局所分散計算: blurで平均
        mean = cv2.GaussianBlur(gray, (ksize,ksize), 0)
        sqr_mean = cv2.GaussianBlur((gray.astype(np.float32)**2), (ksize,ksize), 0)
        std = np.sqrt(np.maximum(sqr_mean - mean.astype(np.float32)**2, 0))
        # 正規化
        if std.max()>1e-6:
            std_norm = (std / std.max() * 255).astype(np.uint8)
        else:
            std_norm = np.zeros_like(gray, dtype=np.uint8)
        # テクスチャ閾値
        tex_low = int(8 + texture_sens*12)   # 8-20
        tex_high = int(70 + texture_sens*30) # 70-100
        _, tex_mask_low = cv2.threshold(std_norm, tex_low, 255, cv2.THRESH_BINARY)
        _, tex_mask_high = cv2.threshold(std_norm, tex_high, 255, cv2.THRESH_BINARY_INV)
        tex_mask = cv2.bitwise_and(tex_mask_low, tex_mask_high) # 中程度のテクスチャがガーゼ

        # --- K-meansオプション (色で2クラス分離) ---
        if use_kmeans:
            try:
                # Labのabでクラスタリング
                data = lab.reshape(-1,3).astype(np.float32)
                # L,a,bのうちa,bを重視
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                _, labels, centers = cv2.kmeans(data, 2, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
                labels = labels.reshape(h_small:=cv_small.shape[0], -1)
                # Lが高い方のクラスタをガーゼとみなす
                # centersのLで判定
                gauze_cluster = np.argmax(centers[:,0])
                kmeans_mask = (labels==gauze_cluster).astype(np.uint8)*255
                # color_maskと統合
                color_mask = cv2.bitwise_or(color_mask, kmeans_mask)
            except Exception as e:
                pass

        # --- 統合 ---
        # ガーゼは色条件を満たしつつ、テクスチャも持つ -> ANDだが、テクスチャが弱い白地も拾うためORも混ぜる
        combined = cv2.bitwise_and(color_mask, tex_mask)
        # 色だけで十分白い領域も足す (テクスチャ弱い場合の救済)
        combined = cv2.bitwise_or(combined, cv2.bitwise_and(color_mask, l_mask))

        # --- モルフォロジー: 織り目の隙間を閉じつつ、いびつな外形は保持 ---
        mk = max(1, morph_k)
        if mk%2==0: mk+=1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk,mk))
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        # 小穴埋め
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_small, iterations=1)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_small, iterations=1)

        # --- 輪郭抽出: いびつでもOKなようにapproxのepsilonを小さめに ---
        contours,_ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            # フォールバック: color_maskをそのまま使う
            final_mask = color_mask
        else:
            # 面積フィルタ: ROIの0.5%以上
            total_area = opened.size
            filtered = [c for c in contours if cv2.contourArea(c) > total_area*0.005]
            if not filtered: filtered = contours
            # 面積順
            filtered = sorted(filtered, key=cv2.contourArea, reverse=True)
            final_mask = np.zeros_like(gray)
            # 上位2つまで統合 (ガーゼが分断している場合)
            for c in filtered[:2]:
                # いびつさを保持: epsilonを小さく (0.5%)
                eps = 0.005 * cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, eps, True)
                cv2.drawContours(final_mask, [approx], -1, 255, -1)
            # 最終クローズで縁を滑らかにしすぎない程度に
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)

        # 元サイズに戻す
        if scale!=1.0:
            final_mask = cv2.resize(final_mask, (w,h), interpolation=cv2.INTER_NEAREST)
            color_mask_full = cv2.resize(color_mask, (w,h), interpolation=cv2.INTER_NEAREST)
            tex_mask_full = cv2.resize(tex_mask, (w,h), interpolation=cv2.INTER_NEAREST)
            combined_full = cv2.resize(combined, (w,h), interpolation=cv2.INTER_NEAREST)
            std_norm_full = cv2.resize(std_norm, (w,h), interpolation=cv2.INTER_NEAREST)
        else:
            color_mask_full = color_mask
            tex_mask_full = tex_mask
            combined_full = combined
            std_norm_full = std_norm

        # オーバーレイ作成
        overlay = cv_img.copy() if scale==1.0 else pil_to_cv(roi_pil)
        # 元ROIサイズのoverlay用
        if scale!=1.0:
            cv_full = pil_to_cv(roi_pil)
            overlay = cv_full

        colored = cv2.applyColorMap(final_mask, cv2.COLORMAP_JET)
        overlay_blend = cv2.addWeighted(overlay, 0.7, colored, 0.3, 0)
        contours_final,_ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_blend, contours_final, -1, (0,255,0), 2)

        return {
            "mask": final_mask,
            "overlay": cv_to_pil(overlay_blend),
            "color_mask": Image.fromarray(color_mask_full),
            "texture_mask": Image.fromarray(tex_mask_full),
            "std_map": Image.fromarray(std_norm_full),
            "combined": Image.fromarray(combined_full)
        }
    except Exception as e:
        st.warning(f"高度検出エラー:{e}")
        import traceback
        st.code(traceback.format_exc())
        h,w = roi_pil.size[1], roi_pil.size[0]
        dummy = np.ones((h,w),dtype=np.uint8)*255
        return {"mask":dummy, "overlay":roi_pil, "color_mask":roi_pil, "texture_mask":roi_pil, "std_map":roi_pil, "combined":roi_pil}

def get_mean_rgb_lab_xyz(pil_img,mask=None):
    try:
        img_np=np.array(pil_img)
        if mask is not None:
            if mask.shape[:2]!=img_np.shape[:2]:
                mask=cv2.resize(mask,(img_np.shape[1],img_np.shape[0]),interpolation=cv2.INTER_NEAREST)
            valid=mask>127
            if np.sum(valid)<10: valid=np.ones((img_np.shape[0],img_np.shape[1]),dtype=bool)
        else: valid=np.ones((img_np.shape[0],img_np.shape[1]),dtype=bool)
        mean_rgb=np.mean(img_np[valid],axis=0)
        rgb_norm=img_np.astype(np.float32)/255.0
        lab=rgb2lab(rgb_norm); xyz=rgb2xyz(rgb_norm)
        mean_lab=np.mean(lab[valid],axis=0); mean_xyz=np.mean(xyz[valid],axis=0)
        return {"R":float(mean_rgb[0]),"G":float(mean_rgb[1]),"B":float(mean_rgb[2]),"L":float(mean_lab[0]),"a":float(mean_lab[1]),"b":float(mean_lab[2]),"X":float(mean_xyz[0]*100),"Y":float(mean_xyz[1]*100),"Z":float(mean_xyz[2]*100)}
    except Exception as e:
        st.warning(f"色彩計算エラー:{e}")
        return {"R":0,"G":0,"B":0,"L":0,"a":0,"b":0,"X":0,"Y":0,"Z":0}

def calc_deltaE76(lab1,lab2): return float(np.sqrt((lab1["L"]-lab2["L"])**2+(lab1["a"]-lab2["a"])**2+(lab1["b"]-lab2["b"])**2))
def calc_deltaE2000(lab1,lab2):
    try:
        L1,a1,b1=lab1["L"],lab1["a"],lab1["b"]; L2,a2,b2=lab2["L"],lab2["a"],lab2["b"]
        C1=np.sqrt(a1**2+b1**2); C2=np.sqrt(a2**2+b2**2); C_bar=(C1+C2)/2
        G=0.5*(1-np.sqrt((C_bar**7)/(C_bar**7+25**7+1e-12)))
        a1p=(1+G)*a1; a2p=(1+G)*a2
        C1p=np.sqrt(a1p**2+b1**2); C2p=np.sqrt(a2p**2+b2**2)
        def hp(ap,b):
            h=np.degrees(np.arctan2(b,ap))
            return h+360 if h<0 else h
        h1p=hp(a1p,b1); h2p=hp(a2p,b2)
        dLp=L2-L1; dCp=C2p-C1p
        dhp=0
        if C1p*C2p>=1e-12:
            dh=h2p-h1p
            if abs(dh)<=180: dhp=dh
            elif dh>180: dhp=dh-360
            else: dhp=dh+360
        dHp=2*np.sqrt(C1p*C2p)*np.sin(np.radians(dhp/2))
        Lp_bar=(L1+L2)/2; Cp_bar=(C1p+C2p)/2
        hp_bar=(h1p+h2p)/2 if abs(h1p-h2p)<=180 else (h1p+h2p+360)/2 if h1p+h2p<360 else (h1p+h2p-360)/2
        T=1-0.17*np.cos(np.radians(hp_bar-30))+0.24*np.cos(np.radians(2*hp_bar))+0.32*np.cos(np.radians(3*hp_bar+6))-0.20*np.cos(np.radians(4*hp_bar-63))
        dtheta=30*np.exp(-((hp_bar-275)/25)**2)
        RC=2*np.sqrt((Cp_bar**7)/(Cp_bar**7+25**7+1e-12))
        SL=1+(0.015*(Lp_bar-50)**2)/np.sqrt(20+(Lp_bar-50)**2+1e-12)
        SC=1+0.045*Cp_bar; SH=1+0.015*Cp_bar*T
        RT=-np.sin(np.radians(2*dtheta))*RC
        return float(np.sqrt((dLp/SL)**2+(dCp/SC)**2+(dHp/SH)**2+RT*(dCp/SC)*(dHp/SH)))
    except: return 0.0

def calc_KS(R): R=np.clip(R,0.001,0.999); return float(((1-R)**2)/(2*R))
def calc_WI_ASTM(XYZ): return float(3.388*XYZ["Z"]-3.0*XYZ["Y"])
def calc_WI_CIE(XYZ):
    s=XYZ["X"]+XYZ["Y"]+XYZ["Z"]+1e-12
    x=XYZ["X"]/s; y=XYZ["Y"]/s
    return float(XYZ["Y"]+800*(0.3127-x)+1700*(0.3290-y))

def generate_pdf_buffer(metadata,weights,results_df,lab_data,wash_rates,images_dict,calibrated_dict):
    if SimpleDocTemplate is None: return None
    buffer=io.BytesIO()
    doc=SimpleDocTemplate(buffer,pagesize=A4,rightMargin=20*mm,leftMargin=20*mm,topMargin=15*mm,bottomMargin=15*mm)
    styles=getSampleStyleSheet()
    story=[]
    story.append(Paragraph("<b>Lab Wash V7 レポート - ROIドラッグ & タップ校正</b>",styles['Title']))
    story.append(Spacer(1,10*mm))
    meta_rows=[["実験ID",metadata.get("exp_id",""),"実施日",str(metadata.get("date",""))],["試料名",metadata.get("sample_name",""),"担当者",metadata.get("operator","")],["温度",f"{metadata.get('temp','')} ℃","時間",f"{metadata.get('time','')} min"],["実験条件",metadata.get("exp_condition",""),"",""]]
    t=Table(meta_rows,colWidths=[25*mm,50*mm,25*mm,50*mm])
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(0,-1),colors.HexColor("#E8F0FE")),('BACKGROUND',(2,0),(2,-1),colors.HexColor("#E8F0FE")),('GRID',(0,0),(-1,-1),0.5,colors.grey),('FONTSIZE',(0,0),(-1,-1),9)]))
    story.append(Paragraph("<b>1. 実験条件</b>",styles['Heading2'])); story.append(t); story.append(Spacer(1,8*mm))
    weight_rows=[["項目","素地","洗浄前","洗浄後"],["重量(g)",f"{weights.get('素地',0):.4f}",f"{weights.get('洗浄前',0):.4f}",f"{weights.get('洗浄後',0):.4f}"],["汚染量(g)",f"{wash_rates.get('contamination',0):.4f}","",""],["除去量(g)",f"{wash_rates.get('removed',0):.4f}","",""],["重量洗浄率(%)",f"{wash_rates.get('weight_rate',0):.2f}","",""]]
    wt=Table(weight_rows,colWidths=[35*mm,35*mm,35*mm,35*mm])
    wt.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor("#D0E0FF")),('GRID',(0,0),(-1,-1),0.5,colors.grey),('FONTSIZE',(0,0),(-1,-1),8)]))
    story.append(Paragraph("<b>2. 重量評価</b>",styles['Heading2'])); story.append(wt); story.append(Spacer(1,8*mm))
    if results_df is not None and not results_df.empty:
        header=["指標"]+list(results_df.columns)
        table_data=[header]
        for idx,row in results_df.iterrows():
            table_data.append([str(idx)]+[f"{v:.3f}" if isinstance(v,float) else str(v) for v in row.values])
        ct=Table(table_data,colWidths=[35*mm]+[25*mm]*(len(header)-1))
        ct.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor("#FFE0B2")),('GRID',(0,0),(-1,-1),0.5,colors.grey),('FONTSIZE',(0,0),(-1,-1),7)]))
        story.append(Paragraph("<b>3. 色彩・光学評価</b>",styles['Heading2'])); story.append(ct)
    doc.build(story)
    buffer.seek(0)
    return buffer

def init_state():
    defaults={
        "metadata":{"exp_id":"EXP-001","sample_name":"ガーゼ試料A","operator":"","temp":40.0,"time":10.0,"exp_condition":"中性洗剤0.1%, 浴比1:50, 振とう100rpm, pH7","date":datetime.date.today()},
        "images":{"素地":None,"洗浄前":None,"洗浄後":None},
        "weights":{"素地":1.0,"洗浄前":1.5,"洗浄後":1.1},
        "black_white_points":{"素地":{"black":(0,0,0),"white":(255,255,255)},"洗浄前":{"black":(0,0,0),"white":(255,255,255)},"洗浄後":{"black":(0,0,0),"white":(255,255,255)}},
        "roi_pct":{"common":{"x":0.05,"y":0.05,"w":0.9,"h":0.9,"use_common":True},"素地":{"x":0.05,"y":0.05,"w":0.9,"h":0.9},"洗浄前":{"x":0.05,"y":0.05,"w":0.9,"h":0.9},"洗浄後":{"x":0.05,"y":0.05,"w":0.9,"h":0.9}},
        "calibrated_images":{"素地":None,"洗浄前":None,"洗浄後":None},
        "masks":{"素地":None,"洗浄前":None,"洗浄後":None},
        "results":None,"results_df":None,"lab_data":{},"wash_rates":{},
    }
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k]=v

init_state()

with st.sidebar:
    st.title("🧪 Lab Wash V7")
    st.caption("ドラッグROI + テクスチャ検出 + タップ校正")
    st.markdown("**新機能**")
    st.markdown("- ROI: ドラッグで選択\n- ガーゼ: 色+テクスチャで自動検出\n- 黒白: 画像タップで真っ黒・真っ白に補正")
    if not HAS_CANVAS:
        st.error("canvas未導入: streamlit-drawable-canvas")
    if not HAS_IMG_COORDS:
        st.error("タップ未導入: streamlit-image-coordinates")
    if not HAS_PASTE:
        st.warning("貼り付け: streamlit-paste-button推奨")
    if st.button("🔄 全リセット"):
        for k in list(st.session_state.keys()): del st.session_state[k]
        init_state(); st.rerun()

tab1,tab2,tab3,tab4,tab5=st.tabs(["① 画像 & メタ","② タップ黒白校正","③ ドラッグROI & 自動検出","④ 解析","⑤ レポート"])

with tab1:
    st.subheader("ステップ1: メタデータ & 画像アップロード")
    c1,c2=st.columns(2)
    with c1:
        st.session_state.metadata["exp_id"]=st.text_input("実験ID",value=st.session_state.metadata.get("exp_id",""))
        st.session_state.metadata["sample_name"]=st.text_input("試料名",value=st.session_state.metadata.get("sample_name",""))
        st.session_state.metadata["operator"]=st.text_input("担当者名",value=st.session_state.metadata.get("operator",""))
    with c2:
        st.session_state.metadata["temp"]=st.number_input("洗浄温度 (℃)",value=float(st.session_state.metadata.get("temp",40)),step=1.0)
        st.session_state.metadata["time"]=st.number_input("洗浄時間 (min)",value=float(st.session_state.metadata.get("time",10)),step=1.0)
        st.session_state.metadata["date"]=st.date_input("実施日",value=st.session_state.metadata.get("date",datetime.date.today()))
    st.session_state.metadata["exp_condition"]=st.text_area("実験条件 (温度・時間以外自由記述)",value=st.session_state.metadata.get("exp_condition",""),height=90,placeholder="例: 中性洗剤0.1%, 浴比1:50, pH7, 硬水, 前処理など")

    st.divider()
    st.markdown("### 画像アップロード - 貼り付け対応")

    st.components.v1.html("""
    <style>#paste-hint{border:2px dashed #4CAF50;border-radius:12px;padding:15px;text-align:center;background:#F1F8E9;}</style>
    <div id="paste-hint" tabindex="0">📋 ここクリック → <b>Ctrl+V / Cmd+V</b> で貼り付け / ドラッグ&ドロップOK<div id="prev"></div></div>
    <script>
    const h=document.getElementById('paste-hint'); h.focus();
    document.addEventListener('paste',e=>{
      for(let it of e.clipboardData.items){ if(it.type.indexOf('image')!==-1){
        const b=it.getAsFile(); const u=URL.createObjectURL(b);
        document.getElementById('prev').innerHTML='<img src="'+u+'" style="max-width:100%;max-height:200px;border-radius:8px;margin-top:8px"><br><small style="color:green">✓ 検出! 下のボタンで取り込んでください</small>';
      }}
    });
    </script>
    """, height=150)

    cols=st.columns(3)
    for idx,key in enumerate(["素地","洗浄前","洗浄後"]):
        with cols[idx]:
            st.markdown(f"#### {key}")
            up=st.file_uploader(f"{key}ファイル",type=["png","jpg","jpeg","bmp","tiff","webp"],key=f"upload_{key}",label_visibility="collapsed")
            if up:
                try:
                    pil_img=Image.open(up).convert("RGB")
                    st.session_state.images[key]=pil_img
                    st.session_state.calibrated_images[key]=None
                    st.session_state.masks[key]=None
                except Exception as e: st.error(f"読込失敗:{e}")
            if HAS_PASTE:
                try:
                    pasted=paste_button(f"📋 貼り付け {key}", key=f"paste_{key}")
                    pil_from=None
                    if pasted is not None:
                        if isinstance(pasted, Image.Image): pil_from=pasted.convert("RGB")
                        elif hasattr(pasted,'image_data') and pasted.image_data is not None:
                            d=pasted.image_data
                            if isinstance(d, Image.Image): pil_from=d.convert("RGB")
                            elif isinstance(d, np.ndarray): pil_from=Image.fromarray(d).convert("RGB")
                            elif isinstance(d,str) and d.startswith("data:image"):
                                pil_from=Image.open(io.BytesIO(base64.b64decode(d.split(",",1)[1]))).convert("RGB")
                    if pil_from:
                        st.session_state.images[key]=pil_from
                        st.session_state.calibrated_images[key]=None
                        st.session_state.masks[key]=None
                        st.success(f"{key} 貼り付け成功"); st.rerun()
                except Exception as e: st.warning(f"貼り付けエラー:{e}")
            if st.session_state.images[key] is not None:
                st.image(st.session_state.images[key], caption=f"{key} {st.session_state.images[key].size[0]}x{st.session_state.images[key].size[1]}", use_container_width=True)
            st.session_state.weights[key]=st.number_input(f"{key} 重量(g)",value=float(st.session_state.weights.get(key,0)),step=0.001,format="%.4f",key=f"weight_{key}")

with tab2:
    st.subheader("ステップ2: タップ式 黒白キャリブレーション")
    st.caption("画像の中の黒い部分・白い部分をタップして、その点を真っ黒(0,0,0)・真っ白(255,255,255)に補正して条件を揃えます。")
    if not HAS_IMG_COORDS:
        st.error("`pip install streamlit-image-coordinates` が必要です。")
        st.code("pip install streamlit-image-coordinates", language="bash")
    else:
        if all(v is None for v in st.session_state.images.values()):
            st.warning("①で画像を登録してください")
        else:
            for key in ["素地","洗浄前","洗浄後"]:
                pil_img = st.session_state.images.get(key)
                if pil_img is None: continue
                st.markdown(f"### {key}")
                # 現在の黒白点表示
                cur_black = st.session_state.black_white_points[key]["black"]
                cur_white = st.session_state.black_white_points[key]["white"]
                c1,c2,c3 = st.columns([1,1,2])
                with c1:
                    st.markdown("**黒点**")
                    st.color_picker(f"現在の黒 {key}", value="#%02x%02x%02x"%cur_black, key=f"show_black_{key}", disabled=True)
                    st.write(f"RGB: {cur_black}")
                    if st.button(f"黒点をリセット {key}", key=f"reset_b_{key}"):
                        st.session_state.black_white_points[key]["black"]=(0,0,0)
                        st.rerun()
                    st.markdown("**白点**")
                    st.color_picker(f"現在の白 {key}", value="#%02x%02x%02x"%cur_white, key=f"show_white_{key}", disabled=True)
                    st.write(f"RGB: {cur_white}")
                    if st.button(f"白点をリセット {key}", key=f"reset_w_{key}"):
                        st.session_state.black_white_points[key]["white"]=(255,255,255)
                        st.rerun()
                    st.divider()
                    tap_mode = st.radio(f"タップモード {key}", ["黒点をタップして取得", "白点をタップして取得"], key=f"tapmode_{key}", horizontal=False)
                    st.caption("画像をクリックすると、その位置の色を平均7x7で取得します。")
                    if st.button(f"自動抽出 (フォールバック) {key}", key=f"auto_{key}"):
                        b,w = extract_bw_auto(pil_img)
                        st.session_state.black_white_points[key]["black"]=b
                        st.session_state.black_white_points[key]["white"]=w
                        st.rerun()

                with c2:
                    # タップ用表示画像: 600px幅にリサイズして表示 (座標変換用に比率保持)
                    display_max = 600
                    disp_img = pil_img.copy()
                    # アスペクト保持でリサイズ
                    if max(disp_img.size) > display_max:
                        ratio = display_max / max(disp_img.size)
                        new_size = (int(disp_img.size[0]*ratio), int(disp_img.size[1]*ratio))
                        disp_img = disp_img.resize(new_size, Image.LANCZOS)
                    st.markdown(f"**{key} 画像をクリック** - {tap_mode}")
                    # 画像座標取得
                    coords = streamlit_image_coordinates(disp_img, key=f"coord_{key}")
                    if coords is not None:
                        # coordsはdisplay画像上の座標
                        # 元画像座標に変換
                        scale_x = pil_img.size[0] / disp_img.size[0]
                        scale_y = pil_img.size[1] / disp_img.size[1]
                        orig_x = int(coords["x"] * scale_x)
                        orig_y = int(coords["y"] * scale_y)
                        sampled = sample_color_at(pil_img, orig_x, orig_y, size=9)
                        if "黒点" in tap_mode:
                            st.session_state.black_white_points[key]["black"] = sampled
                            st.toast(f"{key} 黒点を取得: {sampled} at ({orig_x},{orig_y})")
                        else:
                            st.session_state.black_white_points[key]["white"] = sampled
                            st.toast(f"{key} 白点を取得: {sampled} at ({orig_x},{orig_y})")
                        # 少し待ってリラン
                        st.rerun()

                with c3:
                    # 補正後プレビュー
                    black = st.session_state.black_white_points[key]["black"]
                    white = st.session_state.black_white_points[key]["white"]
                    try:
                        calibrated = apply_bw_calibration(pil_img, black, white)
                        st.session_state.calibrated_images[key] = calibrated
                        ca,cb = st.columns(2)
                        with ca:
                            st.image(pil_img, caption=f"元画像\n黒{black} 白{white}", use_container_width=True)
                        with cb:
                            st.image(calibrated, caption="補正後 (黒→0, 白→255)", use_container_width=True)
                        # ヒストグラム的説明
                        st.caption(f"補正式: (img - {black}) / ({white} - {black}) * 255 → 黒を真っ黒、白を真っ白に正規化して全画像の条件を揃えます")
                    except Exception as e:
                        st.error(f"補正エラー:{e}")
                st.divider()

with tab3:
    st.subheader("ステップ3: ドラッグROI選択 & ガーゼ自動検出 (色+テクスチャ)")
    st.caption("まず大まかにガーゼがある領域をドラッグで選択 → その中から色(白さ・低彩度)とテクスチャ(織り目)でいびつでも自動抽出")
    if not HAS_CANVAS:
        st.error("`pip install streamlit-drawable-canvas` が必要です")
        st.code("pip install streamlit-drawable-canvas", language="bash")
    else:
        has_img = any(v is not None for v in st.session_state.images.values())
        if not has_img:
            st.warning("①で画像を登録してください")
        else:
            # 共通ROIか個別か
            use_common = st.checkbox("共通ROIを使用 (全画像で同じ相対位置)", value=st.session_state.roi_pct["common"].get("use_common", True), key="use_common_roi")
            st.session_state.roi_pct["common"]["use_common"] = use_common

            # 検出パラメータ
            with st.expander("🔧 自動検出パラメータ (色・テクスチャ感度)", expanded=True):
                pc1,pc2,pc3 = st.columns(3)
                with pc1:
                    color_sens = st.slider("色感度 (白さ判定の緩さ)", 0.0, 1.0, 0.5, 0.05, key="color_sens", help="0=厳しめ(真っ白のみガーゼ), 1=緩め(薄汚れもガーゼとして許容)")
                    texture_sens = st.slider("テクスチャ感度", 0.0, 1.0, 0.5, 0.05, key="tex_sens", help="織り目の検出しやすさ")
                with pc2:
                    morph_k = st.slider("モルフォロジー強度", 3, 21, 9, step=2, key="morph_k", help="いびつな隙間を埋める強さ")
                    use_kmeans = st.checkbox("K-means色分離を併用 (背景が複雑な場合)", value=False, key="use_kmeans")
                with pc3:
                    st.info("**検出ロジック**\n- 色: HSV S低(低彩度) + V高(明るい) + Lab L高\n- テクスチャ: 局所標準偏差で織り目を検出\n- いびつでも輪郭をapproxPolyDPで保持")

            def get_roi_from_canvas(canvas_result, canvas_w, canvas_h):
                """canvasのrectからROI%を取得"""
                if canvas_result.json_data is None: return None
                objects = canvas_result.json_data.get("objects", [])
                if not objects: return None
                # 最後のrectを取得
                rects = [o for o in objects if o.get("type")=="rect"]
                if not rects: return None
                r = rects[-1]
                # left, top, width, height
                left = r.get("left",0); top = r.get("top",0); w = r.get("width",0); h = r.get("height",0)
                # canvasサイズに対する割合に変換 (0-1)
                x_pct = left / canvas_w
                y_pct = top / canvas_h
                w_pct = w / canvas_w
                h_pct = h / canvas_h
                # クリップ
                x_pct = max(0,min(x_pct,0.95)); y_pct = max(0,min(y_pct,0.95))
                w_pct = max(0.05, min(w_pct, 1-x_pct)); h_pct = max(0.05, min(h_pct, 1-y_pct))
                return {"x":x_pct,"y":y_pct,"w":w_pct,"h":h_pct}

            def pct_to_pixels(pct, img_w, img_h):
                x = int(pct["x"]*img_w); y = int(pct["y"]*img_h)
                w = int(pct["w"]*img_w); h = int(pct["h"]*img_h)
                x=max(0,min(x,img_w-10)); y=max(0,min(y,img_h-10))
                w=max(10,min(w,img_w-x)); h=max(10,min(h,img_h-y))
                return x,y,w,h

            # 共通ROIモード
            if use_common:
                st.markdown("#### 共通ROIをドラッグで選択")
                # 参照画像選択
                ref_key = st.selectbox("参照画像 (ROI描画のベース)", [k for k,v in st.session_state.images.items() if v is not None], key="ref_img_common")
                ref_img_orig = st.session_state.calibrated_images.get(ref_key) or st.session_state.images.get(ref_key)
                if ref_img_orig:
                    # canvas表示用にリサイズ (600px)
                    canvas_max = 700
                    ratio = canvas_max / max(ref_img_orig.size)
                    if ratio<1:
                        disp_w = int(ref_img_orig.size[0]*ratio)
                        disp_h = int(ref_img_orig.size[1]*ratio)
                        disp_img = ref_img_orig.resize((disp_w,disp_h), Image.LANCZOS)
                    else:
                        disp_w, disp_h = ref_img_orig.size
                        disp_img = ref_img_orig

                    st.caption(f"参照: {ref_key} - 緑の四角をドラッグで描画 → 自動で全画像に適用。描画後は少し待つと下に反映されます。")
                    canvas_result = st_canvas(
                        fill_color="rgba(0,255,0,0.15)",
                        stroke_width=2,
                        stroke_color="#00FF00",
                        background_image=disp_img,
                        background_color="#EEE",
                        width=disp_w,
                        height=disp_h,
                        drawing_mode="rect",
                        key="canvas_common",
                        display_toolbar=True,
                    )
                    pct = get_roi_from_canvas(canvas_result, disp_w, disp_h)
                    if pct:
                        st.session_state.roi_pct["common"].update(pct)
                        st.success(f"ROI取得: x={pct['x']:.2f} y={pct['y']:.2f} w={pct['w']:.2f} h={pct['h']:.2f} (相対)")

                    # 現在のROI%表示
                    cur_pct = st.session_state.roi_pct["common"]
                    st.write(f"現在の共通ROI (相対): {cur_pct}")

                    # 各画像で検出実行
                    for key in ["素地","洗浄前","洗浄後"]:
                        src = st.session_state.calibrated_images.get(key) or st.session_state.images.get(key)
                        if src is None: continue
                        st.markdown(f"##### {key} - ROI内自動検出")
                        x,y,w,h = pct_to_pixels(cur_pct, src.size[0], src.size[1])
                        roi_pil = src.crop((x,y,x+w,y+h))

                        # 高度検出
                        result = detect_gauze_advanced(roi_pil, color_sens=color_sens, texture_sens=texture_sens, morph_k=morph_k, use_kmeans=use_kmeans)

                        # フルサイズマスク
                        full_mask = np.zeros((src.size[1], src.size[0]), dtype=np.uint8)
                        # result maskはroiサイズなので、元の位置に配置
                        # result["mask"]はroiサイズ
                        full_mask[y:y+h, x:x+w] = result["mask"]
                        st.session_state.masks[key] = full_mask

                        c1,c2,c3,c4,c5 = st.columns(5)
                        with c1:
                            # ROI位置
                            cv_full = pil_to_cv(src)
                            cv2.rectangle(cv_full, (x,y), (x+w,y+h), (0,255,0), 2)
                            st.image(cv_to_pil(cv_full), caption="ROI位置", use_container_width=True)
                        with c2:
                            st.image(roi_pil, caption="ROIクロップ", use_container_width=True)
                        with c3:
                            st.image(result["color_mask"], caption="色マスク (白さ)", use_container_width=True)
                            st.image(result["texture_mask"], caption="テクスチャマスク (織り目)", use_container_width=True)
                        with c4:
                            st.image(result["std_map"], caption="局所分散 (テクスチャ強度)", use_container_width=True)
                            st.image(result["combined"], caption="統合マスク", use_container_width=True)
                        with c5:
                            st.image(result["overlay"], caption="最終オーバーレイ (いびつ対応)", use_container_width=True)

            else:
                # 個別ROIモード
                for key in ["素地","洗浄前","洗浄後"]:
                    src = st.session_state.calibrated_images.get(key) or st.session_state.images.get(key)
                    if src is None: continue
                    st.markdown(f"#### {key} - 個別ROIドラッグ")
                    # canvas用リサイズ
                    canvas_max = 600
                    ratio = canvas_max / max(src.size)
                    if ratio<1:
                        disp_w = int(src.size[0]*ratio); disp_h = int(src.size[1]*ratio)
                        disp_img = src.resize((disp_w,disp_h), Image.LANCZOS)
                    else:
                        disp_w, disp_h = src.size
                        disp_img = src

                    canvas_result = st_canvas(
                        fill_color="rgba(0,255,0,0.15)",
                        stroke_width=2,
                        stroke_color="#00FF00",
                        background_image=disp_img,
                        width=disp_w,
                        height=disp_h,
                        drawing_mode="rect",
                        key=f"canvas_{key}",
                        display_toolbar=True,
                    )
                    pct = get_roi_from_canvas(canvas_result, disp_w, disp_h)
                    if pct:
                        st.session_state.roi_pct[key].update(pct)
                        st.success(f"{key} ROI更新")

                    cur_pct = st.session_state.roi_pct.get(key, {"x":0.05,"y":0.05,"w":0.9,"h":0.9})
                    x,y,w,h = pct_to_pixels(cur_pct, src.size[0], src.size[1])
                    roi_pil = src.crop((x,y,x+w,y+h))
                    result = detect_gauze_advanced(roi_pil, color_sens=color_sens, texture_sens=texture_sens, morph_k=morph_k, use_kmeans=use_kmeans)
                    full_mask = np.zeros((src.size[1], src.size[0]), dtype=np.uint8)
                    full_mask[y:y+h, x:x+w] = result["mask"]
                    st.session_state.masks[key] = full_mask

                    c1,c2,c3 = st.columns(3)
                    with c1: st.image(roi_pil, caption="ROI", use_container_width=True)
                    with c2: st.image(result["combined"], caption="統合マスク", use_container_width=True)
                    with c3: st.image(result["overlay"], caption="最終", use_container_width=True)

with tab4:
    st.subheader("ステップ4: 解析・洗浄率評価")
    if rgb2lab is None:
        st.error("scikit-image未導入")
    else:
        if st.button("🔬 解析実行", type="primary"):
            try:
                ws=float(st.session_state.weights.get("素地",0)); wb=float(st.session_state.weights.get("洗浄前",0)); wa=float(st.session_state.weights.get("洗浄後",0))
                cont=wb-ws; rem=wb-wa; w_rate=safe_divide(rem,cont,0)*100
                wash_rates={"contamination":cont,"removed":rem,"weight_rate":w_rate,"w_soil_free":ws,"w_before":wb,"w_after":wa}
                lab_data={}
                for key in ["素地","洗浄前","洗浄後"]:
                    src=st.session_state.calibrated_images.get(key) or st.session_state.images.get(key)
                    if src is None: lab_data[key]={"L":0,"a":0,"b":0,"X":0,"Y":0,"Z":0}; continue
                    info=get_mean_rgb_lab_xyz(src,st.session_state.masks.get(key))
                    R=safe_divide(info["Y"],100,0.5); R=np.clip(R,0.001,0.999)
                    lab_data[key]={**info,"KS":calc_KS(R),"WI_ASTM":calc_WI_ASTM(info),"WI_CIE":calc_WI_CIE(info),"R_ref":R}
                Ls=lab_data["素地"]["L"]; Lb=lab_data["洗浄前"]["L"]; La=lab_data["洗浄後"]["L"]
                L_rate=safe_divide(La-Lb,Ls-Lb,0)*100
                for k in ["洗浄前","洗浄後"]:
                    lab_data[k]["dE76"]=calc_deltaE76(lab_data["素地"],lab_data[k]); lab_data[k]["dE00"]=calc_deltaE2000(lab_data["素地"],lab_data[k])
                lab_data["洗浄後"]["dE76_before"]=calc_deltaE76(lab_data["洗浄前"],lab_data["洗浄後"])
                rows=[]
                for k in ["素地","洗浄前","洗浄後"]:
                    d=lab_data[k]
                    rows.append({"試料":k,"重量":wash_rates.get(f"w_{'soil_free' if k=='素地' else 'before' if k=='洗浄前' else 'after'}",0),"L*":d.get("L",0),"a*":d.get("a",0),"b*":d.get("b",0),"K/S":d.get("KS",0),"WI":d.get("WI_CIE",0),"ΔE00":d.get("dE00",0)})
                df=pd.DataFrame(rows).set_index("試料")
                summary=pd.DataFrame([{"指標":"重量洗浄率%","値":w_rate},{"指標":"L*洗浄率%","値":L_rate},{"指標":"汚染量","値":cont},{"指標":"除去量","値":rem}]).set_index("指標")
                st.session_state.results_df=df; st.session_state.lab_data=lab_data; st.session_state.wash_rates={**wash_rates,"L_rate":L_rate}; st.session_state.results={"df_main":df,"summary":summary}
                st.success("解析完了")
            except Exception as e:
                st.error(f"エラー: {e}"); import traceback; st.code(traceback.format_exc())
        if st.session_state.get("results_df") is not None:
            st.dataframe(st.session_state.results_df.style.format("{:.3f}"),use_container_width=True)
            st.dataframe(st.session_state.results["summary"].style.format("{:.3f}"),use_container_width=True)
            try:
                import plotly.express as px
                df_plot=st.session_state.results_df.reset_index()
                st.plotly_chart(px.bar(df_plot,x="試料",y="L*",color="試料",title="L*比較"),use_container_width=True)
                st.plotly_chart(px.bar(df_plot,x="試料",y="K/S",color="試料",title="K/S比較"),use_container_width=True)
            except: st.bar_chart(st.session_state.results_df[["L*","K/S"]])

with tab5:
    st.subheader("ステップ5: レポート出力")
    if st.session_state.get("results_df") is None: st.warning("先に解析実行")
    else:
        if st.button("📄 PDF生成",type="primary"):
            pdf=generate_pdf_buffer(st.session_state.metadata,st.session_state.weights,st.session_state.results_df,st.session_state.lab_data,st.session_state.wash_rates,st.session_state.images,st.session_state.calibrated_images)
            if pdf: st.session_state["pdf_buffer"]=pdf; st.success("生成完了")
        if "pdf_buffer" in st.session_state and st.session_state["pdf_buffer"]:
            st.download_button("⬇️ PDFダウンロード",data=st.session_state["pdf_buffer"],file_name=f"{st.session_state.metadata.get('exp_id','report')}_V7.pdf",mime="application/pdf")
        csv=st.session_state.results_df.to_csv(encoding="utf-8-sig")
        st.download_button("⬇️ CSV",data=csv,file_name="results.csv",mime="text/csv")
        meta_copy=st.session_state.metadata.copy()
        if isinstance(meta_copy.get("date"),(datetime.date,datetime.datetime)): meta_copy["date"]=meta_copy["date"].isoformat()
        json_str=json.dumps({"metadata":meta_copy,"weights":st.session_state.weights,"lab_data":st.session_state.lab_data,"wash_rates":st.session_state.wash_rates,"roi_pct":st.session_state.roi_pct},ensure_ascii=False,indent=2,default=str)
        st.download_button("⬇️ JSON",data=json_str,file_name="LabWash_V7.json",mime="application/json")

st.divider()
st.caption("Lab Wash V7 | ドラッグROI + 色・テクスチャ自動検出 + タップ式黒白校正 | 真っ黒・真っ白正規化")
