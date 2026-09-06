
# Lab Wash V6.2 - 貼り付け完全対応版
# 修正点: 洗剤濃度→実験条件(自由記述), use_container_width修正, 貼り付け3重対応

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

st.set_page_config(page_title="Lab Wash V6.2", page_icon="🧪", layout="wide")

def safe_divide(a,b,default=0.0):
    try:
        if b==0 or b is None or abs(b)<1e-9: return default
        return a/b
    except: return default

def pil_to_cv(pil_img): return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
def cv_to_pil(cv_img): return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

def apply_bw_calibration(pil_img, black_rgb, white_rgb):
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

def detect_gauze_mask(roi_pil, blur_k=5, thresh=0, use_otsu=True, morph_k=7, invert=False, min_area_ratio=0.1):
    try:
        cv_img=pil_to_cv(roi_pil)
        gray=cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY)
        k=max(1,blur_k); 
        if k%2==0: k+=1
        blurred=cv2.GaussianBlur(gray,(k,k),0)
        if use_otsu: _,binary=cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        else: _,binary=cv2.threshold(blurred,thresh,255,cv2.THRESH_BINARY)
        if invert: binary=cv2.bitwise_not(binary)
        mk=max(1,morph_k)
        if mk%2==0: mk+=1
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(mk,mk))
        closed=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel,iterations=2)
        opened=cv2.morphologyEx(closed,cv2.MORPH_OPEN,kernel,iterations=1)
        contours,_=cv2.findContours(opened,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.ones_like(gray)*255, roi_pil, Image.fromarray(binary)
        total=gray.shape[0]*gray.shape[1]
        filtered=[c for c in contours if cv2.contourArea(c)>total*min_area_ratio*0.01]
        if not filtered: filtered=contours
        filtered=sorted(filtered,key=cv2.contourArea,reverse=True)
        mask=np.zeros_like(gray)
        for c in filtered[:3]:
            cv2.drawContours(mask,[c],-1,255,-1)
            cv2.drawContours(mask,[cv2.convexHull(c)],-1,255,-1)
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel,iterations=1)
        overlay=cv_img.copy()
        overlay=cv2.addWeighted(overlay,0.7,cv2.applyColorMap(mask,cv2.COLORMAP_JET),0.3,0)
        fc,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay,fc,-1,(0,255,0),2)
        return mask,cv_to_pil(overlay),Image.fromarray(opened)
    except Exception as e:
        st.warning(f"マスク検出エラー:{e}")
        h,w=roi_pil.size[1],roi_pil.size[0]
        return np.ones((h,w),dtype=np.uint8)*255,roi_pil,roi_pil

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
    story.append(Paragraph("<b>Lab Wash V6.2 レポート</b>",styles['Title']))
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
        "roi":{"common":{"x":5,"y":5,"w":90,"h":90,"use_common":True},"素地":{"x":5,"y":5,"w":90,"h":90},"洗浄前":{"x":5,"y":5,"w":90,"h":90},"洗浄後":{"x":5,"y":5,"w":90,"h":90}},
        "calibrated_images":{"素地":None,"洗浄前":None,"洗浄後":None},
        "masks":{"素地":None,"洗浄前":None,"洗浄後":None},
        "results":None,"results_df":None,"lab_data":{},"wash_rates":{},
    }
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k]=v

init_state()

with st.sidebar:
    st.title("🧪 Lab Wash V6.2")
    st.caption("温度・時間・実験条件(自由記述) + 貼り付け完全対応")
    if HAS_PASTE:
        st.success("📋 貼り付けボタン: 有効")
    else:
        st.error("貼り付けボタン無効: requirements.txtに streamlit-paste-button を追加")
        st.code("streamlit-paste-button>=0.1.2", language="text")
    if st.button("🔄 全リセット"):
        for k in list(st.session_state.keys()): del st.session_state[k]
        init_state(); st.rerun()

tab1,tab2,tab3,tab4,tab5=st.tabs(["① メタデータ & 画像","② 黒白校正","③ ROI & 自動検出","④ 解析","⑤ レポート"])

with tab1:
    st.subheader("ステップ1: メタデータ入力 & 画像")
    c1,c2=st.columns(2)
    with c1:
        st.session_state.metadata["exp_id"]=st.text_input("実験ID",value=st.session_state.metadata.get("exp_id",""))
        st.session_state.metadata["sample_name"]=st.text_input("試料名",value=st.session_state.metadata.get("sample_name",""))
        st.session_state.metadata["operator"]=st.text_input("担当者名",value=st.session_state.metadata.get("operator",""))
    with c2:
        st.session_state.metadata["temp"]=st.number_input("洗浄温度 (℃)",value=float(st.session_state.metadata.get("temp",40)),step=1.0)
        st.session_state.metadata["time"]=st.number_input("洗浄時間 (min)",value=float(st.session_state.metadata.get("time",10)),step=1.0)
        st.session_state.metadata["date"]=st.date_input("実施日",value=st.session_state.metadata.get("date",datetime.date.today()))
    st.session_state.metadata["exp_condition"]=st.text_area("実験条件 (温度・時間以外を自由記述)",value=st.session_state.metadata.get("exp_condition",""),height=100,placeholder="例: 中性洗剤0.1%, 浴比1:50, 振とう100rpm, pH7, 硬水, 洗剤種別, 前処理など")

    st.divider()
    st.markdown("### 画像アップロード - 3つの方法で貼り付け可能")

    # 方法1: 全体貼り付けエリア
    st.info("💡 **貼り付け方法3種**: 1) 下の緑エリアをクリック→Ctrl+V / Cmd+V  2) 各画像の📋貼り付けボタン  3) ファイルをドラッグ&ドロップ")

    st.components.v1.html("""
    <style>
    #paste-hint{border:2px dashed #4CAF50;border-radius:12px;padding:18px;text-align:center;background:#F1F8E9;font-family:sans-serif;cursor:pointer;transition:0.2s}
    #paste-hint:hover{background:#E8F5E9;border-color:#2E7D32}
    #paste-hint.dragover{background:#C8E6C9;border-color:#1B5E20;transform:scale(1.02)}
    #preview img{max-width:100%;max-height:220px;border-radius:8px;border:2px solid #4CAF50;margin-top:10px}
    </style>
    <div id="paste-hint" tabindex="0">
        <div style="font-size:22px">📋</div>
        <b>ここをクリックして Ctrl+V / Cmd+V</b><br>
        <small>スクリーンショット直後でもOK。画像ファイルをここにドラッグ&ドロップも可能</small>
        <div id="preview"></div>
        <div id="status" style="margin-top:8px;font-size:13px;color:#2E7D32"></div>
    </div>
    <script>
    const hint=document.getElementById('paste-hint');
    const preview=document.getElementById('preview');
    const status=document.getElementById('status');
    hint.focus();
    hint.addEventListener('click',()=>{hint.focus(); status.innerText='フォーカス中... Ctrl+Vで貼り付けしてください';});
    document.addEventListener('paste',(e)=>{
        let found=false;
        for(let item of e.clipboardData.items){
            if(item.type.indexOf('image')!==-1){
                found=true;
                const blob=item.getAsFile();
                const url=URL.createObjectURL(blob);
                preview.innerHTML='<img src="'+url+'"><div style="color:#2E7D32;margin-top:6px;font-weight:bold">✓ 画像を検出！下の「📋貼り付け」ボタンを押すか、この画像を下のアップローダーにドラッグしてください</div>';
                status.innerText='画像検出成功！';
            }
        }
        if(!found){ status.innerText='クリップボードに画像がありません。画像をコピーしてから貼り付けてください'; }
    });
    hint.addEventListener('dragover',(e)=>{e.preventDefault(); hint.classList.add('dragover');});
    hint.addEventListener('dragleave',()=>hint.classList.remove('dragover'));
    hint.addEventListener('drop',(e)=>{
        e.preventDefault(); hint.classList.remove('dragover');
        const file=e.dataTransfer.files[0];
        if(file && file.type.startsWith('image/')){
            const url=URL.createObjectURL(file);
            preview.innerHTML='<img src="'+url+'"><div style="color:#2E7D32;margin-top:6px">✓ ドロップ検出！下のボタンで取り込んでください</div>';
        }
    });
    </script>
    """, height=220)

    cols=st.columns(3)
    for idx,key in enumerate(["素地","洗浄前","洗浄後"]):
        with cols[idx]:
            st.markdown(f"#### {key}")
            # --- ファイルアップロード ---
            up=st.file_uploader(f"{key}ファイル",type=["png","jpg","jpeg","bmp","tiff","webp"],key=f"upload_{key}",label_visibility="collapsed",help="ここにCtrl+Vで直接貼り付けも可能 (Streamlit 1.35+)")
            if up:
                try:
                    pil_img=Image.open(up).convert("RGB")
                    st.session_state.images[key]=pil_img
                    st.session_state.calibrated_images[key]=None
                    st.session_state.masks[key]=None
                    st.success(f"{key} 読み込み成功")
                except Exception as e: st.error(f"読込失敗: {e}")

            # --- 貼り付けボタン (最重要) ---
            if HAS_PASTE:
                try:
                    # paste_buttonはクリックでクリップボード権限を要求
                    paste_result = paste_button(f"📋 クリップボードから貼り付け ({key})", key=f"paste_btn_{key}")
                    # デバッグ用に状態表示 (開発時のみ)
                    # st.write(f"paste_result type: {type(paste_result)}")
                    pil_from = None
                    if paste_result is not None:
                        # パターン1: 直接PIL
                        if isinstance(paste_result, Image.Image):
                            pil_from = paste_result.convert("RGB")
                        # パターン2: オブジェクトにimage_data
                        elif hasattr(paste_result, 'image_data') and paste_result.image_data is not None:
                            d = paste_result.image_data
                            if isinstance(d, Image.Image):
                                pil_from = d.convert("RGB")
                            elif isinstance(d, np.ndarray):
                                # numpy配列の場合
                                if d.dtype != np.uint8:
                                    d = (d*255).astype(np.uint8) if d.max()<=1 else d.astype(np.uint8)
                                pil_from = Image.fromarray(d).convert("RGB")
                            elif isinstance(d, str) and d.startswith("data:image"):
                                # base64 data URL
                                try:
                                    b64_part = d.split(",",1)[1]
                                    pil_from = Image.open(io.BytesIO(base64.b64decode(b64_part))).convert("RGB")
                                except Exception as ex:
                                    st.warning(f"Base64デコード失敗: {ex}")
                            elif isinstance(d, bytes):
                                try:
                                    pil_from = Image.open(io.BytesIO(d)).convert("RGB")
                                except: pass
                        # パターン3: 辞書
                        elif isinstance(paste_result, dict) and 'image_data' in paste_result:
                            d = paste_result['image_data']
                            if isinstance(d, str) and d.startswith("data:image"):
                                b64_part = d.split(",",1)[1]
                                pil_from = Image.open(io.BytesIO(base64.b64decode(b64_part))).convert("RGB")
                    if pil_from is not None:
                        st.session_state.images[key]=pil_from
                        st.session_state.calibrated_images[key]=None
                        st.session_state.masks[key]=None
                        st.success(f"✅ {key} 貼り付け成功! {pil_from.size[0]}x{pil_from.size[1]}")
                        # 強制リロードで画像表示を更新
                        st.rerun()
                    else:
                        # 貼り付けデータがない場合でも、ボタンが押されたことを示すために何もしない
                        pass
                except Exception as e:
                    st.error(f"貼り付けエラー {key}: {e}")
                    import traceback
                    st.code(traceback.format_exc())
            else:
                st.warning(f"貼り付けボタン無効: requirements.txtを確認")

            # --- 画像表示 ---
            if st.session_state.images[key] is not None:
                st.image(st.session_state.images[key], caption=f"{key} {st.session_state.images[key].size[0]}x{st.session_state.images[key].size[1]}", use_container_width=True)
            else:
                st.info(f"{key} 未登録")

            st.session_state.weights[key]=st.number_input(f"{key} 重量(g)",value=float(st.session_state.weights.get(key,0)),step=0.001,format="%.4f",key=f"weight_{key}")

with tab2:
    st.subheader("ステップ2: 黒白校正")
    if all(v is None for v in st.session_state.images.values()): st.warning("①で画像登録してください")
    else:
        for key in ["素地","洗浄前","洗浄後"]:
            pil_img=st.session_state.images.get(key)
            if pil_img is None: continue
            st.markdown(f"#### {key}")
            c1,c2,c3=st.columns([1,1,2])
            with c1:
                if st.button(f"自動抽出 {key}",key=f"auto_{key}"):
                    b,w=extract_bw_auto(pil_img)
                    st.session_state.black_white_points[key]["black"]=b
                    st.session_state.black_white_points[key]["white"]=w
                    st.rerun()
                cur=st.session_state.black_white_points[key]["black"]
                hex_c="#%02x%02x%02x"%cur
                pick=st.color_picker(f"黒 {key}",value=hex_c,key=f"bp_{key}")
                r=int(pick[1:3],16); g=int(pick[3:5],16); b=int(pick[5:7],16)
                st.session_state.black_white_points[key]["black"]=(r,g,b)
            with c2:
                cur=st.session_state.black_white_points[key]["white"]
                hex_c="#%02x%02x%02x"%cur
                pick=st.color_picker(f"白 {key}",value=hex_c,key=f"wp_{key}")
                r=int(pick[1:3],16); g=int(pick[3:5],16); b=int(pick[5:7],16)
                st.session_state.black_white_points[key]["white"]=(r,g,b)
            with c3:
                cal=apply_bw_calibration(pil_img, st.session_state.black_white_points[key]["black"], st.session_state.black_white_points[key]["white"])
                st.session_state.calibrated_images[key]=cal
                ca,cb=st.columns(2)
                with ca: st.image(pil_img,caption="元",use_container_width=True)
                with cb: st.image(cal,caption="補正後",use_container_width=True)
            st.divider()

with tab3:
    st.subheader("ステップ3: ROI & 自動検出")
    if all(v is None for v in st.session_state.images.values()): st.warning("画像なし")
    else:
        use_common=st.checkbox("共通ROI",value=st.session_state.roi["common"].get("use_common",True))
        st.session_state.roi["common"]["use_common"]=use_common
        with st.expander("検出パラメータ",expanded=True):
            c1,c2,c3=st.columns(3)
            with c1:
                blur_k=st.slider("Blur",1,21,5,step=2,key="blur_k")
                morph_k=st.slider("Morph",1,31,9,step=2,key="morph_k")
            with c2:
                use_otsu=st.checkbox("Otsu",value=True,key="use_otsu")
                thresh_val=st.slider("閾値",0,255,127,key="thresh_val",disabled=use_otsu)
                invert_mask=st.checkbox("反転",value=False,key="invert")
            with c3: min_area_ratio=st.slider("最小面積率%",1,50,5,key="min_area")
        def roi_ui(label, default, prefix):
            c1,c2,c3,c4=st.columns(4)
            with c1: x=st.slider(f"x {label}",0,90,int(default.get("x",5)),key=f"x_{prefix}")
            with c2: y=st.slider(f"y {label}",0,90,int(default.get("y",5)),key=f"y_{prefix}")
            with c3: w=st.slider(f"w {label}",10,100,int(default.get("w",90)),key=f"w_{prefix}")
            with c4: h=st.slider(f"h {label}",10,100,int(default.get("h",90)),key=f"h_{prefix}")
            return {"x":x,"y":y,"w":w,"h":h}
        keys=["素地","洗浄前","洗浄後"]
        if use_common:
            common_roi=roi_ui("共通",st.session_state.roi["common"],"common")
            st.session_state.roi["common"].update(common_roi)
            for key in keys:
                src=st.session_state.calibrated_images.get(key) or st.session_state.images.get(key)
                if src is None: continue
                W,H=src.size
                rx=int(W*common_roi["x"]/100); ry=int(H*common_roi["y"]/100); rw=int(W*common_roi["w"]/100); rh=int(H*common_roi["h"]/100)
                rx=max(0,min(rx,W-10)); ry=max(0,min(ry,H-10)); rw=max(10,min(rw,W-rx)); rh=max(10,min(rh,H-ry))
                roi=src.crop((rx,ry,rx+rw,ry+rh))
                mask,overlay,binary=detect_gauze_mask(roi,blur_k,thresh_val,use_otsu,morph_k,invert_mask,min_area_ratio)
                full_mask=np.zeros((H,W),dtype=np.uint8); full_mask[ry:ry+rh,rx:rx+rw]=mask
                st.session_state.masks[key]=full_mask
                c1,c2,c3,c4=st.columns(4)
                with c1:
                    cv_full=pil_to_cv(src); cv2.rectangle(cv_full,(rx,ry),(rx+rw,ry+rh),(0,255,0),3)
                    st.image(cv_to_pil(cv_full),caption="ROI",use_container_width=True)
                with c2: st.image(roi,caption="クロップ",use_container_width=True)
                with c3: st.image(binary,caption="二値",use_container_width=True)
                with c4: st.image(overlay,caption="オーバーレイ",use_container_width=True)
        else:
            for key in keys:
                src=st.session_state.calibrated_images.get(key) or st.session_state.images.get(key)
                if src is None: continue
                per=roi_ui(key,st.session_state.roi.get(key,{"x":5,"y":5,"w":90,"h":90}),key)
                st.session_state.roi[key]=per
                W,H=src.size
                rx=int(W*per["x"]/100); ry=int(H*per["y"]/100); rw=int(W*per["w"]/100); rh=int(H*per["h"]/100)
                rx=max(0,min(rx,W-10)); ry=max(0,min(ry,H-10)); rw=max(10,min(rw,W-rx)); rh=max(10,min(rh,H-ry))
                roi=src.crop((rx,ry,rx+rw,ry+rh))
                mask,overlay,binary=detect_gauze_mask(roi,blur_k,thresh_val,use_otsu,morph_k,invert_mask,min_area_ratio)
                full_mask=np.zeros((H,W),dtype=np.uint8); full_mask[ry:ry+rh,rx:rx+rw]=mask
                st.session_state.masks[key]=full_mask
                c1,c2,c3,c4=st.columns(4)
                with c1:
                    cv_full=pil_to_cv(src); cv2.rectangle(cv_full,(rx,ry),(rx+rw,ry+rh),(0,255,0),3)
                    st.image(cv_to_pil(cv_full),caption="ROI",use_container_width=True)
                with c2: st.image(roi,caption="クロップ",use_container_width=True)
                with c3: st.image(binary,caption="二値",use_container_width=True)
                with c4: st.image(overlay,caption="オーバーレイ",use_container_width=True)

with tab4:
    st.subheader("ステップ4: 解析")
    if st.button("🔬 解析実行",type="primary"):
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
    st.subheader("ステップ5: レポート")
    if st.session_state.get("results_df") is None: st.warning("先に解析実行")
    else:
        if st.button("📄 PDF生成",type="primary"):
            pdf=generate_pdf_buffer(st.session_state.metadata,st.session_state.weights,st.session_state.results_df,st.session_state.lab_data,st.session_state.wash_rates,st.session_state.images,st.session_state.calibrated_images)
            if pdf: st.session_state["pdf_buffer"]=pdf; st.success("生成完了")
        if "pdf_buffer" in st.session_state and st.session_state["pdf_buffer"]:
            st.download_button("⬇️ PDFダウンロード",data=st.session_state["pdf_buffer"],file_name=f"{st.session_state.metadata.get('exp_id','report')}_V6.2.pdf",mime="application/pdf")
        csv=st.session_state.results_df.to_csv(encoding="utf-8-sig")
        st.download_button("⬇️ CSV",data=csv,file_name="results.csv",mime="text/csv")
        meta_copy=st.session_state.metadata.copy()
        if isinstance(meta_copy.get("date"),(datetime.date,datetime.datetime)): meta_copy["date"]=meta_copy["date"].isoformat()
        json_str=json.dumps({"metadata":meta_copy,"weights":st.session_state.weights,"lab_data":st.session_state.lab_data,"wash_rates":st.session_state.wash_rates},ensure_ascii=False,indent=2,default=str)
        st.download_button("⬇️ JSON",data=json_str,file_name="LabWash.json",mime="application/json")
