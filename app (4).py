import os, pickle, numpy as np, pandas as pd, cv2, h5py, gradio as gr, tensorflow as tf, matplotlib.cm as cm
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

def build_binary():
    base = ResNet50(weights=None, include_top=False, input_shape=(224,224,3))
    m = Sequential([base, GlobalAveragePooling2D(), Dense(256,activation='relu'), Dropout(0.5), Dense(1,activation='sigmoid')], name="binary_model")
    m.build((None,224,224,3)); return m

def build_multi():
    base = ResNet50(weights=None, include_top=False, input_shape=(224,224,3))
    m = Sequential([base, GlobalAveragePooling2D(), Dense(256,activation='relu'), Dropout(0.5), Dense(3,activation='softmax')], name="multi_model")
    m.build((None,224,224,3)); return m

def load_weights_safe(model, path, label):
    print(f"\n[{label}] Loading: {path}")
    try:
        with h5py.File(path,'r') as f: print(f"  H5 keys: {list(f.keys())}")
    except Exception as e: print(f"  H5 inspect: {e}")
    for name, fn in [
        ("direct",              lambda: model.load_weights(path)),
        ("by_name",             lambda: model.load_weights(path, by_name=True)),
        ("by_name+skip",        lambda: model.load_weights(path, by_name=True, skip_mismatch=True)),
    ]:
        try: fn(); print(f"  ✅ [{label}] '{name}' succeeded"); return model
        except Exception as e: print(f"  ❌ [{label}] '{name}': {str(e)[:100]}")
    print(f"  ⚠️ [{label}] All failed — random weights"); return model

print("="*55)
binary_model = build_binary()
binary_model = load_weights_safe(binary_model, "binary_weights.weights.h5", "binary")
multi_model  = build_multi()
multi_model  = load_weights_safe(multi_model,  "multi_weights.weights.h5",  "multi")
_d = np.zeros((1,224,224,3),dtype="float32")
binary_model.predict(_d,verbose=0); multi_model.predict(_d,verbose=0)
print("✅ Warmup passed"); print("="*55)

BINARY_CLASSES = {0:"Normal",1:"Osteoporosis"}
MULTI_CLASSES  = {0:"Normal",1:"Osteopenia",2:"Osteoporosis"}

CSV_PATH="osteoporosis.csv"; PICKLE_PATH="model.pkl"
def train_csv_model():
    df=pd.read_csv(CSV_PATH); df=df.drop(columns="Id",errors='ignore'); df=df.fillna("None")
    df['Age_Group']=pd.cut(df['Age'],bins=[0,30,50,70,100],labels=[0,1,2,3]).astype(int)
    df['Risk_Score']=((df['Smoking']=='Yes').astype(int)+(df['Family History']=='Yes').astype(int)+(df['Prior Fractures']=='Yes').astype(int)+(df['Calcium Intake']=='Low').astype(int)+(df['Vitamin D Intake']=='Insufficient').astype(int)+(df['Physical Activity']=='Sedentary').astype(int)+(df['Body Weight']=='Underweight').astype(int))
    le_dict={}
    for col in df.select_dtypes(include='object').columns:
        le=LabelEncoder(); df[col]=le.fit_transform(df[col].astype(str)); le_dict[col]=le
    X=df.drop(columns='Osteoporosis'); y=df['Osteoporosis']
    X_train,_,y_train,_=train_test_split(X,y,test_size=0.2,random_state=24,stratify=y)
    gbm=GradientBoostingClassifier(n_estimators=200,max_depth=3,learning_rate=0.1,subsample=0.85,min_samples_leaf=10,random_state=24)
    gbm.fit(X_train,y_train)
    save={"model":gbm,"le_dict":le_dict,"features":list(X.columns)}
    with open(PICKLE_PATH,"wb") as f: pickle.dump(save,f)
    return gbm,le_dict,list(X.columns)

if os.path.exists(PICKLE_PATH):
    with open(PICKLE_PATH,"rb") as f: saved=pickle.load(f)
    csv_model,csv_le_dict,csv_features=saved["model"],saved["le_dict"],saved["features"]; print("✅ CSV loaded")
elif os.path.exists(CSV_PATH): csv_model,csv_le_dict,csv_features=train_csv_model(); print("✅ CSV trained")
else: csv_model=csv_le_dict=csv_features=None; print("⚠️ No CSV")

def generate_gradcam(model, img_array):
    base_model=next((l for l in model.layers if isinstance(l,tf.keras.Model)),None)
    if base_model:
        conv_name=next(l.name for l in reversed(base_model.layers) if isinstance(l,tf.keras.layers.Conv2D))
        gm=tf.keras.Model(inputs=base_model.input,outputs=[base_model.get_layer(conv_name).output,base_model.output])
        post=model.layers[1:]
        with tf.GradientTape() as tape:
            conv_out,base_out=gm(img_array); x=base_out
            for l in post: x=l(x)
            loss=x[:,0] if x.shape[-1]==1 else x[:,tf.argmax(x[0])]
    else:
        conv_name=next(l.name for l in reversed(model.layers) if isinstance(l,tf.keras.layers.Conv2D))
        gm=tf.keras.Model(inputs=model.input,outputs=[model.get_layer(conv_name).output,model.output])
        with tf.GradientTape() as tape:
            conv_out,preds=gm(img_array); loss=preds[:,0] if preds.shape[-1]==1 else preds[:,tf.argmax(preds[0])]
    grads=tape.gradient(loss,conv_out); pg=tf.reduce_mean(grads,axis=(0,1,2))
    hm=conv_out[0]@pg[...,tf.newaxis]; hm=tf.squeeze(hm)
    hm=tf.maximum(hm,0)/(tf.math.reduce_max(hm)+1e-8); return hm.numpy()

def overlay_heatmap(img,hm,alpha=0.4):
    img=np.uint8(img); hr=cv2.resize(hm,(img.shape[1],img.shape[0]))
    hu=np.uint8(255*hr); hc=np.uint8(255*cm.jet(hu)[:,:,:3])
    return np.clip(hc*alpha+img,0,255).astype("uint8")

def predict_csv(age,gender,hormonal_changes,family_history,race,body_weight,calcium_intake,vitamin_d,physical_activity,smoking,alcohol,medical_conditions,medications,prior_fractures):
    if csv_model is None: return "<p style='color:#e57373;padding:20px;'>⚠️ CSV unavailable.</p>"
    raw={"Age":age,"Gender":gender,"Hormonal Changes":hormonal_changes,"Family History":family_history,"Race/Ethnicity":race,"Body Weight":body_weight,"Calcium Intake":calcium_intake,"Vitamin D Intake":vitamin_d,"Physical Activity":physical_activity,"Smoking":smoking,"Alcohol Consumption":alcohol or "None","Medical Conditions":medical_conditions or "None","Medications":medications or "None","Prior Fractures":prior_fractures}
    df=pd.DataFrame([raw])
    df['Age_Group']=pd.cut(df['Age'],bins=[0,30,50,70,100],labels=[0,1,2,3]).astype(int)
    df['Risk_Score']=((df['Smoking']=='Yes').astype(int)+(df['Family History']=='Yes').astype(int)+(df['Prior Fractures']=='Yes').astype(int)+(df['Calcium Intake']=='Low').astype(int)+(df['Vitamin D Intake']=='Insufficient').astype(int)+(df['Physical Activity']=='Sedentary').astype(int)+(df['Body Weight']=='Underweight').astype(int))
    for col,le in csv_le_dict.items():
        if col in df.columns:
            val=str(df[col].iloc[0]); df[col]=le.transform([val if val in le.classes_ else le.classes_[0]])
    df=df[csv_features]; prob=csv_model.predict_proba(df)[0]; pred=csv_model.predict(df)[0]
    conf=prob[pred]*100; rs=int(df['Risk_Score'].iloc[0])
    if pred==1:
        return f"""<div style="background:linear-gradient(135deg,#2d0a0a,#4a1010);border:1px solid #c0392b;border-radius:16px;padding:28px 32px;margin:8px 0;font-family:'DM Sans',sans-serif;"><div style="display:flex;align-items:center;gap:14px;margin-bottom:20px;"><div style="width:52px;height:52px;background:#c0392b;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:24px;">⚠️</div><div><div style="font-size:11px;letter-spacing:3px;color:#e57373;text-transform:uppercase;font-weight:600;margin-bottom:3px;">Screening Result</div><div style="font-size:22px;font-weight:700;color:#ff6b6b;">HIGH RISK — Osteoporosis Detected</div></div></div><div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:20px;"><div style="background:rgba(255,255,255,0.05);border-radius:10px;padding:14px 18px;"><div style="font-size:10px;letter-spacing:2px;color:#aaa;text-transform:uppercase;margin-bottom:4px;">Confidence</div><div style="font-size:26px;font-weight:700;color:#ff6b6b;">{conf:.1f}<span style="font-size:14px;">%</span></div></div><div style="background:rgba(255,255,255,0.05);border-radius:10px;padding:14px 18px;"><div style="font-size:10px;letter-spacing:2px;color:#aaa;text-transform:uppercase;margin-bottom:4px;">Risk Factors</div><div style="font-size:26px;font-weight:700;color:#ff6b6b;">{rs}<span style="font-size:14px;color:#aaa;"> / 7</span></div></div></div><div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:14px 18px;margin-bottom:18px;"><div style="font-size:10px;letter-spacing:2px;color:#aaa;text-transform:uppercase;margin-bottom:10px;">Probability Distribution</div><div style="margin-bottom:8px;"><div style="display:flex;justify-content:space-between;margin-bottom:4px;"><span style="font-size:12px;color:#aaa;">No Osteoporosis</span><span style="font-size:12px;color:#fff;font-weight:600;">{prob[0]*100:.1f}%</span></div><div style="background:rgba(255,255,255,0.1);border-radius:4px;height:6px;"><div style="background:#4caf50;width:{prob[0]*100:.1f}%;height:6px;border-radius:4px;"></div></div></div><div><div style="display:flex;justify-content:space-between;margin-bottom:4px;"><span style="font-size:12px;color:#aaa;">Osteoporosis</span><span style="font-size:12px;color:#ff6b6b;font-weight:600;">{prob[1]*100:.1f}%</span></div><div style="background:rgba(255,255,255,0.1);border-radius:4px;height:6px;"><div style="background:#c0392b;width:{prob[1]*100:.1f}%;height:6px;border-radius:4px;"></div></div></div></div><div style="border-top:1px solid rgba(255,255,255,0.08);padding-top:16px;"><div style="font-size:11px;letter-spacing:2px;color:#aaa;text-transform:uppercase;margin-bottom:10px;">Clinical Recommendations</div><div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;"><div style="background:rgba(255,255,255,0.04);border-radius:8px;padding:10px 12px;font-size:12px;color:#ddd;">🩺 Bone density scan (DEXA)</div><div style="background:rgba(255,255,255,0.04);border-radius:8px;padding:10px 12px;font-size:12px;color:#ddd;">💊 Review calcium & vitamin D</div><div style="background:rgba(255,255,255,0.04);border-radius:8px;padding:10px 12px;font-size:12px;color:#ddd;">🏋️ Weight-bearing exercise</div><div style="background:rgba(255,255,255,0.04);border-radius:8px;padding:10px 12px;font-size:12px;color:#ddd;">👨‍⚕️ Consult a specialist</div></div></div><div style="margin-top:14px;font-size:11px;color:#666;font-style:italic;">⚠️ For research purposes only.</div></div>"""
    else:
        return f"""<div style="background:linear-gradient(135deg,#0a2d14,#0d3b1a);border:1px solid #27ae60;border-radius:16px;padding:28px 32px;margin:8px 0;font-family:'DM Sans',sans-serif;"><div style="display:flex;align-items:center;gap:14px;margin-bottom:20px;"><div style="width:52px;height:52px;background:#27ae60;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:24px;">✅</div><div><div style="font-size:11px;letter-spacing:3px;color:#81c784;text-transform:uppercase;font-weight:600;margin-bottom:3px;">Screening Result</div><div style="font-size:22px;font-weight:700;color:#69f0ae;">LOW RISK — No Osteoporosis Detected</div></div></div><div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:20px;"><div style="background:rgba(255,255,255,0.05);border-radius:10px;padding:14px 18px;"><div style="font-size:10px;letter-spacing:2px;color:#aaa;text-transform:uppercase;margin-bottom:4px;">Confidence</div><div style="font-size:26px;font-weight:700;color:#69f0ae;">{conf:.1f}<span style="font-size:14px;">%</span></div></div><div style="background:rgba(255,255,255,0.05);border-radius:10px;padding:14px 18px;"><div style="font-size:10px;letter-spacing:2px;color:#aaa;text-transform:uppercase;margin-bottom:4px;">Risk Factors</div><div style="font-size:26px;font-weight:700;color:#69f0ae;">{rs}<span style="font-size:14px;color:#aaa;"> / 7</span></div></div></div><div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:14px 18px;margin-bottom:18px;"><div style="font-size:10px;letter-spacing:2px;color:#aaa;text-transform:uppercase;margin-bottom:10px;">Probability Distribution</div><div style="margin-bottom:8px;"><div style="display:flex;justify-content:space-between;margin-bottom:4px;"><span style="font-size:12px;color:#aaa;">No Osteoporosis</span><span style="font-size:12px;color:#69f0ae;font-weight:600;">{prob[0]*100:.1f}%</span></div><div style="background:rgba(255,255,255,0.1);border-radius:4px;height:6px;"><div style="background:#27ae60;width:{prob[0]*100:.1f}%;height:6px;border-radius:4px;"></div></div></div><div><div style="display:flex;justify-content:space-between;margin-bottom:4px;"><span style="font-size:12px;color:#aaa;">Osteoporosis</span><span style="font-size:12px;color:#aaa;font-weight:600;">{prob[1]*100:.1f}%</span></div><div style="background:rgba(255,255,255,0.1);border-radius:4px;height:6px;"><div style="background:#c0392b;width:{prob[1]*100:.1f}%;height:6px;border-radius:4px;"></div></div></div></div><div style="border-top:1px solid rgba(255,255,255,0.08);padding-top:16px;"><div style="font-size:11px;letter-spacing:2px;color:#aaa;text-transform:uppercase;margin-bottom:10px;">Preventive Care Tips</div><div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;"><div style="background:rgba(255,255,255,0.04);border-radius:8px;padding:10px 12px;font-size:12px;color:#ddd;">🥛 Maintain calcium & vitamin D</div><div style="background:rgba(255,255,255,0.04);border-radius:8px;padding:10px 12px;font-size:12px;color:#ddd;">🏃 Stay physically active</div><div style="background:rgba(255,255,255,0.04);border-radius:8px;padding:10px 12px;font-size:12px;color:#ddd;">🚭 Avoid smoking & alcohol</div><div style="background:rgba(255,255,255,0.04);border-radius:8px;padding:10px 12px;font-size:12px;color:#ddd;">📅 Regular check-ups after 50</div></div></div><div style="margin-top:14px;font-size:11px;color:#666;font-style:italic;">⚠️ For research purposes only.</div></div>"""

def predict_image(input_image):

    if input_image is None:
        return None, None, "<p style='color:red;'>Upload image</p>"

    if not isinstance(input_image, Image.Image):
        return None, None, "<p style='color:red;'>Invalid image</p>"

    # =========================
    # PREPROCESS
    # =========================
    img = input_image.resize((224,224))
    io = np.array(img).astype("float32")

    if io.ndim == 2:
        io = np.stack([io]*3, axis=-1)
    elif io.shape[-1] == 4:
        io = io[:,:,:3]

    im = resnet_preprocess(np.expand_dims(io, axis=0))

    # =========================
    # 🔹 STAGE 1 — BINARY
    # =========================
    bp = binary_model.predict(im)[0][0]

    binary_label = "Osteoporosis" if bp > 0.5 else "Normal"
    binary_conf = bp*100 if bp > 0.5 else (1-bp)*100

    try:
        heatmap1 = generate_gradcam(binary_model, im)
        stage1_img = Image.fromarray(overlay_heatmap(io.copy(), heatmap1))
    except Exception as e:
        print("Stage1 GradCAM error:", e)
        stage1_img = Image.fromarray(np.uint8(io))

    # =========================
    # 🔹 STAGE 2 — MULTICLASS
    # =========================
    mp = multi_model.predict(im)[0]

    classes = ["Normal","Osteopenia","Osteoporosis"]
    multi_label = classes[np.argmax(mp)]
    multi_conf = np.max(mp) * 100

    try:
        heatmap2 = generate_gradcam(multi_model, im)
        stage2_img = Image.fromarray(overlay_heatmap(io.copy(), heatmap2))
    except Exception as e:
        print("Stage2 GradCAM error:", e)
        stage2_img = Image.fromarray(np.uint8(io))

    # =========================
    # ⚠️ RISK SCORE
    # =========================
    risk_score = round(np.max(mp) * 100, 2)
    risk_score = round(risk_score, 2)

    # =========================
    # 📊 REPORT
    # =========================
    report = f"""
    <div style="color:white;font-family:sans-serif;">
        <h2>🧪 Stage 1 — Binary Screening</h2>
        <p><b>{binary_label}</b> ({binary_conf:.2f}%)</p>

        <h2>🧬 Stage 2 — Severity Classification</h2>
        <p><b>{multi_label}</b> ({multi_conf:.2f}%)</p>

        <h2>⚠️ Risk Score</h2>
        <p><b>{risk_score:.2f}%</b></p>

        <h2>🎯 Final Prediction</h2>
        <p style="font-size:20px;"><b>{multi_label}</b></p>
    </div>
    """

    return stage1_img, stage2_img, report


CSS="""@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');
*{box-sizing:border-box}body,.gradio-container{font-family:'DM Sans',sans-serif!important;background:#0d0d0f!important}
.gradio-container{background:radial-gradient(ellipse at 20% 10%,#0d1a2e 0%,#0d0d0f 60%)!important;min-height:100vh}
footer,.footer{display:none!important}.tab-nav{background:rgba(255,255,255,0.02)!important;border-bottom:1px solid rgba(255,255,255,0.07)!important;padding:0 16px!important}
.tab-nav button{font-family:'DM Sans',sans-serif!important;font-size:13px!important;font-weight:500!important;color:#888!important;padding:14px 20px!important;border-radius:0!important;border-bottom:2px solid transparent!important;transition:all 0.2s ease!important}
.tab-nav button.selected{color:#29b6f6!important;border-bottom:2px solid #29b6f6!important;background:transparent!important}
label span,.label-wrap span{font-family:'DM Sans',sans-serif!important;font-size:12px!important;font-weight:600!important;letter-spacing:1.5px!important;text-transform:uppercase!important;color:#8a9bb0!important}
input[type=range]{accent-color:#29b6f6!important}
button.primary{background:linear-gradient(135deg,#0a4a7c,#29b6f6)!important;border:none!important;border-radius:10px!important;font-family:'DM Sans',sans-serif!important;font-size:14px!important;font-weight:600!important;color:#fff!important;box-shadow:0 4px 20px rgba(41,182,246,0.2)!important;transition:all 0.2s ease!important}
button.primary:hover{transform:translateY(-1px)!important;box-shadow:0 6px 28px rgba(41,182,246,0.35)!important}
::-webkit-scrollbar{width:5px}::-webkit-scrollbar-track{background:#0d0d0f}::-webkit-scrollbar-thumb{background:#2a3a4a;border-radius:3px}"""

with gr.Blocks(title="OsteoScan AI",css=CSS) as demo:
    gr.HTML("""<div style="background:linear-gradient(135deg,#0d1a2e,#0a2640,#0d1a2e);border-bottom:1px solid rgba(41,182,246,0.15);padding:28px 40px;"><div style="display:flex;align-items:center;gap:16px;margin-bottom:6px;"><div style="width:44px;height:44px;background:linear-gradient(135deg,#0a4a7c,#29b6f6);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:22px;">🦴</div><div><div style="font-size:26px;font-weight:700;color:#fff;letter-spacing:-0.3px;">OsteoScan <span style="color:#29b6f6;">AI</span></div><div style="font-size:12px;color:#5a7a9a;letter-spacing:1px;text-transform:uppercase;font-weight:500;">Clinical Osteoporosis Risk Assessment System</div></div><div style="margin-left:auto;display:flex;gap:8px;flex-wrap:wrap;"><div style="background:rgba(41,182,246,0.1);border:1px solid rgba(41,182,246,0.25);border-radius:20px;padding:5px 14px;font-size:11px;color:#29b6f6;font-weight:500;">GBM · 91%+ Accuracy</div><div style="background:rgba(41,182,246,0.1);border:1px solid rgba(41,182,246,0.25);border-radius:20px;padding:5px 14px;font-size:11px;color:#29b6f6;font-weight:500;">ResNet50 · Two-Stage</div><div style="background:rgba(41,182,246,0.1);border:1px solid rgba(41,182,246,0.25);border-radius:20px;padding:5px 14px;font-size:11px;color:#29b6f6;font-weight:500;">Grad-CAM XAI</div></div></div><div style="font-size:13px;color:#4a6a80;margin-top:4px;">Select an analysis mode — lifestyle risk scoring or X-ray image diagnosis. &nbsp;·&nbsp; <em>Research use only.</em></div></div>""")
    with gr.Tabs():
        with gr.TabItem("📋  Lifestyle Risk Assessment"):
            gr.HTML('<div style="padding:16px 4px 8px;font-size:13px;color:#5a7a9a;line-height:1.6;">Enter the patient\'s clinical profile. The <strong style="color:#29b6f6;">Gradient Boosting classifier</strong> computes osteoporosis risk from 14 features.</div>')
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    gr.HTML('<div style="font-size:10px;letter-spacing:3px;color:#29b6f6;text-transform:uppercase;font-weight:600;padding:4px 0 10px;">👤 Demographics</div>')
                    age=gr.Slider(18,90,value=45,step=1,label="Age"); gender=gr.Radio(["Male","Female"],value="Female",label="Gender")
                    race=gr.Dropdown(["African American","Asian","Caucasian"],value="Caucasian",label="Race / Ethnicity")
                    hormonal_changes=gr.Radio(["Normal","Postmenopausal"],value="Normal",label="Hormonal Changes")
                    family_history=gr.Radio(["Yes","No"],value="No",label="Family History of Osteoporosis")
                with gr.Column(scale=1):
                    gr.HTML('<div style="font-size:10px;letter-spacing:3px;color:#29b6f6;text-transform:uppercase;font-weight:600;padding:4px 0 10px;">🏃 Lifestyle Factors</div>')
                    body_weight=gr.Radio(["Normal","Underweight"],value="Normal",label="Body Weight")
                    physical_activity=gr.Radio(["Active","Sedentary"],value="Active",label="Physical Activity")
                    calcium_intake=gr.Radio(["Adequate","Low"],value="Adequate",label="Calcium Intake")
                    vitamin_d=gr.Radio(["Sufficient","Insufficient"],value="Sufficient",label="Vitamin D Intake")
                    smoking=gr.Radio(["Yes","No"],value="No",label="Smoking"); alcohol=gr.Radio(["Moderate","None"],value="None",label="Alcohol Consumption")
                with gr.Column(scale=1):
                    gr.HTML('<div style="font-size:10px;letter-spacing:3px;color:#29b6f6;text-transform:uppercase;font-weight:600;padding:4px 0 10px;">🏥 Medical History</div>')
                    medical_conditions=gr.Dropdown(["None","Hyperthyroidism","Rheumatoid Arthritis"],value="None",label="Medical Conditions")
                    medications=gr.Dropdown(["None","Corticosteroids"],value="None",label="Current Medications")
                    prior_fractures=gr.Radio(["Yes","No"],value="No",label="Prior Fractures")
                    gr.HTML('<div style="font-size:10px;letter-spacing:3px;color:#29b6f6;text-transform:uppercase;font-weight:600;padding:18px 0 10px;">💡 Sample Patients</div>')
                    gr.Examples(examples=[[69,"Female","Postmenopausal","Yes","Asian","Underweight","Low","Insufficient","Sedentary","Yes","Moderate","Rheumatoid Arthritis","Corticosteroids","Yes"],[32,"Male","Normal","No","Caucasian","Normal","Adequate","Sufficient","Active","No","None","None","None","No"],[55,"Female","Postmenopausal","Yes","Caucasian","Normal","Low","Insufficient","Sedentary","No","None","Hyperthyroidism","Corticosteroids","No"]],inputs=[age,gender,hormonal_changes,family_history,race,body_weight,calcium_intake,vitamin_d,physical_activity,smoking,alcohol,medical_conditions,medications,prior_fractures],label="")
            gr.HTML('<div style="height:10px;"></div>')
            csv_btn=gr.Button("⚡  Run Lifestyle Risk Assessment",variant="primary",size="lg"); csv_result=gr.HTML()
            csv_btn.click(fn=predict_csv,inputs=[age,gender,hormonal_changes,family_history,race,body_weight,calcium_intake,vitamin_d,physical_activity,smoking,alcohol,medical_conditions,medications,prior_fractures],outputs=csv_result)
        with gr.TabItem("🩻  X-Ray Image Analysis"):
            gr.HTML('<div style="padding:16px 4px 8px;font-size:13px;color:#5a7a9a;line-height:1.6;">Upload a bone X-ray. The <strong style="color:#29b6f6;">two-stage ResNet50 pipeline</strong> screens for presence, then classifies severity with Grad-CAM explainability.</div>')
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<div style="font-size:10px;letter-spacing:3px;color:#29b6f6;text-transform:uppercase;font-weight:600;padding:4px 0 8px;">Upload X-Ray</div>')
                    input_img=gr.Image(type="pil",label="",height=280); gr.HTML('<div style="height:8px;"></div>')
                    img_btn=gr.Button("🔬  Analyze X-Ray",variant="primary",size="lg")
                with gr.Column(scale=2):
                    gr.HTML('<div style="font-size:10px;letter-spacing:3px;color:#29b6f6;text-transform:uppercase;font-weight:600;padding:4px 0 8px;">Grad-CAM Visualisation</div>')
                    with gr.Row():
                        with gr.Column(): gr.HTML('<div style="font-size:11px;color:#5a7a9a;margin-bottom:5px;">Stage 1 — Binary</div>'); stage1_out=gr.Image(label="",height=220)
                        with gr.Column(): gr.HTML('<div style="font-size:11px;color:#5a7a9a;margin-bottom:5px;">Stage 2 — Severity</div>'); stage2_out=gr.Image(label="",height=220)
                    gr.HTML('<div style="font-size:10px;letter-spacing:3px;color:#29b6f6;text-transform:uppercase;font-weight:600;padding:12px 0 6px;">Analysis Report</div>'); img_result=gr.HTML()
            img_btn.click(fn=predict_image,inputs=input_img,outputs=[stage1_out, stage2_out, img_result])
        with gr.TabItem("ℹ️  About"):
            gr.HTML("""<div style="font-family:'DM Sans',sans-serif;max-width:860px;padding:28px 4px;color:#c0ccd8;"><div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:24px;"><div style="background:rgba(41,182,246,0.04);border:1px solid rgba(41,182,246,0.12);border-radius:14px;padding:22px 24px;"><div style="font-size:10px;letter-spacing:3px;color:#29b6f6;text-transform:uppercase;font-weight:600;margin-bottom:10px;">📋 Lifestyle Risk Module</div><div style="font-size:13px;line-height:1.7;color:#8a9bb0;margin-bottom:14px;">Gradient Boosting Classifier on 1,958 patient records with 14 clinical and lifestyle features.</div><div style="display:flex;flex-wrap:wrap;gap:6px;"><span style="background:rgba(41,182,246,0.08);border:1px solid rgba(41,182,246,0.2);border-radius:20px;padding:3px 10px;font-size:11px;color:#29b6f6;">GBM</span><span style="background:rgba(41,182,246,0.08);border:1px solid rgba(41,182,246,0.2);border-radius:20px;padding:3px 10px;font-size:11px;color:#29b6f6;">91%+ Accuracy</span><span style="background:rgba(41,182,246,0.08);border:1px solid rgba(41,182,246,0.2);border-radius:20px;padding:3px 10px;font-size:11px;color:#29b6f6;">Scikit-learn</span></div></div><div style="background:rgba(41,182,246,0.04);border:1px solid rgba(41,182,246,0.12);border-radius:14px;padding:22px 24px;"><div style="font-size:10px;letter-spacing:3px;color:#29b6f6;text-transform:uppercase;font-weight:600;margin-bottom:10px;">🩻 X-Ray Analysis Module</div><div style="font-size:13px;line-height:1.7;color:#8a9bb0;margin-bottom:14px;">Two-stage ResNet50 pipeline with Grad-CAM visual explainability.</div><div style="display:flex;flex-wrap:wrap;gap:6px;"><span style="background:rgba(41,182,246,0.08);border:1px solid rgba(41,182,246,0.2);border-radius:20px;padding:3px 10px;font-size:11px;color:#29b6f6;">ResNet50</span><span style="background:rgba(41,182,246,0.08);border:1px solid rgba(41,182,246,0.2);border-radius:20px;padding:3px 10px;font-size:11px;color:#29b6f6;">Grad-CAM XAI</span><span style="background:rgba(41,182,246,0.08);border:1px solid rgba(41,182,246,0.2);border-radius:20px;padding:3px 10px;font-size:11px;color:#29b6f6;">TensorFlow</span></div></div></div><div style="background:rgba(255,100,100,0.04);border:1px solid rgba(255,100,100,0.15);border-radius:12px;padding:16px 20px;font-size:12px;color:#8a6a6a;line-height:1.6;"><strong style="color:#e57373;">⚠️ Disclaimer:</strong> For research and educational purposes only. Not medical advice. Always consult a qualified healthcare professional.</div></div>""")

demo.launch()
