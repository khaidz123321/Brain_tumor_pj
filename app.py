import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input

# 1. Cấu hình trang web
st.set_page_config(page_title="AI Khám U Não", page_icon="🧠", layout="centered")

# 2. Xây dựng cấu trúc và Tải trọng số (Bypass Keras bug)
@st.cache_resource
def load_brain_model():
    model_path = os.path.join("pred", "BrainTumor_ResNet50_Final.keras") 
    
    if not os.path.exists(model_path):
        st.error(f"Không tìm thấy file mô hình tại: {model_path}")
        st.stop()
        
    # Bước A: Dựng lại CHÍNH XÁC "khung xương" giống hệt file main.ipynb của bạn
    base_model = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
    
    model = Sequential([
        base_model,
        Flatten(), # Vẫn giữ Flatten vì bạn đã train với nó
        Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    
    # Bước B: Chỉ nạp "trọng số" (Cách này né được hoàn toàn lỗi List Object của Keras)
    model.load_weights(model_path)
    
    return model

# Khởi tạo mô hình
model = load_brain_model()

# 3. Giao diện chính
st.title("CDUNAO")
st.write("Sức mạnh bởi ResNet50")
st.divider()

# 4. Khu vực tải ảnh lên
uploaded_file = st.file_uploader("Tải ảnh MRI não lên (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ảnh chụp MRI")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Kết quả phân tích")
        with st.spinner('AI đang quét...'):
            # Tiền xử lý ảnh ĐÚNG CHUẨN ResNet50
            img_array = np.array(image.convert('RGB'))
            img_array = cv2.resize(img_array, (224, 224))
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array.astype('float32'))

            # Dự đoán
            prediction = model.predict(img_array)
            prob_no = prediction[0][0]
            prob_yes = prediction[0][1]

            # Hiển thị kết quả
            if prob_yes > 0.5:
                st.error("CẢNH BÁO: Phát hiện dấu hiệu U não!")
                st.write(f"**Độ tin cậy:** {prob_yes * 100:.2f}%")
                st.progress(float(prob_yes))
            else:
                st.success("AN TOÀN: Không phát hiện khối u.")
                st.write(f"**Độ tin cậy:** {prob_no * 100:.2f}%")
                st.progress(float(prob_no))
                
    st.divider()
    if st.button("Chẩn đoán ảnh mới"):
        st.rerun()