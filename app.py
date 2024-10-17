import sys
sys.path.append('./pycode')

from pycode import predict, preprocess

import streamlit as st
import json
import pandas as pd


# Đọc config từ file config.json

with open("/Users/bimac/Documents/CTU/CT294/CT294-PAEBMR/config.json", 'r') as file:
    config = json.load(file)

# Streamlit UI setup
st.set_page_config(layout="wide", page_title="ECFMR", page_icon=config["icon"])

# Định nghĩa biến HTML đóng thẻ </div>
CLOSE_DIV = '</div>'

# Thêm CSS để tùy chỉnh bố cục
st.markdown(
    """
    <style>
    .centered-title, .full-screen, .column {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .centered-title {
        height: 80px;
        margin-top: -70px;
    }
    .full-screen {
        height: calc(100vh - 80px);
        width: 100%;
    }
    .column {
        flex-direction: column;
        height: 100%;
    }
    .column > * {
        margin: 10px 0;
        width: 80%;
    }
    .rounded-box {
        border: 2px solid #C0C0C0;
        border-radius: 15px;
        padding: 20px;
        margin: 20px;
        background-color: #f9f9f9;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        color: black;
    }
    .rounded-box p {
        font-size: 18px;
        text-align: justify;
    }
    .large-letter{
        font-size: 24px;
    }
    
    </style>
    """,
    unsafe_allow_html=True,
)

def submit_text(text):
    st.markdown(f"<p class='large-letter'><i>You entered:</i></p>", unsafe_allow_html=True)
    st.markdown(f"<div class='rounded-box'><p>{text}</p></div>", unsafe_allow_html=True)
    processed_review = preprocess.preprocess_review(text)
    classification = predict.classify_review(processed_review)
    st.markdown(f"<p class='large-letter'><i>Predicted result:</i> This is a <span style='color: red;'><strong>{classification.upper()}</strong></span> comment.</p>", unsafe_allow_html=True)

def submit_file(df):
    # Cột đầu tiên chứa các bình luận
    review_column = df.iloc[:, 0]
    # Tạo cột mới 'Predict' trong DataFrame để lưu kết quả phân loại
    df['Predict'] = ""
    # Duyệt qua từng dòng trong cột review
    for i, review in enumerate(review_column):
        # Xử lý và phân loại từng bình luận
        processed_review = preprocess.preprocess_review(review)
        classification = predict.classify_review(processed_review)
        
        # Lưu kết quả phân loại vào cột 'Predict'
        df.at[i, 'Predict'] = classification
    
    # Hiển thị kết quả sau khi phân loại
    st.markdown(f"<p class='large-letter'><i>Predicted result:</p>", unsafe_allow_html=True)
    st.dataframe(df, width=None, height=None)


# Hàng đầu tiên: Tiêu đề ở giữa trang
st.markdown('<div class="centered-title"><h1>Emotion Classification For Movie Reviews</h1>' +
            CLOSE_DIV, unsafe_allow_html=True)

st.markdown('<div class="column full-screen">', unsafe_allow_html=True)
user_text = st.text_area("Enter your text here", height=250)
# Hoặc tải lên tệp văn bản Excel
uploaded_file = st.file_uploader("Upload an Excel or CSV file", type=["xlsx", "xls", "csv"])
submit_button = st.button("Submit")
st.markdown(CLOSE_DIV, unsafe_allow_html=True)

if submit_button:


    if uploaded_file is not None and user_text:
        st.warning("Import file or text only.")
    elif uploaded_file is not None:
        # Kiểm tra định dạng của tệp để đọc tương ứng
        try:
            if uploaded_file.name.endswith(".csv"):
                # Đọc tệp CSV
                df = pd.read_csv(uploaded_file)
            else:
                # Đọc tệp Excel
                df = pd.read_excel(uploaded_file)
            if df.shape[1] == 1:
                submit_file(df)
            else:
                st.warning("Please enter upload a Excel file have one column.")
        except Exception as e:
            st.error(f"Error reading the file: {e}")
    elif user_text:
        # Hiển thị đoạn văn bản người dùng nhập
        submit_text(user_text)
    else:
        st.warning("Please enter text or upload a Excel file.")
# streamlit run /Users/bimac/Documents/CTU/CT294/CT294-PAEBMR/app.py