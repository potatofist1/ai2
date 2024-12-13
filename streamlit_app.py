#분류 결과 + 이미지 + 텍스트와 함께 분류 결과에 따라 다른 출력 보여주기
#파일 이름 streamlit_app.py
import streamlit as st
from fastai.vision.all import *
from PIL import Image
import gdown

# Google Drive 파일 ID
file_id = '1WVUPb4R8lbnjhiD2zfMvtRPCy4rl-c2y'

# Google Drive에서 파일 다운로드 함수
@st.cache(allow_output_mutation=True)
def load_model_from_drive(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'model.pkl'
    gdown.download(url, output, quiet=False)

    # Fastai 모델 로드
    learner = load_learner(output)
    return learner

def display_left_content(image, prediction, probs, labels):
    st.write("### 왼쪽: 기존 출력 결과")
    if image is not None:
        st.image(image, caption="업로드된 이미지", use_container_width=True)
    st.write(f"예측된 클래스: {prediction}")
    st.markdown("<h4>클래스별 확률:</h4>", unsafe_allow_html=True)
    for label, prob in zip(labels, probs):
        st.markdown(f"""
            <div style="background-color: #f0f0f0; border-radius: 5px; padding: 5px; margin: 5px 0;">
                <strong style="color: #333;">{label}:</strong>
                <div style="background-color: #d3d3d3; border-radius: 5px; width: 100%; padding: 2px;">
                    <div style="background-color: #4CAF50; width: {prob*100}%; padding: 5px 0; border-radius: 5px; text-align: center; color: white;">
                        {prob:.4f}
                    </div>
                </div>
        """, unsafe_allow_html=True)

def display_right_content(prediction, data):
    st.write("### 오른쪽: 동적 분류 결과")
    cols = st.columns(2)

    # 1st Row - Images
    for i in range(1):
        with cols[i]:
            st.image(data['images'][i], caption=f"이미지: {prediction}", use_container_width=True)
    # 3rd Row - Text
    for i in range(1):
        with cols[i]:
            st.write(data['texts'][i])

# 모델 로드
st.write("모델을 로드 중입니다. 잠시만 기다려주세요...")
learner = load_model_from_drive(file_id)
st.success("모델이 성공적으로 로드되었습니다!")

labels = learner.dls.vocab

# 스타일링을 통해 페이지 마진 줄이기
st.markdown("""
    <style>
    .reportview-container .main .block-container {
        max-width: 90%;
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 분류에 따라 다른 콘텐츠 관리
content_data = {
    labels[0]: {
        'images': ["https://i.ibb.co/28d5XfN/1.jpg"],
        'texts': ["화농성 여드름의 주요 원인은 피지과다 미생물 번식 호르몬 변화가 있습니다.화농성 여드름을 가라앉히는 방법에는 세안하기,약물치료,여드름 패치,청소하기,자외선 차단하기 등이 도움이 될 수 있습니다."]},
    labels[1]: {
        'images': ["https://i.ibb.co/FYjtVcS/50.jpg"],
        'texts': ["낭포성 여드름은 농포가 밖으로 터져 나오지 못하고 안에서 누적되어 결절 형태를 이루는 상태를 말합니다.이는 주로 피부유형과 더불어 호르몬 변화 스트레스,영양불균형,또는 잘못된 피부관리로 인해 발생할 수 있습니다.낭포성 여드름을 가라앉히는 방법에는 부드럽고 순한 클렌징 제품을 사용하여 적절한 피부관리가 필요하고,지속적인 보습,식습관 관리,약물치료,레이저 치료등이 도움이 될수 있습니다."]},
    labels[2]: {
        'images': ["https://i.ibb.co/jf714hN/images-1.jpg"],
        'texts': ["결절성 여드름은 여드름진행 단계중 3단계에 해당하는 중증도 여드름으로, 구진과 농포를 거쳐 염증반응이 지속됨에 따라 정상적이 모낭 내벽의 형태가 파괴되고 여러개의 염증이 합쳐져 피부속에 더 크고 깊게 자리하는 것이 특성입니다."결절성 여드름을 가라앉히는 방법에는 염증주사,식습관 개선,예를 들어 맵고 자극적인 음식을 먹는 것을 피하는 것 그리고 클렌징을 통해 세안을 해주는 것이 도움이 될 수 있습니다."]
    }
}

# 레이아웃 설정
left_column, right_column = st.columns([1, 2])  # 왼쪽과 오른쪽의 비율 조정

# 파일 업로드 컴포넌트 (jpg, png, jpeg, webp, tiff 지원)
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg", "webp", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = PILImage.create(uploaded_file)
    prediction, _, probs = learner.predict(img)

    with left_column:
        display_left_content(image, prediction, probs, labels)

    with right_column:
        # 분류 결과에 따른 콘텐츠 선택
        data = content_data.get(prediction, {
            'images': ["https://via.placeholder.com/300"] * 3,
            'videos': ["https://www.youtube.com/watch?v=3JZ_D3ELwOQ"] * 3,
            'texts': ["기본 텍스트"] * 3
        })
        display_right_content(prediction, data)

