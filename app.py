# jsonify: 사용자가 json data를 내보내도록 제공하는 flask의 함수
#           파이썬 객체를 JSON 형식으로 변환해주고, 이를 HTTP 응답으로 반환해줍니다.

from flask import Flask, jsonify, request
import torch
import torch.nn as nn

# 데이터 처리의 분석을 위한 라이브러리.
# 행과 열로 이루어진 데이터 객체를 만들어 다룰 수 있음.
# 대용량의 데이터들을 처리하는데 매우 편리
import pandas as pd

# 행렬/배열 처리 및 연산
# 난수생성
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
device = torch.device('cpu')


class Custom_Dataset(Dataset):
    def __init__(self, X, y, train_mode=True, transforms=None):
        self.X = X
        self.y = y
        self.train_mode = train_mode
        self.transforms = transforms

    def __getitem__(self, index):
        X = self.X[index]

        if self.transforms is not None:
            X = self.transforms(X)

        if self.train_mode:
            y = self.y[index]
            return X, y
        else:
            return X

    def __len__(self):
        return len(self.X)


app = Flask(__name__)

'''벡터화된 오디오 데이터가 포함된 JSON 페이로드로 POST 요청을 수락하는 /predict 경로를 정의합니다. 
데이터를 numpy 배열로 변환하고 PyTorch 모델의 입력 모양과 일치하도록 모양을 변경합니다. 
그런 다음 추론을 수행하고 예측된 클래스를 JSON 응답으로 반환합니다.'''


# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json['data']
#     data = np.array(data).astype(np.float32)
#     data = data.reshape(1, 1, -1)
#     data = torch.from_numpy(data)

#     with torch.no_grad():
#         output = model(data)

#     _, predicted = torch.max(output.data, 1)
#     response = {'class': predicted.item()}

#     return jsonify(response)


# # 안드로이드로 추출한 소리값 보내기
# @app.route('/', methods=['GET', 'POST'])
# def json_data():
#     data = request.preprocess_audio()
#     return jsonify(data)


# def data_slice(a, i): return a[:, 0:i] if a.shape[1] > i else np.hstack(
#     a, shape[1], np.zeros(i - a.shape[1]))


# 소리 데이터 전처리
@app.route('/upload_audio', methods=['POST'])
def preprocess_audio():
    # list = []
    # file = request.files['audio_file']
    # audio_data, _ = librosa.load(file, sr=22050)
    # print(type(audio_data))
    # print(audio_data)
    # audio_data = librosa.feature.mfcc(audio_data, n_mfcc=40, n_fft=400)
    # audio_data = data_slice(audio_data, 80)
    # delta_mfcc = librosa.feature.delta(test_mfcc_prev)
    # delta_mfcc2 = librosa.feature.delta(test_mfcc_prev, order=2)
    # audio_data = np.concatenate([audio_data, delta_mfcc, delta_mfcc2], axis=0)
    # list.append(audio_data)
    # list = np.array(list)
    # list = list.reshape(-1, list.shape[1], list.shape[2], 1)
    # data_set = Custom_Dataset(list, None)
    # data_load = DataLoader(dataset, batch_size=6, shuffle=False)

    # # cnn으로 학습된 소리데이터 파일
    # model = torch.load("cnn_best_model0.pt")
    # predict_list = prediction(model, data_lode, device)
    return {"message": "hello"}  # 이건 배열 형태임. 안드로이드 쪽에서 배열형태를 받을 수 있을지 의문. 안되면 json 형식으로 보내기


# 들어온 소리가 학습된 소리데이터 중 어떤 것과 가장 비슷한지 추론하여 결과를 반환
def prediction(model, data, device):
    predic_list = []
    model.eval()
    for wav in iter(data):
        wav = wav.to(device).float()
        logit, softmax = model(wav)
        pred = logit.argmax(dim=1, keepdim=True)
        predic_list.append(pred.tolist())
    return softmax


if __name__ == '__main__':
    app.run(host='0.0.0.0')


# 이 코드를 사용하면 벡터화된 오디오 파일을 JSON 페이로드로 /predict 경로에 보내고 JSON 형식으로 예측된 ​​클래스를 받을 수 있어야 합니다.
