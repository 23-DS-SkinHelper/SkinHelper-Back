from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
import numpy as np
from keras.models import load_model
from io import BytesIO
from django.shortcuts import render
import os
from Skinhelper.settings import BASE_DIR
from .forms import UploadImageForm
from .models import UploadedImage
from keras import backend as K
import keras

def F1_score(y_true, y_pred):
    precision = np.sum(np.round(np.clip(y_true * y_pred, 0, 1))) / (np.sum(np.round(np.clip(y_pred, 0, 1))) + K.epsilon())
    recall = np.sum(np.round(np.clip(y_true * y_pred, 0, 1))) / (np.sum(np.round(np.clip(y_true, 0, 1))) + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val

@csrf_exempt
def predict(image):
    #keras.utils.get_custom_objects()['F1_score'] = F1_score

    model_path = os.path.join(BASE_DIR, 'efficientnet.h5')
    model = load_model(model_path, custom_objects={'F1_score': F1_score})

    image = Image.open(image).convert('RGB')
    image = image.resize((640, 310))

    # numpy 배열로 변환
    image_array = np.array(image) / 255.0

    predictions = model.predict(np.expand_dims(image_array, axis=0))

    result = {
        'predictions': predictions.tolist(),
        'label': str(np.argmax(predictions))
    }

    print(result)

    return result

def upload_image(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            # 이미지 업로드하고 DB에 저장
            uploaded_image = form.save()

            # 이미지를 predict 함수에 전달하여 결과 예측
            predicted_result = predict(uploaded_image.image.path)

            return JsonResponse({'result': predicted_result['label']})

    else:
        form = UploadImageForm()

    return render(request, 'upload_image.html', {'form': form})