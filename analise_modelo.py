import sys
import cv2
import torch
import ssl
import numpy as np

# Ignorar verificação de certificado SSL
ssl._create_default_https_context = ssl._create_unverified_context

# Carregar o modelo YOLOv5s do PyTorch Hub
print("Carregando o modelo YOLOv5s...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.1  # Reduzir a confiança mínima para exibir mais detecções

# Definir IDs para classes específicas
PERSON_CLASS_ID = 0  # Classe "person" no modelo COCO é representada pelo ID 0
ROOF_CLASS_ID = 1  # Ajuste conforme necessário, dependendo da classe do telhado no modelo

# Caminho do vídeo
video_path = '/Users/alansms/PycharmProjects/Fiap/NEXT_2024/SISTEMA DE PONTUAÇÃO/analise_video_modelo/static/IMG_2260.MOV'

# Abrir o vídeo
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Erro ao abrir o vídeo: {video_path}")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fim do vídeo ou erro na leitura do frame.")
        break

    # Converter o frame para RGB
    img = frame[..., ::-1]  # BGR para RGB

    # Realizar a detecção
    results = model(img)

    # Acessar as detecções
    detections = results.pred[0]

    # Variáveis para rastrear se o telhado e a pessoa já foram detectados
    person_detected = False
    roof_detected = False

    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)

        # Verificar a detecção de pessoa
        if int(cls) == PERSON_CLASS_ID and not person_detected:
            person_detected = True
            color = (0, 255, 0)  # Verde para "person"
            label = f'Person: {conf:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # Exibir detecção de "person" no terminal
            print(f'Detecção de Pessoa: Confiança={conf:.2f}, Coordenadas=({x1}, {y1}), ({x2}, {y2})')

        # Verificar a detecção do telhado
        elif not roof_detected:
            roof_detected = True
            color = (0, 0, 255)  # Vermelho para "telhado"
            label = f'Telhado: {conf:.2f}'  # Inclui a confiança no label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # Exibir detecção de "telhado" no terminal
            print(f'Detecção de Telhado: Confiança={conf:.2f}, Coordenadas=({x1}, {y1}), ({x2}, {y2})')

    # Exibir o frame com as detecções
    cv2.imshow('Deteccao', frame)

    # Verificar se a tecla 'q' foi pressionada para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()