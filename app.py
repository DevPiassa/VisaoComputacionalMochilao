import cv2
from deepface import DeepFace

# Carrega um detector de faces do OpenCV (mais rápido para detecção inicial)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicia a captura da webcam
cap = cv2.VideoCapture(0)

# Dicionário para traduzir as emoções para o português
traducoes_emocao = {
    'angry': 'Raiva',
    'disgust': 'Nojo',
    'fear': 'Medo',
    'happy': 'Feliz',
    'sad': 'Triste',
    'surprise': 'Surpreso',
    'neutral': 'Neutro'
}


print("Iniciando webcam... Pressione 'q' para sair.")
# A primeira execução pode demorar um pouco para baixar os modelos.

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converte o frame para escala de cinza (melhor para o detector de faces do OpenCV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta faces no frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Para cada face detectada...
    for (x, y, w, h) in faces:
        try:
            # A mágica acontece aqui: DeepFace analisa a região do rosto
            # actions=['emotion'] faz com que ele foque apenas na análise de emoção
            resultado = DeepFace.analyze(frame[y:y+h, x:x+w], actions=['emotion'], enforce_detection=False)
            
            # DeepFace retorna uma lista de dicionários, pegamos o primeiro resultado
            if isinstance(resultado, list) and len(resultado) > 0:
                emocao_en = resultado[0]['dominant_emotion']
                emocao_pt = traducoes_emocao.get(emocao_en, emocao_en) # Traduz ou usa o original

                # Desenha o retângulo e o texto da emoção no frame original
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, emocao_pt, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        except Exception as e:
            # Ignora frames onde a análise de emoção falha
            # print(f"Erro ao analisar o rosto: {e}")
            pass

    # Mostra o resultado
    cv2.imshow('Reconhecimento de Emoções', frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
