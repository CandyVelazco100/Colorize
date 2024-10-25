import numpy as np
import cv2
from cv2 import dnn

#-------- Configuración de rutas --------#
PROTO_FILE = './model/colorization_deploy_v2.prototxt'
MODEL_FILE = './model/colorization_release_v2.caffemodel'
HULL_PTS = './model/pts_in_hull.npy'
IMG_PATH = './images/girl.jpg'
RESULT_PATH = './results/colorized_final.png'
IMG_SIZE = (640, 640)

#-------- Funciones Auxiliares --------#
def cargar_modelo(proto_file, model_file, hull_pts):
    net = dnn.readNetFromCaffe(proto_file, model_file)
    kernel = np.load(hull_pts)
    return net, kernel

def preparar_imagen(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"No se encontró la imagen: {img_path}")
    scaled = img.astype("float32") / 255.0
    lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    return img, lab_img

def configurar_red(net, kernel):
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = kernel.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

def aplicar_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.merge((l, a, b))

def ajustar_gamma(img, gamma=1.2):  # Ajuste de gamma
    """Aplica corrección Gamma para mejorar el brillo."""
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def colorizar_imagen(net, lab_img, original_shape):
    resized = cv2.resize(lab_img, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab_channel = cv2.resize(ab_channel, original_shape[:2][::-1])

    L_original = cv2.split(lab_img)[0]
    colorized = np.concatenate((L_original[:, :, np.newaxis], ab_channel), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)

    # Aplicar suavizado bilateral
    colorized = cv2.bilateralFilter(colorized, d=9, sigmaColor=75, sigmaSpace=75)

    # Ajustar saturación
    hsv = cv2.cvtColor(colorized, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.5, 0, 255).astype(np.uint8)  # Aumentar saturación
    hsv[..., 2] = np.clip(hsv[..., 2] * 1.1, 0, 255).astype(np.uint8)  # Aumentar brillo
    colorized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return np.clip(colorized, 0, 1)

def ajustar_balance_blancos(img):
    """Balance de blancos más preciso para reducir dominancia verde."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    a_mean, b_mean = np.mean(a), np.mean(b)

    # Reducción del canal verde y ajuste cálido
    a = np.clip(a.astype(np.float32) - (a_mean - 128) * 0.5, 0, 255).astype(np.uint8)  # Reducción leve
    b = np.clip(b.astype(np.float32) + (140 - b_mean) * 0.5, 0, 255).astype(np.uint8)  # Ajuste leve

    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def mostrar_guardar_resultado(img, colorized, result_path):
    img = cv2.resize(img, IMG_SIZE)
    colorized = cv2.resize((255 * colorized).astype("uint8"), IMG_SIZE)

    # Aplicar CLAHE, balance de blancos y corrección Gamma
    colorized = aplicar_clahe(colorized)
    colorized = ajustar_balance_blancos(colorized)
    colorized = ajustar_gamma(colorized, gamma=1.1)

    # Mezcla ponderada de la imagen original y la colorizada
    blended = cv2.addWeighted(img, 0.6, colorized, 0.4, 0)

    resultado = cv2.hconcat([img, blended])
    cv2.imshow("Grayscale -> Colour", resultado)
    cv2.imwrite(result_path, blended)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#-------- Flujo Principal --------#
if __name__ == "__main__":
    net, kernel = cargar_modelo(PROTO_FILE, MODEL_FILE, HULL_PTS)
    img, lab_img = preparar_imagen(IMG_PATH)
    configurar_red(net, kernel)
    colorized = colorizar_imagen(net, lab_img, img.shape)
    mostrar_guardar_resultado(img, colorized, RESULT_PATH)
