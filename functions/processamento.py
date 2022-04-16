import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import pytesseract
import easyocr

#idiomas que o easyOCR vai reconhecer
reader = easyocr.Reader(['en'])

#Aplica OCR na Imagem
def applyEasyOCR(image):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image, detail=0)

    return result

#Escreve resultados no arquivo de texto
def generateRelatory(textRead,count):
    nameOutputArchive = (r'resultado'+str(count)+'.txt') #caminho do arquivo
    outputArchive = open(nameOutputArchive, 'w')
    for i in list(textRead):
        outputArchive.write('\n'+str(i))
    outputArchive.close()

#Converte imagem para gray scale
def grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

#Aplica um Gradiente Moforlogico
def gradientemoforlogico(img):
    c = 3
    d = 3  
    kernel = np.ones((c,d),np.uint8)
    grad = cv2.morphologyEx(img,cv2.MORPH_GRADIENT, kernel)
    return grad

#Limpa a Imagem
def limpaImg(img1, novaImg):
    img = Image.open(img1)
    img = img.point(lambda x: 0 if x<100 else 255)
    img.save(novaImg)
    img2 = cv2.imread(novaImg)
    return img2

#Função para facilitar a escrita nas imagem
def escreve(img, texto, cor=(255,0,0)):
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, texto, (10,20), fonte, 0.5, cor, 0,
    cv2.LINE_AA)

''' DESCRIÇÃO DA FUNÇÃO DETECÇÃO DE BORDAS 1 (Logo a Seguir)'''
#img: Imagem de entrada
#limiar_inferior: Qualquer conjunto de pixels com gradiente de intensidade menor do que limiar_inferior não é uma borda.
#limiar_superior: Qualquer conjunto de pixels com gradiente maior do que limiar_superior é certamente uma borda.
#tamanho_kernel: É o tamanho da matriz do kernel Sobel  utilizado para o cálculo dos gradientes. O valor padrão é 3.

#Detecção de Bordas 1
def detectorBordasCanny(img):
    limiar_inferior = 30
    limiar_superior = 90
    tamanho_kernel = 5

    img1 = cv2.Canny(img, limiar_inferior, limiar_superior, tamanho_kernel)

    return img1

#Detecção de Bordas2
def detectBorde(img):
    edges = cv2.Canny(img,100,200)
    fig, ax = plt.subplots(ncols=2,figsize=(15,5))
    ax[0].imshow(img,cmap = 'gray')
    ax[0].set_title('Original Image') 
    ax[0].axis('off')
    ax[1].imshow(edges,cmap = 'gray')
    ax[1].set_title('Edge Image')
    ax[1].axis('off')
    return edges

#Aplica um thresholding na imagem
def thresholding(img):
    th1 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,27,5)
    return th1

#Aplica Erosão
def erosao(img):
    kernel = np.ones((1,1),np.uint8)
    erosion = cv2.erode(img,kernel)
    fig, ax = plt.subplots(ncols=2,figsize=(15,5))
    ax[0].imshow(img,cmap = 'gray')
    ax[0].set_title('Original') 
    plt.axis('off')
    ax[1].imshow(erosion,cmap = 'gray')
    ax[1].set_title('Eroded')
    plt.axis('off')
    return erosion

#Aplica Dilatação
def dilatacao(img):
    edges = cv2.Canny(img,100,200)
    kernel = np.ones((1,1),np.uint8)
    dilation = cv2.dilate(edges,kernel,iterations = 1)
    fig, ax = plt.subplots(ncols=2,figsize=(15,5))
    ax[0].imshow(edges,cmap = 'gray')
    ax[0].set_title('Original Edge') 
    ax[0].axis('off')
    ax[1].imshow(dilation,cmap = 'gray')
    ax[1].set_title('Dilated')
    plt.axis('off')
    return dilation

#Aplica Abertura
def abertura(img):
    kernel = np.ones((3,3),np.uint8)
    edges_3 = cv2.Canny(img,10,50)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    fig, ax = plt.subplots(ncols=2,figsize=(15,5))
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Original Image') 
    ax[0].axis('off')
    ax[1].imshow(opening,cmap = 'gray')
    ax[1].set_title('Opening Image')
    ax[1].axis('off')
    return opening

#Aplica Fechamento
def fechamento(img):
    kernel = np.ones((2,2),np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    fig, ax = plt.subplots(ncols=2,figsize=(15,5))
    ax[0].imshow(img,cmap = 'gray')
    ax[0].set_title('Image') 
    ax[0].axis('off')
    ax[1].imshow(closing,cmap = 'gray')
    ax[1].set_title('Closing Image')
    ax[1].axis('off')
    return closing
