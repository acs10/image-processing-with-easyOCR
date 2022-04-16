# Importar Bibliotecas
import cv2
from matplotlib import pyplot as plt
import pytesseract
import functions.filtros as fu
import functions.processamento as pp
import easyocr

'''Declara objeto imagem'''
photo = 'data/teste.jpeg'

'''Lê imagem a ser analisada pelo OCR'''
img = cv2.imread(photo)

'''Criando cópia da Imagem'''
im2 = img.copy()

'''Aplicando recortes na Imagem'''
tela = im2[35:695, 30:1170] #recortando imagem da urna
tela2 = im2[95:595, 50:790] #recortando imagem da tela
tela3 = im2[125:505, 790:1170] #recortando imagem do espaço do botões

'''Processos para IMG'''
# result = pp.gradientemoforlogico(im2) # aplica um gradiente moforlogico
# result = pp.limpaImg(photo, 'data/teste2.jpeg') #limpa imagem e gera outra limpa
# y = pp.grayscale(tela2) #converte para sacala de gray
# thrs = pp.thresholding(tela2) # Aplica um thresholding
# x = pp.detectorBordasCanny(tela2) # usa o canny para detectar bordas
# z = pp.erosao(y) #Aplica erosão
# z1 = pp.abertura(z) #Aplica abertura
# z2 = pp.fechamento(z) #Aplica fechamento
# z3 = pp.dilatacao(tela2) #Aplica dilatacao

'''Processamento da IMG (Tela 2)'''
result1 = pp.grayscale(tela2) 

'''Processamento da IMG (Tela 3)'''
result2 = pp.grayscale(tela3) 
result22 = pp.erosao(result2)
result222 = pp.fechamento(result22)

'''Teste pós processamento'''
# cv2.imshow('teste', result222) #exibir img
# cv2.waitKey(0)
# frame = cv2.cvtColor(x, cv2.COLOR_BGR2RGB) #converte para RGB
# fu.contornoPalavras(frame) #Aplica contorno em palavras e exibe (necessario conversão RGB)

'''Aplica OCR na Imagem (Tela 2 processada)(Gera resultado1.Txt)'''
text = pp.applyEasyOCR(result1)
pp.generateRelatory(text,1)

'''Aplica OCR na Imagem (Tela 3 processada)(Gera resultado2.Txt)'''
text = pp.applyEasyOCR(result222)
pp.generateRelatory(text,2)

