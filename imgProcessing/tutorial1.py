import cv2
import numpy as np
from pytesseract import pytesseract
from pprint import pprint
#reading the image
img = cv2.imread('assets/p45_47.jpg',-1)

#img = cv2.resize(img, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)
#writing a modified image
#cv2.imwrite('new_img.jpg',img)

#print the shape of image
#print(img.shape)
 
###### ALGORITMO PARA LIMPIAR LAS IMAGENES DE TABLAS CON PIXELES DAÑADOS

### Nota: puede que no sea necesario utilizar este algoritmo
#blurring for sharpening
gaussian_blur= cv2.GaussianBlur(img, (7,7),2)

#sharpening using addWeighted
def call_steiner(grimmer):
    awaken = cv2.resize(grimmer, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)
    drawn = cv2.cvtColor(awaken,cv2.COLOR_RGB2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    grayscale_dilated = cv2.dilate(drawn, kernel, iterations=1)
    grayscale_eroded = cv2.erode(grayscale_dilated, kernel, iterations=1)

    grayscale_blurred = cv2.threshold(cv2.bilateralFilter(grayscale_eroded, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return grayscale_blurred

output = cv2.addWeighted(img,1.5, gaussian_blur, -0.5, 0)

grayscale = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
thresh, black = cv2.threshold(grayscale,220,255,cv2.THRESH_BINARY)

kernel = np.ones((1, 1), np.uint8)
grayscale_dilated = cv2.dilate(grayscale, kernel, iterations=1)
grayscale_eroded = cv2.erode(grayscale_dilated, kernel, iterations=1)

grayscale_blurred = cv2.threshold(cv2.bilateralFilter(grayscale_eroded, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#show an image in a window
#print(black)
# cont=0

###### QUITAR LO QUE ESTÁ AFUERA DEL BORDE DE LA TABLA
# quitamos las lineas de la imagen que no pertenecen a la tabla

top_edge = 0
bottom_edge = 0
right_edge = 0
left_edge = 0

line_len = len(black[0])

for i in range(len(black)):    
    white_count = len([pixel for pixel in black[i] if pixel==255])
    if white_count/line_len < 0.2:
        for j in range(line_len):
            if black[i][j]!=255:
                left_edge = j+1
                break
        top_edge = i
        break
for i in range(len(black)-1,0,-1):

    white_count = len([pixel for pixel in black[i] if pixel==255])
    if white_count/line_len < 0.2:
        for j in range(line_len-1,0,-1):
            if black[i][j]!=255:
                right_edge = j
                break
        bottom_edge = i
        break

print('top_edge: ',top_edge)
print('bottom_edge: ',bottom_edge)
print('left_edge',left_edge)
print('right_edge',right_edge)

#cv2.imwrite('new_img.jpg',img[top_edge:bottom_edge,left_edge:right_edge])
cropped_img = img[top_edge:bottom_edge,left_edge:right_edge]
cropped_black = black[top_edge:bottom_edge,left_edge:right_edge]
cropped_grayscale = grayscale[top_edge:bottom_edge,left_edge:right_edge]
thresh, black2 = cv2.threshold(grayscale,170,255,cv2.THRESH_BINARY)
cropped_black2 = black2[top_edge:bottom_edge,left_edge:right_edge]
cropped_sharp = output[top_edge:bottom_edge,left_edge:right_edge]
cropped_blur = gaussian_blur[top_edge:bottom_edge,left_edge:right_edge]
cropped_cleaned = grayscale_eroded[top_edge:bottom_edge,left_edge:right_edge]
###### OBTENER CELDAS
# obtener filas
rows = []
row_start = 0
for i in range(1,len(cropped_black)):
    if cropped_black[i][2] != cropped_black[i-1][2]:
        rows.append((row_start,i))
        row_start = i
print("cantidad de filas",len(rows))


### obtener columnas
# quitar supercolumnas
header = cropped_black[rows[0][0]:rows[0][1]]
supercol_start = False

cut_supercol= 0
for i in range(len(header)):
    white_count = len([pixel for pixel in header[i] if pixel==255])  
    if white_count == 0 and supercol_start:
        cut_supercol = i
        break
    if white_count >=5 and not supercol_start:
        supercol_start= True



print("corte supercolumnas", cut_supercol)
rows[0] = (cut_supercol,rows[0][1])
# obtener corte de las columnas
cols = []
col_start = 0
line_len = len(cropped_black[0]) 


for i in range(4,line_len):
    if cropped_black[rows[1][0]+1][i]==255 and cropped_black[rows[1][0]+1][i-1]== 0:
        cols.append((col_start,i))
        col_start = i
cols.append((col_start,i))


print("cantidad columna",len(cols))

# for col in cols:
#     cv2.imshow(f"col{col}",cropped_img[0:,col[0]:col[1]])

# for row in rows:
#     cv2.imshow(f"col{row}",cropped_img[row[0]:row[1]])

# ALGORITMO PARA SACAR EL TEXTO DE LAS DISTINTAS CELDAS
partial_csv=[]
for row in rows[0:3]:
    csv_row = []
    for col in cols:
        victim_part = cropped_img[row[0]:row[1],col[0]:col[1]] 
        text = pytesseract.image_to_string(
            call_steiner(victim_part),lang="spa")
        csv_row.append(text.strip())
    partial_csv.append(csv_row)

pprint(partial_csv)
# ALGORITMO PARA AGRUPAR EL TEXTO ENCONTRADO, EN UN CSV
row_index = 1
col_index = 0
#cv2.imshow("black",cropped_sharp)
cv2.waitKey(0)
cv2.destroyAllWindows()

##necesitamos cortar por otro lado para sacar las filas correctamente
##podemos revisar la posición final de la segunda columna para cortar desde ahi

#para cortar la supercolumna, utilizar imagen con grises y quitar un gris en especifico