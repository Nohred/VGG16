import os
import random
import shutil

# Ruta a la carpeta principal que contiene las subcarpetas con imágenes
carpeta_principal = 'CamVid'

# Crear una carpeta principal donde se guardarán las subcarpetas con las imágenes extraídas
carpeta_destino_principal = os.path.join(carpeta_principal, "C:/Personal Local/Recuperacion/Escuela/5to Semestre/Machine Learning/conv_imag/CamVid_bal20")

# Crear la carpeta destino principal si no existe
if not os.path.exists(carpeta_destino_principal):
    os.makedirs(carpeta_destino_principal)

# Número de imágenes aleatorias que deseas extraer de cada subcarpeta
cantidad = 20  # Cambia este número según tus necesidades

# Recorrer todas las subcarpetas y archivos dentro de la carpeta principal
for root, dirs, files in os.walk(carpeta_principal):
    # Obtener el nombre de la subcarpeta actual
    nombre_subcarpeta = os.path.basename(root)

    # Crear la subcarpeta correspondiente en la carpeta destino
    subcarpeta_destino = os.path.join(carpeta_destino_principal, nombre_subcarpeta)
    if not os.path.exists(subcarpeta_destino):
        os.makedirs(subcarpeta_destino)

    # Filtrar solo archivos de imagen
    imagenes = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    # Verificar que haya imágenes suficientes en la subcarpeta
    if len(imagenes) == 0:
        print(f"No se encontraron imágenes en la subcarpeta: {root}")
        continue

    # Si la cantidad solicitada es mayor que el número de imágenes disponibles, ajustamos la cantidad
    if cantidad > len(imagenes):
        print(f"La cantidad solicitada excede el número de imágenes en {root}. Seleccionando todas las disponibles.")
        cantidad_actual = len(imagenes)
    else:
        cantidad_actual = cantidad

    # Seleccionar aleatoriamente la cantidad especificada de imágenes
    imagenes_seleccionadas = random.sample(imagenes, cantidad_actual)

    # Copiar las imágenes seleccionadas a la nueva subcarpeta con el sufijo '_ext'
    for imagen in imagenes_seleccionadas:
        # Obtener la ruta completa de la imagen en la subcarpeta original
        ruta_imagen_origen = os.path.join(root, imagen)
        
        # Obtener el nombre del archivo y su extensión
        nombre_archivo, extension = os.path.splitext(imagen)
        
        # Crear el nuevo nombre agregando '_ext'
        nuevo_nombre = f"{nombre_archivo}_ext{extension}"
        
        # Ruta completa de la nueva ubicación de la imagen en la subcarpeta destino
        ruta_destino = os.path.join(subcarpeta_destino, nuevo_nombre)
        
        # Copiar la imagen con el nuevo nombre
        shutil.copy(ruta_imagen_origen, ruta_destino)

    # Mostrar las imágenes seleccionadas que se copiaron
    nuevas_imagenes = [os.path.join(subcarpeta_destino, f"{os.path.splitext(imagen)[0]}_ext{os.path.splitext(imagen)[1]}") for imagen in imagenes_seleccionadas]
    print(f"Se guardaron las siguientes imágenes en la subcarpeta '{subcarpeta_destino}': {nuevas_imagenes}")
