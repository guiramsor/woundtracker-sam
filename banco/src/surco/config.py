"""Configuracion unica del banco de pruebas.

Todo numero del proyecto vive aqui; ningun otro modulo declara constantes.
Cada bloque cita el apartado del documento de diseno del que sale, o dice
que se comprobo en disco.

F0 no lleva logica: este modulo solo declara valores.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

RAIZ = Path(__file__).resolve().parents[2]

DIR_DATOS = RAIZ / "data"
DIR_IMAGENES = DIR_DATOS / "test"
DIR_GT = DIR_DATOS / "ground_truth"
# Todo lo que puso una persona a mano vive junto: los cinco clics del autor y
# las ROIs de las dos anotadoras. Es el criterio que separa lo irrecuperable de
# lo regenerable, y por eso `clics.json` no tiene carpeta propia.
DIR_ANOTACIONES = RAIZ / "anotaciones"
DIR_SALIDAS = RAIZ / "salidas"

# En la raiz de `salidas/` se queda **solo lo que se mira**: las dos tablas
# finales, su LEEME y las figuras. Todo lo demas es material intermedio y vive
# en `proceso/`, que es la mayor parte del peso y del recuento.
#
# Una sola carpeta y no una por fase: repartirlo en diez obliga a decidir a cual
# va cada fichero en cada uno de los veinte scripts que escriben, y cinco de
# ellos solo se ejercitan dentro de los entornos de modelo, asi que ese reparto
# no se podria verificar reejecutando. Con una sola no hay nada que decidir.
DIR_PROCESO = DIR_SALIDAS / "proceso"

# Las figuras de la memoria, aparte de los CSV: son ficheros que se miran, no
# que se leen, asi que se quedan arriba.
DIR_FIGURAS = DIR_SALIDAS / "figuras"

# Las candidatas de cada prompt, empaquetadas a bits: 153 ficheros y 62 MB, el
# grueso del proceso. F10 no puede existir sin ellas, asi que quien clone el
# proyecto tiene que regenerarlas con F8.
DIR_CANDIDATAS = DIR_PROCESO / "candidatas"

# El prefijo distingue de que corrida salen: vacio para la rejilla de F8, `m4_`
# para el encadenado y `suelto_` para la medida suelta de MedSAM2.
PATRON_CANDIDATAS = "candidatas_{prefijo}{nombre}_{checkpoint}_s{serie}-L{etiqueta}.npz"
# Los pesos viven FUERA del repositorio: son decenas de GB de binarios
# descargables entre los trece modelos y todos sus checkpoints, y no pertenecen
# a un arbol versionado. La variable de entorno permite moverlos sin tocar
# codigo, por ejemplo al clonar el repositorio en otra maquina.
DIR_PESOS = Path(os.environ.get("SURCO_PESOS", r"E:\pesos_sam"))

# Misma logica para los repositorios de terceros que no son instalables por pip
# y hay que clonar, como TinySAM: codigo ajeno fuera del arbol versionado.
DIR_REPOS = Path(os.environ.get("SURCO_REPOS", r"E:\repos_sam"))

# La unica entrada manual de todo el trabajo, y por eso versionada.
RUTA_CLICS = DIR_ANOTACIONES / "clics.json"

# Los dos patrones difieren en el relleno con ceros. Comprobado en disco:
# la imagen es "CTL 3_w1CSU488_s1_t1.TIF" y el GT es "serie_1/gt_t01.tif".
PATRON_IMAGEN = "CTL 3_{canal}_s{serie}_t{frame}.TIF"
PATRON_GT = "serie_{serie}/gt_t{frame:02d}.tif"

# ---------------------------------------------------------------------------
# Adquisicion (apartado 3, contrastado con data/test/CTL 3.nd)
# ---------------------------------------------------------------------------

LADO_PX = 1024
DTYPE_IMAGEN = "uint16"
CANAL = "w1CSU488"  # el .nd declara tambien w2TRANS, solo en t1, y no se usa
N_FRAMES = 21
N_PLANOS_Z = 5  # el .TIF en disco es el Z-stack; proyectar el maximo es de F1

SERIES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
SERIES_CON_GT = (1, 2, 3, 4, 5, 6, 7, 9, 10)  # falta la 8

# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------
# Las mascaras estan etiquetadas por herida: un entero distinto por lesion
# dentro de la misma mascara. La unidad de evaluacion y el sujeto estadistico
# es la pareja (serie, etiqueta), y sale de los valores unicos distintos de
# cero, no de componentes conexas.
#
# El numero de etiquetas varia entre frames de una misma serie (la serie 5
# tiene 2 heridas en t3 y 3 en t4), asi que las unidades de una serie salen de
# recorrer sus 21 frames y unir los valores, nunca de leer una sola mascara.

DTYPE_GT = "uint8"
ETIQUETA_FONDO = 0

# Fragmentacion del GT: la conectividad canonica es de 8 vecinos, y la columna
# que cuenta es `n_comp_c8`. `n_comp_c4` se conserva solo como diagnostico.
CONECTIVIDAD_FRAGMENTACION = 8
COLUMNA_FRAGMENTACION = "n_comp_c8"

# ---------------------------------------------------------------------------
# Recuentos esperados (apartado 3)
# ---------------------------------------------------------------------------
# F1 los tiene que reproducir leyendo el GT. Un frame cuenta como anotado si su
# mascara tiene algun pixel distinto de ETIQUETA_FONDO.

N_UNIDADES = 17
N_FRAMES_ANOTADOS = 163
N_PARES = 312

PARES_POR_SERIE = {1: 37, 2: 36, 3: 18, 4: 38, 5: 56, 6: 13, 7: 38, 9: 38, 10: 38}

# ---------------------------------------------------------------------------
# Geometria del surco: valores ESPERADOS, no parametros (apartado 3)
# ---------------------------------------------------------------------------
# Son estimaciones del trabajo anterior, obtenidas modelando el surco como una
# elipse a partir del area y la elongacion medias del GT. Los valores reales se
# separan bastante: en serie_5/gt_t03.tif las elongaciones medidas son 8.33 y
# 10.04 frente al 6.91 de aqui. F1 los recalcula y los reporta; ningun test
# compara contra ellos y ningun calculo los usa como umbral.
#
# No existe un SEMIANCHO_PX global, y es deliberado. F1 mide el semiancho de
# cada unidad sobre su propio GT, y F2 compara la deriva perpendicular de cada
# lesion contra el suyo (apartado 8).

AREA_MEDIA_PX2 = 1552
ELONGACION_GT = 6.91
SEMIEJE_MENOR_PX = 8.5

# ---------------------------------------------------------------------------
# Protocolo de anotacion de clics (apartados 4.1 y 4.4)
# ---------------------------------------------------------------------------
# Orden fijo: 0 centro del surco, 1 a un cuarto del eje, 2 a tres cuartos del
# eje, 3 fondo en nucleoplasma (primero el lado mas arriba en la imagen),
# 4 fondo en el lado opuesto.
#
# Se anota una sola vez, con los cinco. Los cuatro protocolos son subconjuntos
# de indices sobre esa anotacion unica: cuatro niveles sin reanotar.

N_CLICS = 5
CLICS_POSITIVOS = (0, 1, 2)
CLICS_NEGATIVOS = (3, 4)

PROTOCOLOS = {
    "P1": (0,),
    "P2": (0, 3),
    "P3": (0, 1, 2),
    "P4": (0, 1, 2, 3, 4),
}

# ---------------------------------------------------------------------------
# Factores del diseno experimental (apartado 5)
# ---------------------------------------------------------------------------
# M0 estatico (suelo), M1 clic fijo (caso base), M3 memoria de video (solo lo
# soporta SAM 2.1 de la rejilla, asi que M1 contra M3 vive dentro de ese modelo)
# y M4 encadenado de mascara, que aceptan los cuatro.
#
# M4 entra como nivel de pleno derecho: esta medido en los cuatro modelos y los
# cuatro protocolos, y es **el unico modo temporal embarcable**, porque M3 no se
# puede llevar por la ruta SAMJ (apartado 9). Que su integracion siga siendo
# condicional es otra cosa y se declara en el apartado 9.
MODOS = ("M0", "M1", "M3", "M4")

# M2, clic corregido por deriva, quedo **no implementable** y su celda sale vacia
# por dependencia no disponible (apartado 5.1). Se conserva aqui nombrado, y no
# borrado, porque el hueco es el resultado.
MODOS_CONDICIONALES = ("M2",)

# S0 es el maximo score del modelo; S1 es el criterio propio, con umbrales
# derivados dejando una serie fuera.
SELECTORES = ("S0", "S1")

# Los k = 4 que pasaron el cribado, uno por familia, cada uno con el checkpoint
# que lo representa (apartado 5.3). Vive aqui y no en cada script **porque es la
# unica constante del trabajo que puede fallar en silencio**: si un
# representante cambiara y solo se corrigiera en algunos sitios, los scripts
# medirian modelos distintos sin que nada reventara.
REJILLA_PROFUNDIDAD = {
    "microsam_lm": "vit_b",
    "sammed2d": "sam_med2d",
    "tinysam": "vit_t",
    "sam21": "tiny",
}

# La mejor celda del trabajo, la que se dibuja en los overlays y con la que se
# compara el techo humano del apartado 7. **Sale medida, no elegida**, y por eso
# `verificacion` comprueba que sigue siendo la mejor en vez de solo que sea una
# celda valida: antes decia M1 x P4 x S1 y se quedo vieja cuando M4 subio el
# techo a 0.5517, sin que nada avisara.
MEJOR_CELDA = {
    "modelo": "microsam_lm",
    "checkpoint": "vit_b",
    "modo": "M4",
    "protocolo": "P4",
    "selector": "S1",
}

# Familias del apartado 2, con la biomedica subdividida segun el apartado 5.2.
# La subdivision no es organizativa: son tres niveles de ajuste de dominio
# (ninguno, modalidad equivocada, modalidad correcta con objeto equivocado) y esa
# gradacion es la hipotesis.
FAMILIAS = ("linaje", "eficiencia", "microscopia", "radiologia", "clasico")

# Conversion de la microscopia uint16 a las tres bandas uint8 que esperan los
# modelos. Vive aqui, y no en cada adaptador, porque tiene que ser identica para
# todos: si cada modelo recibiera una imagen preparada de otra forma, la
# comparativa dejaria de comparar modelos y pasaria a comparar preprocesados.
#
# Confirmado por el autor. Lo que estuvo abierto fue el ambito, no el valor, y
# se cerro al pasarlo de por imagen a por serie: normalizar cada frame por su
# cuenta borraba la diferencia de dificultad entre frames debiles y brillantes,
# que es lo que el modelo tiene que enfrentar.
PERCENTIL_BLANCO_MODELOS = 99.9

# ---------------------------------------------------------------------------
# Vision clasica (apartado 5.4)
# ---------------------------------------------------------------------------
# El apartado deja abiertas dos elecciones de metodo y no fija ningun numero.
# Las dos elecciones las cerro el autor por el mismo criterio, que es no meter
# parametros que habria que calibrar:
#
#   - de "watershed o random walker", **watershed con marcadores**: toma los
#     clics positivos y negativos como marcadores sin nada que ajustar, mientras
#     que random walker necesita un beta. Ademas es nativo en Fiji.
#   - de "Frangi o Sato", **Sato**: esta pensado para estructuras tipo linea,
#     mientras que Frangi generaliza a blobs y tubos con dos parametros mas.
#
# **Y los dos numeros salen de sitios distintos**, que es lo que el 5.4 titula
# "de donde sale cada uno". Lo que comparten, y es lo que importa, es que
# **ninguno se calibro contra los resultados**; de ahi no se sigue que ninguno de
# los dos este derivado del GT, y decirlo asi fue un error que estuvo escrito
# aqui: la ventana sale de una longitud **estimada** y el rango de escalas **se
# eligio**. El unico de los tres valores que esta medido es el paso.

# Lado de la ventana cuadrada centrada en el clic dentro de la que trabajan los
# tres metodos. 128 px es del orden de la longitud del surco del apartado 3, que
# es una **estimacion del trabajo anterior** y no una medida del GT: modelando la
# lesion como elipse salia un semieje mayor de 58 px, o sea unos 116 px de punta
# a punta. F1 midio despues que los valores reales se separan bastante de esa
# elipse, y aun asi la ventana se queda: lo que hace falta es que sea del orden
# del surco, no que lo ajuste.
LADO_REGION_CLIC_PX = 128

# Rango de escalas del filtro de cresta, en pixeles. **Elegido, no derivado**: se
# escoge para cubrir los semianchos que midio F1, de 5.50 a 21.28 px, acotado por
# abajo para que el filtro no responda al ruido y por arriba para que no responda
# al nucleo. No hay ninguna relacion escrita entre sigma y el semiancho.
#
# **El rango y el paso se declaran por separado, y eso importa.** Antes solo
# existia la tupla ya construida, y `scripts/robustez_sigma.py` sacaba de ella
# los extremos del barrido con `SIGMAS_SATO[0]` y `SIGMAS_SATO[-1]`. Al adoptar
# el paso 4 la tupla paso a acabar en 10, asi que el script se puso a barrer de
# 2 a 10 en vez de 2 a 11: media su propia conclusion. Con el rango aparte, el
# barrido no depende del paso que se haya elegido.
RANGO_SIGMAS_SATO = (2, 11)
#
# **El paso es 4, y esta medido, no elegido.** Diez escalas en paso 1 cubren el
# rango entero y podrian estar ganando por fuerza bruta, asi que se repitio la
# mejor celda del filtro, Sato x P3, con los pasos del 1 al 6
# (`scripts/robustez_sigma.py`, salidas/robustez_sigma.csv). Frente al paso 1,
# pareado por unidad:
#
#   paso 2, cinco escalas:  0.4084, -0.0059 [-0.0195, +0.0069]   aguanta
#   paso 3, cuatro escalas: 0.4233, +0.0090 [+0.0017, +0.0187]   aguanta
#   paso 4, tres escalas:   0.4165, +0.0022 [-0.0087, +0.0139]   aguanta
#   paso 5, dos escalas:    0.3764, -0.0378 [-0.0642, -0.0141]   rompe
#   paso 6, dos escalas:    0.3884, -0.0259 [-0.0466, -0.0064]   rompe
#
# O sea que el resultado no viene del barrido fino: aguanta hasta tres escalas y
# rompe en dos, con los dos pasos de dos escalas cayendo con el intervalo entero
# por debajo del cero. Se queda el mas grosero que lo mantiene, tres escalas en
# vez de diez, que es menos computo dentro de Fiji.
PASO_SIGMAS_SATO = 4
SIGMAS_SATO = tuple(
    range(RANGO_SIGMAS_SATO[0], RANGO_SIGMAS_SATO[1] + 1, PASO_SIGMAS_SATO)
)

# Los tres, con la familia que les toca en la tabla del apartado 2.
METODOS_CLASICOS = ("otsu_local", "watershed", "sato")

# ---------------------------------------------------------------------------
# Metricas (apartados 6.1 a 6.5)
# ---------------------------------------------------------------------------
# Los catastroficos dejan de ser una metrica aparte: son el punto t = 0.1 de la
# curva de umbrales. Este 0.1 si sale de la base.
UMBRAL_CATASTROFICO = 0.1

# El apartado 6.4 pide "intervalo de confianza" remuestreando unidades por
# bootstrap, pero no fija nivel, numero de remuestreos ni semilla. Los tres
# valores son convenciones, no salen de la base, y estan aqui y no escondidos en
# el codigo para que se vean. Confirmados por el autor el 2026-08-02. La semilla
# es fija porque el apartado 10 dice que el pipeline es determinista y que no se
# promedian ejecuciones.
NIVEL_CONFIANZA = 0.95
N_REMUESTREOS = 10_000
SEMILLA_BOOTSTRAP = 20260802

# ---------------------------------------------------------------------------
# Los cinco huecos, todos cerrados
# ---------------------------------------------------------------------------
# Donde vive cada uno ahora:
#
# 1. La lista de modelos. Congelada en **once candidatos** (apartado 5.2). Eran
#    trece y al instalarlos cayeron SAM 3 y FastSAM, los dos con exclusion
#    declarada y verificacion reproducible. No hay constante: la lista es el
#    registro de adaptadores, que se declara solo.
# 2. Cuantos modelos biomedicos promptables por punto existen: **cuatro**
#    (micro-sam LM, micro-sam EM, SAM-Med2D y MedSAM2). Era el numero del que
#    dependia el enunciado de la hipotesis.
# 3. La medida del apartado 8. Hecha en F2: el clic fijo se sale de la lesion en
#    15 de las 17 unidades, asi que M2 esta justificado. Y F6 midio que la senal
#    para implementarlo no existe, asi que M2 queda **declarado no
#    implementable** (apartado 5.1). Por eso no hay nivel M2 en MODOS.
# 4. Los umbrales del selector S1, cerrado **por una via distinta a la
#    prevista: no hay umbrales**. FRAC_MAX = 0.55 era el unico parametro libre
#    del criterio y era el de la guarda anti-nucleo, que cae porque el nucleo no
#    es segmentable en w1CSU488 (apartado 5.5). El S1 reformulado -contener el
#    clic 0, no contener negativos, ordenar por z, desempatar por elongacion- no
#    tiene nada que calibrar, asi que no hay constante que declarar aqui y las
#    rondas de dejar una serie fuera no calibran: comprueban consistencia.
# 5. El k del cribado: **k = 4**, uno por familia (apartado 5.3).
#
# Los cinco quedan cerrados. Ninguno lleva valor de relleno, ni siquiera None:
# una constante ausente rompe el import y se ve; un None viaja hasta el fondo
# del pipeline y no se nota.
#
# Tampoco hay factor de escala del apartado 9, y no por falta de a que
# aplicarselo: quedan dos parametros en pixeles, LADO_REGION_CLIC_PX y
# RANGO_SIGMAS_SATO. Es que no se reescalan multiplicando por un factor, se
# rehacen mirando la geometria de los datos nuevos (apartado 9).
