"""Los tres metodos clasicos del apartado 5.4: la hipotesis nula del trabajo.

Si uno de estos llega lo bastante cerca, embeber SAM no esta justificado, porque
un plugin de Fiji con umbral y morfologia se escribe en Java directamente, sin
SAMJ, sin Appose y sin Python.

Los tres se alimentan de **los mismos cinco clics** y del mismo frame preparado
que reciben los modelos, asi que el factor prompt del apartado 5.5 tambien les
aplica y la comparacion no cambia de entrada por el camino. Y los tres trabajan
dentro de la **ventana del clic**, de lado `config.LADO_REGION_CLIC_PX`, centrada
en el clic 0 y pegada al borde si hace falta.

Que la ventana este centrada en el clic y no en la lesion es una propiedad del
metodo, no un descuido: son metodos de clic fijo, como M1, y F2 midio que el clic
fijo se sale de la lesion en 15 de las 17 unidades. Si la ventana recorta la
lesion en los frames tardios, eso es parte de lo que se esta midiendo.

Aqui no hay modelos ni pesos: entra una imagen y salen mascaras booleanas, y
todo es determinista.
"""

import numpy as np
from skimage.filters import sato, sobel, threshold_multiotsu, threshold_otsu
from skimage.measure import label as etiquetar
from skimage.segmentation import watershed

from surco import config, geometria


def a_gris(imagen: np.ndarray) -> np.ndarray:
    """Los modelos reciben tres bandas iguales; aqui basta con una."""
    return imagen[:, :, 0] if imagen.ndim == 3 else imagen


def recortar(plano: np.ndarray, punto: tuple[int, int]):
    """La ventana del clic y su esquina, dentro de un plano del tamano del frame."""
    lado = config.LADO_REGION_CLIC_PX
    fila0, col0 = (
        int(np.clip(coordenada - lado // 2, 0, plano.shape[eje] - lado))
        for eje, coordenada in enumerate(punto)
    )
    return plano[fila0 : fila0 + lado, col0 : col0 + lado], (fila0, col0)


def clics_dentro(puntos, etiquetas, origen, lado: int) -> list:
    """Los clics que caen en la ventana, ya en coordenadas de la ventana.

    Un clic que se salga no se usa. Puede pasar con los negativos, que van en el
    nucleoplasma a un lado y otro del surco, y no es motivo para agrandar la
    ventana: agrandarla seria elegir un numero, y el numero esta derivado del GT.
    """
    fila0, col0 = origen
    dentro = []
    for (fila, columna), etiqueta in zip(puntos, etiquetas):
        local = (fila - fila0, columna - col0)
        if 0 <= local[0] < lado and 0 <= local[1] < lado:
            dentro.append((local, etiqueta))
    return dentro


def devolver(mascara_ventana: np.ndarray, origen, forma) -> np.ndarray:
    """La mascara de la ventana, colocada en el frame completo.

    La forma llega como argumento y no de `config.LADO_PX` para que los tests
    puedan correr sobre casos sinteticos pequenos sin arrastrar un frame de
    1024 x 1024 detras.
    """
    completa = np.zeros(forma, dtype=bool)
    fila0, col0 = origen
    alto, ancho = mascara_ventana.shape
    completa[fila0 : fila0 + alto, col0 : col0 + ancho] = mascara_ventana
    return completa


def componentes_de_los_clics(binaria: np.ndarray, locales: list) -> np.ndarray:
    """Componentes con algun clic positivo, menos las que tengan uno negativo.

    Es la traduccion literal del prompt de puntos a una binarizacion: los
    positivos dicen que entra y los negativos dicen que sale. Con 8 vecinos, que
    es la conectividad canonica del trabajo.
    """
    etiquetadas = etiquetar(binaria, connectivity=geometria.CONECTIVIDAD_8)
    positivas, negativas = set(), set()
    for (fila, columna), etiqueta in locales:
        valor = int(etiquetadas[fila, columna])
        if valor == 0:
            continue
        (positivas if etiqueta == 1 else negativas).add(valor)
    elegidas = sorted(positivas - negativas)
    if not elegidas:
        return np.zeros(binaria.shape, dtype=bool)
    return np.isin(etiquetadas, elegidas)


def _binarizar_por_otsu(ventana: np.ndarray) -> np.ndarray:
    """Otsu dentro de la ventana. Si es plana no hay dos clases que separar."""
    if ventana.min() == ventana.max():
        return np.zeros(ventana.shape, dtype=bool)
    return ventana > threshold_otsu(ventana)


def otsu_local(imagen, puntos, etiquetas) -> np.ndarray:
    """Umbral local tipo Otsu en la region del clic: el suelo clasico."""
    gris = a_gris(imagen)
    ventana, origen = recortar(gris, puntos[0])
    locales = clics_dentro(puntos, etiquetas, origen, ventana.shape[0])
    binaria = _binarizar_por_otsu(ventana)
    return devolver(componentes_de_los_clics(binaria, locales), origen, gris.shape)


def watershed_con_marcadores(imagen, puntos, etiquetas) -> np.ndarray:
    """Watershed con los clics como marcadores: el analogo clasico del prompt.

    El borde de la ventana entra como marcador de fondo. No es un parametro
    libre: watershed necesita al menos dos marcadores, y en P1 y P3 no hay
    ningun clic negativo que se lo de. Sin el, en P1 el unico marcador se
    quedaria con la imagen entera.
    """
    gris = a_gris(imagen)
    ventana, origen = recortar(gris, puntos[0])
    locales = clics_dentro(puntos, etiquetas, origen, ventana.shape[0])
    marcadores = np.zeros(ventana.shape, dtype=np.int32)
    marcadores[[0, -1], :] = 2
    marcadores[:, [0, -1]] = 2
    for (fila, columna), etiqueta in locales:
        marcadores[fila, columna] = 1 if etiqueta == 1 else 2
    regiones = watershed(sobel(ventana.astype(float)), marcadores)
    return devolver(regiones == 1, origen, gris.shape)


_ULTIMA_CRESTA: dict = {"gris": None, "sigmas": None, "respuesta": None}


def respuesta_de_cresta(gris: np.ndarray, sigmas=None) -> np.ndarray:
    """Sato sobre el frame **entero**, cacheado por imagen y por escalas.

    Entero y no sobre la ventana para que las escalas grandes no vean el borde
    del recorte como una cresta. Y cacheado porque la respuesta no depende del
    protocolo: los cuatro comparten centro de ventana, que es el clic 0.

    Las escalas entran como argumento para poder medir si el resultado depende
    del paso del barrido (`scripts/robustez_sigma.py`). El valor por defecto, y
    el unico que usa el pipeline, es el del config.
    """
    sigmas = tuple(sigmas) if sigmas is not None else config.SIGMAS_SATO
    anterior = _ULTIMA_CRESTA["gris"]
    if _ULTIMA_CRESTA["sigmas"] == sigmas and (
        anterior is gris
        or (
            anterior is not None
            and anterior.shape == gris.shape
            and np.array_equal(anterior, gris)
        )
    ):
        return _ULTIMA_CRESTA["respuesta"]
    _ULTIMA_CRESTA["gris"] = gris
    _ULTIMA_CRESTA["sigmas"] = sigmas
    _ULTIMA_CRESTA["respuesta"] = sato(
        gris.astype(float), sigmas=sigmas, black_ridges=False
    )
    return _ULTIMA_CRESTA["respuesta"]


def cresta_sato(imagen, puntos, etiquetas, sigmas=None) -> np.ndarray:
    """Filtro de cresta de Sato: el a priori de forma alargada.

    Es el unico de los tres que ataca la limitacion 2 del trabajo anterior, que
    las candidatas no tienen forma de surco. Sato responde a estructuras tipo
    linea, asi que produce esa forma por construccion.
    """
    respuesta = respuesta_de_cresta(a_gris(imagen), sigmas)
    ventana, origen = recortar(respuesta, puntos[0])
    locales = clics_dentro(puntos, etiquetas, origen, ventana.shape[0])
    binaria = _binarizar_por_otsu(ventana)
    return devolver(componentes_de_los_clics(binaria, locales), origen, respuesta.shape)


def candidatas_de_cresta(imagen, puntos, etiquetas, sigmas=None):
    """Tres candidatas anidadas del filtro de cresta, con su puntuacion.

    Existe para poder darle al clasico **la misma oportunidad que a los
    modelos**: SAM propone tres mascaras y un selector elige, asi que comparar
    SAM con selector contra un clasico de una sola mascara mediria dos cosas a
    la vez. Con esto, el mismo S1 se aplica a los dos lados.

    Los tres cortes salen de un Otsu **multiclase de cuatro clases** sobre la
    respuesta dentro de la ventana. No hay ningun umbral elegido: es el mismo
    Otsu que ya usaba el metodo, pedido con tres cortes en vez de uno.

    La puntuacion de cada candidata es la respuesta media de cresta dentro de
    ella, que es lo mas parecido que tiene el filtro a la confianza que publica
    SAM. Se declara porque S0 sobre estas candidatas depende de ella.
    """
    respuesta = respuesta_de_cresta(a_gris(imagen), sigmas)
    ventana, origen = recortar(respuesta, puntos[0])
    locales = clics_dentro(puntos, etiquetas, origen, ventana.shape[0])
    mascaras, puntuaciones = [], []
    for umbral in threshold_multiotsu(ventana, classes=4):
        elegida = componentes_de_los_clics(ventana > umbral, locales)
        mascaras.append(devolver(elegida, origen, respuesta.shape))
        puntuaciones.append(float(ventana[elegida].mean()) if elegida.any() else 0.0)
    return np.stack(mascaras), np.array(puntuaciones, dtype=float)


METODOS = {
    "otsu_local": otsu_local,
    "watershed": watershed_con_marcadores,
    "sato": cresta_sato,
}

if tuple(METODOS) != config.METODOS_CLASICOS:
    raise ImportError("METODOS y config.METODOS_CLASICOS se han separado")


def segmentar(metodo: str, imagen, puntos, etiquetas) -> np.ndarray:
    """Una mascara booleana del tamano del frame. Sin candidatas y sin score.

    Los clasicos no proponen varias y no puntuan, asi que el factor selector no
    les aplica: en su fila S0 y S1 son la misma cosa.
    """
    if metodo not in METODOS:
        raise KeyError(f"metodo desconocido: {metodo!r}; hay {tuple(METODOS)}")
    return METODOS[metodo](imagen, puntos, etiquetas)
