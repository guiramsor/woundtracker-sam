"""Inventario del ground truth.

Enumera las unidades (serie, etiqueta), fija el `start` de cada una y mide cada
par (unidad, frame). Las funciones que hacen el trabajo reciben arrays, no
rutas, para poder probarlas sobre casos sinteticos.
"""

import numpy as np
import pandas as pd

from surco import config, datos, geometria


def etiquetas_de_serie(gt_serie: np.ndarray) -> tuple[int, ...]:
    """Union de los valores distintos de fondo sobre los 21 frames.

    Se unen los frames y no se lee uno solo porque el numero de etiquetas varia
    dentro de la misma serie: la 5 tiene 2 heridas en t3 y 3 en t4.
    """
    valores = np.unique(gt_serie)
    return tuple(int(v) for v in valores if v != config.ETIQUETA_FONDO)


def frames_de_unidad(gt_serie: np.ndarray, etiqueta: int) -> tuple[int, ...]:
    """Frames 1-based en los que aparece esa etiqueta."""
    presente = (gt_serie == etiqueta).any(axis=(1, 2))
    return tuple(int(i) + 1 for i in np.flatnonzero(presente))


def frames_anotados(gt_serie: np.ndarray) -> tuple[int, ...]:
    """Frames 1-based con algun pixel distinto del fondo, sea de la herida que sea."""
    anotado = (gt_serie != config.ETIQUETA_FONDO).any(axis=(1, 2))
    return tuple(int(i) + 1 for i in np.flatnonzero(anotado))


def medir_unidad(gt_serie: np.ndarray, serie: int, etiqueta: int) -> list[dict]:
    filas = []
    for frame in frames_de_unidad(gt_serie, etiqueta):
        mascara = gt_serie[frame - 1] == etiqueta
        fila = {"serie": serie, "etiqueta": etiqueta, "frame": frame}
        fila.update(geometria.medir(mascara))
        filas.append(fila)
    return filas


def inventariar_serie(serie: int, gt_serie: np.ndarray) -> pd.DataFrame:
    """Una fila por par (unidad, frame) de esa serie."""
    filas: list[dict] = []
    for etiqueta in etiquetas_de_serie(gt_serie):
        filas.extend(medir_unidad(gt_serie, serie, etiqueta))
    return pd.DataFrame(filas)


def construir_inventario() -> pd.DataFrame:
    """El inventario de las nueve series con GT, leido de disco."""
    trozos = [
        inventariar_serie(serie, datos.cargar_gt_serie(serie))
        for serie in config.SERIES_CON_GT
    ]
    return pd.concat(trozos, ignore_index=True)


def _salto_centroide_maximo(grupo: pd.DataFrame) -> float:
    """Mayor desplazamiento del centroide entre dos frames consecutivos.

    Es el diagnostico de estabilidad de etiquetas: las lesiones estan quietas,
    asi que un salto de cientos de pixeles seria la firma de una renumeracion
    (una herida nueva que aparece a la izquierda se lleva el 1 y desplaza al
    resto). No se compara contra ningun umbral: se reporta el numero.
    """
    if len(grupo) < 2:
        return 0.0
    ordenado = grupo.sort_values("frame")
    saltos = np.hypot(
        np.diff(ordenado["centroide_fila"].to_numpy()),
        np.diff(ordenado["centroide_col"].to_numpy()),
    )
    return float(saltos.max())


def tabla_unidades(inventario: pd.DataFrame) -> pd.DataFrame:
    """Una fila por unidad: start, ultimo frame, numero de pares y diagnostico."""
    filas = []
    for (serie, etiqueta), grupo in inventario.groupby(["serie", "etiqueta"]):
        frames = sorted(grupo["frame"])
        filas.append(
            {
                "serie": int(serie),
                "etiqueta": int(etiqueta),
                "start": frames[0],
                "ultimo": frames[-1],
                "n_pares": len(frames),
                "contiguo": frames == list(range(frames[0], frames[-1] + 1)),
                "salto_centroide_max_px": _salto_centroide_maximo(grupo),
                "centroide_col_media": float(grupo["centroide_col"].mean()),
            }
        )
    return pd.DataFrame(filas)


def _hay_tardia_a_la_izquierda(starts: np.ndarray, columnas: np.ndarray) -> bool:
    """Si alguna unidad arranca despues que otra y cae a su izquierda."""
    return any(
        starts[j] > starts[i] and columnas[j] < columnas[i]
        for i in range(len(starts))
        for j in range(len(starts))
    )


def orden_de_etiquetas(unidades: pd.DataFrame) -> pd.DataFrame:
    """Comprueba serie a serie la regla de numeracion del GT.

    Las lesiones se numeran de izquierda a derecha. Lo que romperia la regla es
    que apareciera una herida nueva a la izquierda de las existentes: se
    llevaria el 1 y desplazaria a todas las demas. Se comprueba sin umbral, que
    es lo bueno de esta via: basta ver si el orden de las etiquetas coincide con
    el orden por columna, y si alguna unidad que arranca mas tarde cae a la
    izquierda de otra que ya estaba.
    """
    filas = []
    for serie, grupo in unidades.groupby("serie"):
        ordenado = grupo.sort_values("etiqueta")
        columnas = ordenado["centroide_col_media"].to_numpy()
        starts = ordenado["start"].to_numpy()
        filas.append(
            {
                "serie": int(serie),
                "n_unidades": len(ordenado),
                "etiquetas_ordenadas_por_columna": bool(np.all(np.diff(columnas) > 0)),
                "tardia_a_la_izquierda": _hay_tardia_a_la_izquierda(starts, columnas),
                "salto_centroide_max_px": float(ordenado["salto_centroide_max_px"].max()),
            }
        )
    return pd.DataFrame(filas)


def duplicados_de_serie(serie: int, gt_serie: np.ndarray) -> list[dict]:
    """Frames consecutivos cuya mascara es identica pixel a pixel.

    La ROI se duplico en vez de redibujarse, asi que las dos anotaciones no son
    independientes y el par sobra en la muestra de concordancia (apartado 7).
    """
    filas = []
    for etiqueta in etiquetas_de_serie(gt_serie):
        frames = frames_de_unidad(gt_serie, etiqueta)
        for anterior, siguiente in zip(frames, frames[1:]):
            if siguiente != anterior + 1:
                continue
            iguales = np.array_equal(
                gt_serie[anterior - 1] == etiqueta, gt_serie[siguiente - 1] == etiqueta
            )
            if iguales:
                filas.append(
                    {
                        "serie": serie,
                        "etiqueta": etiqueta,
                        "frame_a": anterior,
                        "frame_b": siguiente,
                    }
                )
    return filas


def fragmentacion_en_start(inventario_df: pd.DataFrame) -> pd.DataFrame:
    """El estado del GT de cada unidad en su propio frame `start`.

    Importa para F3: `start` es el frame sobre el que se anotan los cinco clics,
    y si la lesion ya esta partida ahi, los clics **1 y 2** (`un cuarto` y `tres
    cuartos`, numerados desde 0 como en `clics.json` y en `config.PROTOCOLOS`)
    pueden caer en hueco oscuro y entra la clausula de excepcion del apartado 4.2.
    """
    primeros = inventario_df.loc[
        inventario_df.groupby(["serie", "etiqueta"])["frame"].idxmin()
    ]
    columnas = [
        "serie",
        "etiqueta",
        "frame",
        config.COLUMNA_FRAGMENTACION,
        "n_comp_c4",
        "area_px2",
        "elongacion",
        "semiancho_px",
    ]
    tabla = primeros[columnas].rename(columns={"frame": "start"}).reset_index(drop=True)
    tabla["partida_en_start"] = tabla[config.COLUMNA_FRAGMENTACION] > 1
    return tabla


def duplicados_de_todas_las_series() -> pd.DataFrame:
    """Todos los pares de frames con mascara identica, leyendo de disco."""
    filas: list[dict] = []
    for serie in config.SERIES_CON_GT:
        filas.extend(duplicados_de_serie(serie, datos.cargar_gt_serie(serie)))
    return pd.DataFrame(filas, columns=["serie", "etiqueta", "frame_a", "frame_b"])


def recuento_frames_anotados() -> pd.DataFrame:
    """Frames con anotacion por serie, leyendo de disco."""
    filas = []
    for serie in config.SERIES_CON_GT:
        anotados = frames_anotados(datos.cargar_gt_serie(serie))
        filas.append(
            {"serie": serie, "n_frames_anotados": len(anotados), "frames": anotados}
        )
    return pd.DataFrame(filas)
