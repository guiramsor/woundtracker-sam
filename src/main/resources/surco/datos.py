"""Lectura de las series de microscopia y del ground truth.

Aqui no hay analisis: se abren ficheros y se comprueba que lo que sale es lo
que declara el config. Los frames son 1-based en disco (t1..t21) y 0-based en
los arrays apilados.
"""

from pathlib import Path

import numpy as np
import tifffile

from surco import config


def ruta_imagen(serie: int, frame: int) -> Path:
    nombre = config.PATRON_IMAGEN.format(
        canal=config.CANAL, serie=serie, frame=frame
    )
    return config.DIR_IMAGENES / nombre


def ruta_gt(serie: int, frame: int) -> Path:
    return config.DIR_GT / config.PATRON_GT.format(serie=serie, frame=frame)


def proyectar_maximo(stack: np.ndarray) -> np.ndarray:
    """Maximo a lo largo de Z. El .TIF en disco es el mini Z-stack (apartado 3)."""
    if stack.ndim != 3:
        raise ValueError(f"se esperaba un stack de 3 ejes, llego {stack.shape}")
    return stack.max(axis=0)


def _comprobar(arr: np.ndarray, ruta: Path, dtype_esperado: str) -> None:
    lado = config.LADO_PX
    if arr.shape != (lado, lado):
        raise ValueError(f"{ruta.name}: forma {arr.shape}, se esperaba ({lado}, {lado})")
    if arr.dtype != np.dtype(dtype_esperado):
        raise ValueError(f"{ruta.name}: dtype {arr.dtype}, se esperaba {dtype_esperado}")


def cargar_frame(serie: int, frame: int) -> np.ndarray:
    """Un frame de microscopia, ya proyectado por el maximo sobre Z."""
    ruta = ruta_imagen(serie, frame)
    stack = tifffile.imread(ruta)
    if stack.ndim != 3 or stack.shape[0] != config.N_PLANOS_Z:
        raise ValueError(
            f"{ruta.name}: forma {stack.shape}, se esperaban "
            f"{config.N_PLANOS_Z} planos de {config.LADO_PX}x{config.LADO_PX}"
        )
    proyeccion = proyectar_maximo(stack)
    _comprobar(proyeccion, ruta, config.DTYPE_IMAGEN)
    return proyeccion


def cargar_gt(serie: int, frame: int) -> np.ndarray:
    """Una mascara de GT, etiquetada por herida."""
    ruta = ruta_gt(serie, frame)
    mascara = tifffile.imread(ruta)
    _comprobar(mascara, ruta, config.DTYPE_GT)
    return mascara


def cargar_gt_serie(serie: int) -> np.ndarray:
    """Los 21 frames de GT de una serie: (N_FRAMES, lado, lado)."""
    frames = range(1, config.N_FRAMES + 1)
    return np.stack([cargar_gt(serie, f) for f in frames])


def comprobar_imagenes() -> int:
    """Abre las 210 imagenes y devuelve cuantas pasan forma, dtype y planos.

    Lento (unos 2 GB de lectura) y por eso separado de todo lo demas: no hace
    falta para el inventario, solo confirma que la serie entera es legible.
    """
    leidas = 0
    for serie in config.SERIES:
        for frame in range(1, config.N_FRAMES + 1):
            cargar_frame(serie, frame)
            leidas += 1
    return leidas
