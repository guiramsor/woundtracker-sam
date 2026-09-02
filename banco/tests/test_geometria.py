"""Tests de geometria sobre figuras cuyo resultado se conoce a mano.

Para un rectangulo lleno de w columnas por h filas, el momento central de
segundo orden es (w**2 - 1) / 12 y (h**2 - 1) / 12, asi que la razon de ejes de
la elipse equivalente es sqrt(((w**2 - 1)) / (h**2 - 1)). Con w = 21 y h = 3 eso
es sqrt(440 / 8) = sqrt(55).
"""

import math

import numpy as np
import pytest

from surco import geometria


def _barra(alto: int = 3, ancho: int = 21, lado: int = 40) -> np.ndarray:
    mascara = np.zeros((lado, lado), dtype=bool)
    fila = (lado - alto) // 2
    columna = (lado - ancho) // 2
    mascara[fila : fila + alto, columna : columna + ancho] = True
    return mascara


def test_area_de_la_barra_es_el_numero_de_pixeles():
    assert geometria.medir(_barra())["area_px2"] == 63


def test_elongacion_de_la_barra_es_la_razon_conocida():
    medidas = geometria.medir(_barra(alto=3, ancho=21))
    assert medidas["elongacion"] == pytest.approx(math.sqrt(55))


def test_los_ejes_salen_de_la_elipse_equivalente():
    medidas = geometria.medir(_barra(alto=3, ancho=21))
    assert medidas["eje_mayor_px"] == pytest.approx(4 * math.sqrt(440 / 12))
    assert medidas["eje_menor_px"] == pytest.approx(4 * math.sqrt(8 / 12))


def test_el_semiancho_es_la_mitad_del_eje_menor():
    medidas = geometria.medir(_barra())
    assert medidas["semiancho_px"] == pytest.approx(medidas["eje_menor_px"] / 2)


def test_el_centroide_cae_en_el_centro_de_la_barra():
    medidas = geometria.medir(_barra(alto=3, ancho=21, lado=40))
    assert medidas["centroide_fila"] == pytest.approx(19.0)
    assert medidas["centroide_col"] == pytest.approx(19.0)


def test_la_barra_horizontal_tiene_el_eje_mayor_horizontal():
    medidas = geometria.medir(_barra())
    assert abs(math.sin(medidas["orientacion_rad"])) == pytest.approx(1.0)


def test_una_barra_entera_es_una_sola_componente():
    medidas = geometria.medir(_barra())
    assert medidas["n_comp_c4"] == 1
    assert medidas["n_comp_c8"] == 1


def test_una_barra_partida_cuenta_dos_componentes():
    mascara = _barra()
    mascara[:, 19] = False  # se abre un hueco de una columna en mitad de la barra
    medidas = geometria.medir(mascara)
    assert medidas["n_comp_c4"] == 2
    assert medidas["n_comp_c8"] == 2


def test_el_contacto_diagonal_distingue_las_dos_conectividades():
    mascara = np.zeros((10, 10), dtype=bool)
    mascara[2, 2] = True
    mascara[3, 3] = True
    medidas = geometria.medir(mascara)
    assert medidas["n_comp_c4"] == 2
    assert medidas["n_comp_c8"] == 1


def test_los_ejes_se_miden_sobre_la_lesion_entera_aunque_este_partida():
    partida = _barra()
    partida[:, 19] = False  # fuera la columna central: quedan dos mitades de 10
    medidas = geometria.medir(partida)
    assert medidas["area_px2"] == 60
    assert medidas["n_comp_c4"] == 2

    # Las 20 columnas que quedan estan a distancias 1..10 del centro, asi que el
    # momento es 2 * (1**2 + ... + 10**2) / 20 = 38.5.
    assert medidas["eje_mayor_px"] == pytest.approx(4 * math.sqrt(38.5))

    # Una sola de las dos mitades da un eje muy inferior: la prueba de que la
    # medida de arriba abarca las dos componentes y no una.
    mitad = partida.copy()
    mitad[:, 20:] = False
    assert geometria.medir(mitad)["eje_mayor_px"] == pytest.approx(
        4 * math.sqrt(99 / 12)
    )


def test_una_mascara_vacia_no_se_mide():
    with pytest.raises(ValueError, match="vacia"):
        geometria.medir(np.zeros((10, 10), dtype=bool))
