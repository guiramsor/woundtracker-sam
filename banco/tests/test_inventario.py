"""Tests del inventario sobre una serie sintetica de recuentos conocidos.

La serie imita lo que hace la de verdad: dos heridas que arrancan en frames
distintos (3 y 4), anchos que cambian frame a frame, y un duplicado deliberado
en el que la ROI de t9 es copia exacta de la de t8.
"""

import numpy as np
import pytest

from surco import config, inventario


@pytest.fixture
def gt_serie() -> np.ndarray:
    gt = np.zeros((config.N_FRAMES, 30, 30), dtype=np.uint8)
    for frame in range(3, config.N_FRAMES + 1):
        gt[frame - 1, 10:13, 2 : 2 + 5 + frame % 4] = 1
    for frame in range(4, config.N_FRAMES + 1):
        gt[frame - 1, 20:23, 18 : 18 + 4 + frame % 3] = 2
    # t9 copia de t8 para la etiqueta 1, sin tocar la etiqueta 2.
    gt[8][gt[8] == 1] = config.ETIQUETA_FONDO
    gt[8][gt[7] == 1] = 1
    return gt


def test_las_etiquetas_salen_de_unir_los_frames_no_de_leer_uno(gt_serie):
    # En t3 solo hay una herida; la segunda no aparece hasta t4.
    assert inventario.etiquetas_de_serie(gt_serie[2:3]) == (1,)
    assert inventario.etiquetas_de_serie(gt_serie) == (1, 2)


def test_el_fondo_no_es_una_etiqueta(gt_serie):
    assert config.ETIQUETA_FONDO not in inventario.etiquetas_de_serie(gt_serie)


def test_cada_unidad_tiene_sus_frames(gt_serie):
    assert inventario.frames_de_unidad(gt_serie, 1) == tuple(range(3, 22))
    assert inventario.frames_de_unidad(gt_serie, 2) == tuple(range(4, 22))


def test_un_frame_esta_anotado_si_tiene_algun_pixel_de_cualquier_herida(gt_serie):
    assert inventario.frames_anotados(gt_serie) == tuple(range(3, 22))
    assert len(inventario.frames_anotados(gt_serie)) == 19


def test_el_start_es_por_unidad_y_no_por_serie(gt_serie):
    unidades = inventario.tabla_unidades(inventario.inventariar_serie(5, gt_serie))
    starts = dict(zip(unidades["etiqueta"], unidades["start"]))
    assert starts == {1: 3, 2: 4}


def test_los_pares_de_la_serie_son_la_suma_de_los_de_cada_unidad(gt_serie):
    unidades = inventario.tabla_unidades(inventario.inventariar_serie(5, gt_serie))
    assert list(unidades["n_pares"]) == [19, 18]
    assert unidades["n_pares"].sum() == 37


def test_las_unidades_sin_huecos_se_marcan_contiguas(gt_serie):
    unidades = inventario.tabla_unidades(inventario.inventariar_serie(5, gt_serie))
    assert unidades["contiguo"].all()


def test_un_hueco_en_medio_rompe_la_contiguidad(gt_serie):
    gt_serie[10][gt_serie[10] == 1] = config.ETIQUETA_FONDO  # se borra la 1 en t11
    unidades = inventario.tabla_unidades(inventario.inventariar_serie(5, gt_serie))
    contiguo = dict(zip(unidades["etiqueta"], unidades["contiguo"]))
    assert contiguo[1] is np.False_ or contiguo[1] is False
    assert contiguo[2]


def test_se_ve_si_una_unidad_arranca_ya_partida(gt_serie):
    # Se abre un hueco en la barra de la etiqueta 1 en su frame start, el t3.
    gt_serie[2, 10:13, 5] = config.ETIQUETA_FONDO
    tabla = inventario.fragmentacion_en_start(inventario.inventariar_serie(5, gt_serie))
    partida = dict(zip(tabla["etiqueta"], tabla["partida_en_start"]))
    assert partida[1]
    assert not partida[2]
    assert list(tabla["start"]) == [3, 4]


def test_se_detecta_la_mascara_duplicada_y_solo_esa(gt_serie):
    duplicados = inventario.duplicados_de_serie(5, gt_serie)
    assert duplicados == [{"serie": 5, "etiqueta": 1, "frame_a": 8, "frame_b": 9}]


def test_el_inventario_tiene_una_fila_por_par(gt_serie):
    tabla = inventario.inventariar_serie(5, gt_serie)
    assert len(tabla) == 37
    assert set(tabla["serie"]) == {5}
    assert tabla["area_px2"].min() > 0


def test_lesiones_quietas_dan_saltos_de_centroide_pequenos(gt_serie):
    unidades = inventario.tabla_unidades(inventario.inventariar_serie(5, gt_serie))
    # Las barras solo cambian de ancho, asi que el centroide se mueve poco.
    assert unidades["salto_centroide_max_px"].max() < 2.0


def test_la_regla_de_numeracion_se_cumple_en_la_serie_sintetica(gt_serie):
    unidades = inventario.tabla_unidades(inventario.inventariar_serie(5, gt_serie))
    orden = inventario.orden_de_etiquetas(unidades).iloc[0]
    assert orden["etiquetas_ordenadas_por_columna"]
    assert not orden["tardia_a_la_izquierda"]


def test_una_herida_nueva_por_la_izquierda_rompe_la_regla(gt_serie):
    # La etiqueta 2 pasa a ser la de la izquierda y a arrancar mas tarde: es
    # exactamente el caso que renumeraria a todas las demas.
    gt_serie[gt_serie == 2] = config.ETIQUETA_FONDO
    for frame in range(6, config.N_FRAMES + 1):
        gt_serie[frame - 1, 20:23, 0:5] = 2
    unidades = inventario.tabla_unidades(inventario.inventariar_serie(5, gt_serie))
    orden = inventario.orden_de_etiquetas(unidades).iloc[0]
    assert not orden["etiquetas_ordenadas_por_columna"]
    assert orden["tardia_a_la_izquierda"]


def test_una_renumeracion_se_delata_como_salto_grande(gt_serie):
    # Se simula que la etiqueta 1 pasa a nombrar la herida de la derecha en t12.
    gt_serie[11][gt_serie[11] == 1] = config.ETIQUETA_FONDO
    gt_serie[11][gt_serie[11] == 2] = 1
    unidades = inventario.tabla_unidades(inventario.inventariar_serie(5, gt_serie))
    salto = dict(zip(unidades["etiqueta"], unidades["salto_centroide_max_px"]))
    assert salto[1] > 10.0
