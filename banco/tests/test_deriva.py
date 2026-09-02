"""Tests de la deriva sobre lesiones sinteticas que se mueven lo que se les dice.

La lesion de referencia es una barra horizontal de 3 filas por 21 columnas: su
eje mayor va por columnas, su eje menor por filas, y su semiancho es
4 * sqrt(8 / 12) / 2 = 1.632993..., el mismo valor que comprueba test_geometria.
"""

import math

import numpy as np
import pytest

from surco import config, deriva


SEMIANCHO_BARRA = 4 * math.sqrt(8 / 12) / 2


def _barra(lado: int = 60, fila: int = 20, columna: int = 20) -> np.ndarray:
    mascara = np.zeros((lado, lado), dtype=bool)
    mascara[fila : fila + 3, columna : columna + 21] = True
    return mascara


def _serie_con_desplazamiento(dfila: int, dcolumna: int) -> np.ndarray:
    """Serie sintetica: la barra arranca en t3 y se mueve entre t3 y t4."""
    gt = np.zeros((config.N_FRAMES, 60, 60), dtype=np.uint8)
    gt[2][_barra()] = 1
    gt[3][_barra(fila=20 + dfila, columna=20 + dcolumna)] = 1
    return gt


def test_el_eje_mayor_de_una_barra_horizontal_va_por_columnas():
    mayor, menor = deriva.ejes_pca(_barra())
    assert abs(mayor[1]) == pytest.approx(1.0)
    assert abs(mayor[0]) == pytest.approx(0.0, abs=1e-9)
    assert abs(menor[0]) == pytest.approx(1.0)


def test_los_dos_ejes_son_perpendiculares_y_unitarios():
    mayor, menor = deriva.ejes_pca(_barra())
    assert float(mayor @ menor) == pytest.approx(0.0, abs=1e-9)
    assert np.linalg.norm(mayor) == pytest.approx(1.0)
    assert np.linalg.norm(menor) == pytest.approx(1.0)


def test_una_barra_diagonal_orienta_su_eje_a_45_grados():
    mascara = np.zeros((40, 40), dtype=bool)
    for i in range(5, 30):
        mascara[i, i] = True
    mayor, _ = deriva.ejes_pca(mascara)
    assert abs(mayor[0]) == pytest.approx(1 / math.sqrt(2))
    assert abs(mayor[1]) == pytest.approx(1 / math.sqrt(2))


def test_el_centroide_de_la_barra_cae_en_su_centro():
    np.testing.assert_allclose(deriva.centroide(_barra()), [21.0, 30.0])


def test_un_desplazamiento_a_lo_largo_del_surco_es_todo_paralelo():
    tabla = deriva.deriva_de_unidad(_serie_con_desplazamiento(0, 7), 1, 1)
    fila = tabla.iloc[0]
    assert abs(fila["paralela_px"]) == pytest.approx(7.0)
    assert abs(fila["perpendicular_px"]) == pytest.approx(0.0, abs=1e-9)
    assert fila["perpendicular_norm"] == pytest.approx(0.0, abs=1e-9)


def test_un_desplazamiento_perpendicular_al_surco_es_todo_perpendicular():
    tabla = deriva.deriva_de_unidad(_serie_con_desplazamiento(5, 0), 1, 1)
    fila = tabla.iloc[0]
    assert abs(fila["perpendicular_px"]) == pytest.approx(5.0)
    assert abs(fila["paralela_px"]) == pytest.approx(0.0, abs=1e-9)


def test_el_cociente_normaliza_por_el_semiancho_de_la_unidad():
    tabla = deriva.deriva_de_unidad(_serie_con_desplazamiento(5, 0), 1, 1)
    fila = tabla.iloc[0]
    assert fila["semiancho_start_px"] == pytest.approx(SEMIANCHO_BARRA)
    assert fila["perpendicular_norm"] == pytest.approx(5.0 / SEMIANCHO_BARRA)


def test_el_cociente_usa_el_semiancho_del_frame_no_el_de_start():
    # La barra se ensancha de 3 a 7 filas y ademas se desplaza 5 px: el cociente
    # tiene que dividir por el semiancho nuevo, mas grande, no por el de start.
    gt = np.zeros((config.N_FRAMES, 60, 60), dtype=np.uint8)
    gt[2][_barra()] = 1
    ancha = np.zeros((60, 60), dtype=bool)
    ancha[25:32, 20:41] = True
    gt[3][ancha] = 1

    fila = deriva.deriva_de_unidad(gt, 1, 1).iloc[0]
    semiancho_ancho = 4 * math.sqrt((7**2 - 1) / 12) / 2
    assert fila["semiancho_frame_px"] == pytest.approx(semiancho_ancho)
    assert fila["semiancho_start_px"] == pytest.approx(SEMIANCHO_BARRA)
    assert fila["perpendicular_norm"] == pytest.approx(
        abs(fila["perpendicular_px"]) / semiancho_ancho
    )
    # Con el semiancho de start el cociente saldria mucho mayor: eso es lo que
    # se queria dejar de hacer.
    assert deriva.norma_con_semiancho_de_start(
        deriva.deriva_de_unidad(gt, 1, 1)
    ).iloc[0] > fila["perpendicular_norm"]


def test_una_lesion_quieta_no_deriva():
    tabla = deriva.deriva_de_unidad(_serie_con_desplazamiento(0, 0), 1, 1)
    assert tabla["perpendicular_norm"].max() == pytest.approx(0.0, abs=1e-9)


def test_la_ventana_empieza_en_start_mas_uno():
    tabla = deriva.deriva_de_unidad(_serie_con_desplazamiento(5, 0), 1, 1)
    assert list(tabla["start"]) == [3]
    assert list(tabla["frame"]) == [4]


def test_una_barra_que_no_gira_da_giro_cero():
    tabla = deriva.deriva_de_unidad(_serie_con_desplazamiento(5, 3), 1, 1)
    assert tabla["giro_eje_grados"].max() == pytest.approx(0.0, abs=1e-9)


def test_el_resumen_marca_la_unidad_que_pasa_de_su_semiancho():
    # 5 px perpendiculares sobre un semiancho de 1.63 es un cociente de 3.06.
    resumen = deriva.resumen_por_unidad(
        deriva.deriva_de_unidad(_serie_con_desplazamiento(5, 0), 1, 1)
    )
    fila = resumen.iloc[0]
    assert fila["supera_semiancho"]
    assert fila["perp_max_norm"] == pytest.approx(5.0 / SEMIANCHO_BARRA)
    assert fila["frame_del_maximo"] == 4


def test_una_deriva_menor_que_el_semiancho_no_marca_la_unidad():
    resumen = deriva.resumen_por_unidad(
        deriva.deriva_de_unidad(_serie_con_desplazamiento(1, 0), 1, 1)
    )
    # 1 px sobre un semiancho de 1.63: el clic sigue dentro.
    assert not resumen.iloc[0]["supera_semiancho"]


def test_el_pixel_del_clic_es_el_centroide_redondeado():
    assert deriva.pixel_del_centroide(_barra()) == (21, 30)


def test_el_clic_sigue_dentro_si_la_lesion_apenas_se_mueve():
    # La barra ocupa las filas 20 a 22 y el clic esta en la 21. Desplazada una
    # fila ocupa de la 21 a la 23, asi que el clic sigue cayendo dentro.
    tabla = deriva.deriva_de_unidad(_serie_con_desplazamiento(1, 0), 1, 1)
    assert tabla.iloc[0]["clic_dentro"]


def test_el_clic_se_sale_cuando_la_lesion_se_va():
    # Desplazada cinco filas ocupa de la 25 a la 27: el clic de la fila 21 queda
    # fuera. Es exactamente lo que le pasa al clic fijo de M1.
    tabla = deriva.deriva_de_unidad(_serie_con_desplazamiento(5, 0), 1, 1)
    assert not tabla.iloc[0]["clic_dentro"]


def test_un_desplazamiento_a_lo_largo_del_surco_no_saca_el_clic():
    tabla = deriva.deriva_de_unidad(_serie_con_desplazamiento(0, 7), 1, 1)
    assert tabla.iloc[0]["clic_dentro"]


def test_las_dos_fracciones_del_test_directo():
    tabla = deriva.deriva_de_unidad(_serie_con_desplazamiento(5, 0), 1, 1)
    fracciones = deriva.fracciones_del_clic_fijo(tabla)
    assert fracciones["pares_fuera"] == 1
    assert fracciones["pares_totales"] == 1
    assert fracciones["fraccion_pares_fuera"] == 1.0
    assert fracciones["unidades_que_se_salen"] == 1
    assert fracciones["fraccion_unidades_que_se_salen"] == 1.0


def test_una_lesion_quieta_no_saca_el_clic_en_ningun_frame():
    fracciones = deriva.fracciones_del_clic_fijo(
        deriva.deriva_de_unidad(_serie_con_desplazamiento(0, 0), 1, 1)
    )
    assert fracciones["fraccion_pares_fuera"] == 0.0
    assert fracciones["fraccion_unidades_que_se_salen"] == 0.0


def test_el_resumen_recoge_el_primer_frame_en_que_se_sale():
    fila = deriva.resumen_por_unidad(
        deriva.deriva_de_unidad(_serie_con_desplazamiento(5, 0), 1, 1)
    ).iloc[0]
    assert fila["sale_alguna_vez"]
    assert fila["pares_fuera"] == 1
    assert fila["primer_frame_fuera"] == 4
    assert fila["clic_dentro_en_start"]
