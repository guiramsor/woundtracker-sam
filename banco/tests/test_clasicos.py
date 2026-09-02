"""Los tres metodos clasicos, sobre casos sinteticos con respuesta conocida.

La imagen de prueba es de dos niveles y con una barra recta, asi que el
resultado correcto no es una opinion: es la barra. Lo que se comprueba aqui no
es la calidad de los metodos sobre microscopia real, que la mide F9, sino que
cada uno hace lo que dice hacer y que el prompt de puntos se aplica.
"""

import numpy as np
import pytest

from surco import clasicos, config, metricas

LADO = 256
FONDO, SENAL = 20, 200
FILAS_BARRA = slice(126, 133)
COLUMNAS_BARRA = slice(88, 168)
CLIC_BARRA = (129, 128)
FILAS_OTRA = slice(160, 167)
CLIC_OTRA = (163, 128)


def _imagen(con_segunda_barra: bool = False):
    imagen = np.full((LADO, LADO), FONDO, dtype=np.uint8)
    imagen[FILAS_BARRA, COLUMNAS_BARRA] = SENAL
    if con_segunda_barra:
        imagen[FILAS_OTRA, COLUMNAS_BARRA] = SENAL
    return imagen


def _barra() -> np.ndarray:
    referencia = np.zeros((LADO, LADO), dtype=bool)
    referencia[FILAS_BARRA, COLUMNAS_BARRA] = True
    return referencia


def test_la_ventana_va_centrada_en_el_clic_y_no_se_sale():
    ventana, origen = clasicos.recortar(_imagen(), CLIC_BARRA)
    lado = config.LADO_REGION_CLIC_PX
    assert ventana.shape == (lado, lado)
    assert origen == (CLIC_BARRA[0] - lado // 2, CLIC_BARRA[1] - lado // 2)


def test_la_ventana_se_pega_al_borde_sin_encogerse():
    lado = config.LADO_REGION_CLIC_PX
    ventana, origen = clasicos.recortar(_imagen(), (2, 2))
    assert origen == (0, 0)
    assert ventana.shape == (lado, lado)


def test_los_clics_de_fuera_de_la_ventana_no_se_usan():
    lado = config.LADO_REGION_CLIC_PX
    origen = (65, 64)
    dentro = clasicos.clics_dentro(
        [CLIC_BARRA, (5, 5), (250, 250)], [1, 1, 0], origen, lado
    )
    assert len(dentro) == 1
    assert dentro[0] == ((CLIC_BARRA[0] - 65, CLIC_BARRA[1] - 64), 1)


def test_otsu_local_recupera_la_barra_exactamente():
    salida = clasicos.otsu_local(_imagen(), [CLIC_BARRA], [1])
    assert metricas.dice(salida, _barra()) == pytest.approx(1.0)


def test_un_clic_negativo_quita_su_componente():
    imagen = _imagen(con_segunda_barra=True)
    con_negativo = clasicos.otsu_local(imagen, [CLIC_BARRA, CLIC_OTRA], [1, 0])
    assert metricas.dice(con_negativo, _barra()) == pytest.approx(1.0)
    # Y sin el negativo, la segunda barra no entra igualmente: no la toca ningun
    # clic positivo, asi que la seleccion por componentes ya la deja fuera.
    sin_negativo = clasicos.otsu_local(imagen, [CLIC_BARRA], [1])
    assert not sin_negativo[FILAS_OTRA, COLUMNAS_BARRA].any()


def test_un_positivo_sobre_la_segunda_barra_si_la_mete():
    imagen = _imagen(con_segunda_barra=True)
    salida = clasicos.otsu_local(imagen, [CLIC_BARRA, CLIC_OTRA], [1, 1])
    assert salida[FILAS_BARRA, COLUMNAS_BARRA].all()
    assert salida[FILAS_OTRA, COLUMNAS_BARRA].all()


def test_sin_ningun_clic_util_la_mascara_sale_vacia():
    """Un clic positivo sobre fondo no cae en ninguna componente."""
    salida = clasicos.otsu_local(_imagen(), [(100, 100)], [1])
    assert not salida.any()


def test_una_imagen_plana_no_inventa_dos_clases():
    plana = np.full((LADO, LADO), FONDO, dtype=np.uint8)
    assert not clasicos.otsu_local(plana, [CLIC_BARRA], [1]).any()


def test_watershed_llega_a_la_barra_desde_un_solo_marcador():
    """P1 no trae negativos: el borde de la ventana es el marcador de fondo."""
    salida = clasicos.watershed_con_marcadores(_imagen(), [CLIC_BARRA], [1])
    assert metricas.dice(salida, _barra()) > 0.9


def test_sato_responde_a_la_barra():
    salida = clasicos.cresta_sato(_imagen(), [CLIC_BARRA], [1])
    assert salida.any()
    dentro = salida[FILAS_BARRA, COLUMNAS_BARRA].mean()
    assert dentro > 0.9, "la cresta tiene que cubrir casi toda la barra"


def test_los_tres_dejan_a_cero_todo_lo_de_fuera_de_la_ventana():
    lado = config.LADO_REGION_CLIC_PX
    fila0, col0 = CLIC_BARRA[0] - lado // 2, CLIC_BARRA[1] - lado // 2
    for metodo in config.METODOS_CLASICOS:
        salida = clasicos.segmentar(metodo, _imagen(), [CLIC_BARRA], [1])
        assert salida.shape == (LADO, LADO)
        assert salida.dtype == np.bool_
        fuera = salida.copy()
        fuera[fila0 : fila0 + lado, col0 : col0 + lado] = False
        assert not fuera.any(), f"{metodo} pinta fuera de la ventana del clic"


def test_la_cache_de_la_cresta_no_confunde_dos_imagenes():
    una = clasicos.cresta_sato(_imagen(), [CLIC_BARRA], [1])
    otra = clasicos.cresta_sato(_imagen(con_segunda_barra=True), [CLIC_OTRA], [1])
    assert otra[FILAS_OTRA, COLUMNAS_BARRA].any()
    assert not una[FILAS_OTRA, COLUMNAS_BARRA].any()


def test_metodo_desconocido_falla_en_vez_de_devolver_vacio():
    with pytest.raises(KeyError):
        clasicos.segmentar("frangi", _imagen(), [CLIC_BARRA], [1])
