"""Tests de la capa de lectura. Sin tocar disco salvo para comprobar nombres."""

import numpy as np
import pytest

from surco import config, datos


def test_los_dos_patrones_rellenan_ceros_de_forma_distinta():
    # Comprobado en disco: la imagen es "_t1" y el GT es "gt_t01".
    assert datos.ruta_imagen(1, 1).name == "CTL 3_w1CSU488_s1_t1.TIF"
    assert datos.ruta_gt(1, 1).name == "gt_t01.tif"
    assert datos.ruta_gt(1, 1).parent.name == "serie_1"


def test_ruta_imagen_no_rellena_el_frame_de_dos_cifras():
    assert datos.ruta_imagen(10, 21).name == "CTL 3_w1CSU488_s10_t21.TIF"


def test_proyectar_maximo_toma_el_maximo_plano_a_plano():
    stack = np.array(
        [
            [[1, 9], [0, 0]],
            [[5, 2], [7, 0]],
            [[3, 3], [0, 4]],
        ],
        dtype=np.uint16,
    )
    esperado = np.array([[5, 9], [7, 4]], dtype=np.uint16)
    np.testing.assert_array_equal(datos.proyectar_maximo(stack), esperado)


def test_proyectar_maximo_conserva_el_dtype():
    stack = np.zeros((config.N_PLANOS_Z, 4, 4), dtype=np.uint16)
    assert datos.proyectar_maximo(stack).dtype == np.uint16


def test_proyectar_maximo_rechaza_una_imagen_plana():
    with pytest.raises(ValueError, match="3 ejes"):
        datos.proyectar_maximo(np.zeros((4, 4), dtype=np.uint16))
