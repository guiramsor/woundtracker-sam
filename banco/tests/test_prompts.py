"""Tests del fichero de prompts: formato, reanudacion y diagnosticos.

El picker no se prueba aqui: es la ventana, y lo que importa que no se rompa es
el fichero, que es la unica entrada irrecuperable del trabajo.
"""

import json

import numpy as np
import pytest

from surco import config, prompts


CINCO = [(100, 200), (90, 180), (110, 220), (70, 200), (130, 200)]


@pytest.fixture
def ruta(tmp_path):
    return tmp_path / "clics.json"


@pytest.fixture
def mascara():
    """Una barra horizontal centrada en la fila 100, columnas 150 a 250."""
    m = np.zeros((config.LADO_PX, config.LADO_PX), dtype=bool)
    m[98:103, 150:251] = True
    return m


def test_un_fichero_que_no_existe_da_el_esquema_vacio(ruta):
    datos = prompts.cargar(ruta)
    assert datos["unidades"] == []
    assert datos["n_clics"] == config.N_CLICS


def test_lo_guardado_se_relee_igual(ruta, mascara):
    datos = prompts.registrar(
        prompts.esquema_vacio(), 5, 2, 3, CINCO, False, mascara
    )
    prompts.guardar(datos, ruta)
    assert prompts.cargar(ruta)["unidades"] == datos["unidades"]


def test_el_start_viaja_con_los_clics(ruta, mascara):
    # Sin el start, el apartado 6.2 no sabe donde empieza su ventana.
    datos = prompts.registrar(prompts.esquema_vacio(), 6, 1, 9, CINCO, False, mascara)
    prompts.guardar(datos, ruta)
    assert prompts.cargar(ruta)["unidades"][0]["start"] == 9


def test_se_guardan_los_cinco_clics_en_orden(mascara):
    datos = prompts.registrar(prompts.esquema_vacio(), 1, 1, 3, CINCO, False, mascara)
    assert datos["unidades"][0]["clics"] == [list(c) for c in CINCO]


def test_no_se_admite_un_numero_de_clics_distinto_de_cinco(mascara):
    with pytest.raises(ValueError, match="hacen falta"):
        prompts.registrar(prompts.esquema_vacio(), 1, 1, 3, CINCO[:3], False, mascara)


def test_no_se_admite_un_clic_fuera_de_la_imagen(mascara):
    fuera = CINCO[:4] + [(config.LADO_PX, 10)]
    with pytest.raises(ValueError, match="fuera de la imagen"):
        prompts.registrar(prompts.esquema_vacio(), 1, 1, 3, fuera, False, mascara)


def test_las_pendientes_son_las_que_faltan(mascara):
    todas = [(1, 1), (1, 2), (5, 3)]
    datos = prompts.registrar(prompts.esquema_vacio(), 1, 2, 4, CINCO, False, mascara)
    assert prompts.pendientes(datos, todas) == [(1, 1), (5, 3)]


def test_reanudar_una_sesion_a_medias(ruta, mascara):
    todas = [(1, 1), (1, 2), (5, 3)]
    datos = prompts.registrar(prompts.esquema_vacio(), 1, 1, 3, CINCO, False, mascara)
    prompts.guardar(datos, ruta)
    # Se cierra la herramienta y se vuelve a abrir.
    recargado = prompts.cargar(ruta)
    assert prompts.anotadas(recargado) == {(1, 1)}
    assert prompts.pendientes(recargado, todas) == [(1, 2), (5, 3)]


def test_reanotar_una_unidad_la_reemplaza_y_no_la_duplica(mascara):
    datos = prompts.registrar(prompts.esquema_vacio(), 1, 1, 3, CINCO, False, mascara)
    otros = [(101, 201)] + CINCO[1:]
    datos = prompts.registrar(datos, 1, 1, 3, otros, True, mascara)
    assert len(datos["unidades"]) == 1
    assert datos["unidades"][0]["clics"][0] == [101, 201]
    assert datos["unidades"][0]["excepcion"]


def test_las_unidades_quedan_ordenadas(mascara):
    datos = prompts.esquema_vacio()
    for serie, etiqueta in ((10, 1), (1, 2), (5, 3)):
        datos = prompts.registrar(datos, serie, etiqueta, 3, CINCO, False, mascara)
    claves = [(u["serie"], u["etiqueta"]) for u in datos["unidades"]]
    assert claves == sorted(claves)


def test_el_diagnostico_mide_del_clic_central_al_centroide(mascara):
    # El centroide de la barra cae en (100, 200) y el clic 1 esta ahi mismo.
    diagnostico = prompts.diagnosticos(CINCO, mascara)
    assert diagnostico["dist_clic_centroide"] == pytest.approx(0.0)
    assert diagnostico["clic_dentro_gt"]


def test_un_clic_central_desplazado_da_distancia_y_queda_fuera(mascara):
    desplazado = [(130, 200)] + CINCO[1:]
    diagnostico = prompts.diagnosticos(desplazado, mascara)
    assert diagnostico["dist_clic_centroide"] == pytest.approx(30.0)
    assert not diagnostico["clic_dentro_gt"]


def test_la_excepcion_se_guarda_tal_como_se_marco(mascara):
    datos = prompts.registrar(prompts.esquema_vacio(), 2, 2, 4, CINCO, True, mascara)
    assert datos["unidades"][0]["excepcion"] is True


def test_los_protocolos_salen_por_subconjunto_de_la_anotacion_unica():
    assert prompts.subconjunto(CINCO, "P1") == [CINCO[0]]
    assert prompts.subconjunto(CINCO, "P2") == [CINCO[0], CINCO[3]]
    assert prompts.subconjunto(CINCO, "P3") == CINCO[:3]
    assert prompts.subconjunto(CINCO, "P4") == CINCO


def test_no_se_lee_un_fichero_de_otra_version(ruta):
    ruta.write_text(json.dumps({"version": 99, "unidades": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        prompts.cargar(ruta)


def test_el_temporal_no_sobrevive_al_guardado(ruta, mascara):
    prompts.guardar(prompts.esquema_vacio(), ruta)
    assert ruta.exists()
    assert not ruta.with_name(ruta.name + ".tmp").exists()
