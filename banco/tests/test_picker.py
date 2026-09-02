"""Tests de la maquina de estados del picker, sin ventana ni disco.

Lo que se prueba es la parte que puede arruinar una sesion de anotacion larga:
la cola de pendientes, el limite de cinco clics, deshacer, la excepcion y el
avance. El dibujo es de matplotlib y no se prueba aqui.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from surco import config, picker, prompts  # noqa: E402


UNIDADES = [
    {"serie": 1, "etiqueta": 1, "start": 3, "partida_en_start": False},
    {"serie": 2, "etiqueta": 2, "start": 4, "partida_en_start": True},
    {"serie": 5, "etiqueta": 3, "start": 4, "partida_en_start": False},
]

CINCO = [(100, 200), (90, 180), (110, 220), (70, 200), (130, 200)]


@pytest.fixture
def sin_disco(monkeypatch):
    """Anula la lectura del GT y la escritura del fichero."""
    mascara = np.zeros((config.LADO_PX, config.LADO_PX), dtype=bool)
    mascara[98:103, 150:251] = True
    monkeypatch.setattr(picker.datos, "cargar_gt", lambda serie, frame: mascara)
    guardados = []
    monkeypatch.setattr(
        picker.prompts, "guardar", lambda datos, ruta=None: guardados.append(datos)
    )
    return guardados


def test_la_cola_arranca_con_todas_las_unidades():
    p = picker.Picker(UNIDADES, prompts.esquema_vacio())
    assert len(p.cola) == 3
    assert p.nombre() == "1-L1"
    assert not p.terminado


def test_la_cola_omite_lo_ya_anotado(sin_disco):
    previos = prompts.registrar(
        prompts.esquema_vacio(), 1, 1, 3, CINCO, False,
        np.ones((config.LADO_PX, config.LADO_PX), dtype=bool),
    )
    p = picker.Picker(UNIDADES, previos)
    assert [u for u in p.cola] == [(2, 2), (5, 3)]
    assert p.nombre() == "2-L2"


def test_rehacer_reabre_una_unidad_ya_anotada(sin_disco):
    previos = prompts.registrar(
        prompts.esquema_vacio(), 5, 3, 4, CINCO, False,
        np.ones((config.LADO_PX, config.LADO_PX), dtype=bool),
    )
    p = picker.Picker(UNIDADES, previos, rehacer=[(5, 3)])
    assert p.cola == [(5, 3)]
    assert p.nombre() == "5-L3"


def test_rehacer_no_admite_una_unidad_que_no_existe():
    with pytest.raises(KeyError, match="no existen"):
        picker.Picker(UNIDADES, prompts.esquema_vacio(), rehacer=[(8, 1)])


def test_rehacer_reemplaza_y_no_duplica(sin_disco):
    previos = prompts.registrar(
        prompts.esquema_vacio(), 1, 1, 3, CINCO, False,
        np.ones((config.LADO_PX, config.LADO_PX), dtype=bool),
    )
    p = picker.Picker(UNIDADES, previos, rehacer=[(1, 1)])
    for fila, columna in [(300, 400)] + CINCO[1:]:
        p.poner_clic(fila, columna)
    p.confirmar()
    assert len(p.datos["unidades"]) == 1
    assert p.datos["unidades"][0]["clics"][0] == [300, 400]


@pytest.mark.parametrize(
    "texto,esperado", [("9-L2", (9, 2)), ("10-l1", (10, 1)), (" 5-L3 ", (5, 3))]
)
def test_se_entiende_el_nombre_de_la_unidad(texto, esperado):
    assert picker.parsear_unidad(texto) == esperado


def test_un_nombre_mal_escrito_no_pasa():
    with pytest.raises(ValueError, match="mal escrita"):
        picker.parsear_unidad("9_2")


def test_no_se_ponen_mas_de_cinco_clics():
    p = picker.Picker(UNIDADES, prompts.esquema_vacio())
    for fila, columna in CINCO + [(1, 1), (2, 2)]:
        p.poner_clic(fila, columna)
    assert len(p.clics) == config.N_CLICS
    assert p.clics[-1] == CINCO[-1]


def test_el_clic_se_redondea_al_pixel():
    p = picker.Picker(UNIDADES, prompts.esquema_vacio())
    p.poner_clic(100.6, 199.4)
    assert p.clics == [(101, 199)]


def test_confirmar_guarda_avanza_y_limpia(sin_disco):
    p = picker.Picker(UNIDADES, prompts.esquema_vacio())
    for fila, columna in CINCO:
        p.poner_clic(fila, columna)
    p.excepcion = True
    p.confirmar()

    assert len(sin_disco) == 1, "tiene que guardar en cuanto se cierra una unidad"
    assert prompts.anotadas(p.datos) == {(1, 1)}
    assert p.datos["unidades"][0]["excepcion"] is True
    assert p.datos["unidades"][0]["start"] == 3
    assert p.clics == []
    assert p.excepcion is False
    assert p.nombre() == "2-L2"


def test_saltar_no_guarda_pero_avanza(sin_disco):
    p = picker.Picker(UNIDADES, prompts.esquema_vacio())
    p.avanzar()
    assert sin_disco == []
    assert prompts.anotadas(p.datos) == set()
    assert p.nombre() == "2-L2"


def test_la_sesion_termina_al_pasar_la_ultima():
    p = picker.Picker(UNIDADES, prompts.esquema_vacio())
    for _ in range(3):
        p.avanzar()
    assert p.terminado


def test_el_panel_recuerda_el_orden_y_la_curva():
    p = picker.Picker(UNIDADES, prompts.esquema_vacio())
    texto = p.texto_lateral()
    assert "ORDEN DE LOS CINCO CLICS" in texto
    assert "SOBRE EL TRAZO" in texto
    assert "centro del surco" in texto
    assert "nucleoplasma" in texto


def test_el_panel_avisa_solo_en_las_unidades_que_arrancan_partidas():
    p = picker.Picker(UNIDADES, prompts.esquema_vacio())
    assert "ARRANCA PARTIDA" not in p.texto_lateral()
    p.avanzar()
    assert p.nombre() == "2-L2"
    assert "ARRANCA PARTIDA" in p.texto_lateral()


def test_el_panel_lleva_la_cuenta_de_lo_que_falta():
    p = picker.Picker(UNIDADES, prompts.esquema_vacio())
    assert "anotadas 0 de 3" in p.texto_lateral()
    p.poner_clic(*CINCO[0])
    assert "puestos: 1 de 5" in p.texto_lateral()


def test_se_arranca_mirando_el_frame_start():
    p = picker.Picker(UNIDADES, prompts.esquema_vacio())
    assert p.frame_visible == 3
    assert p.en_start
    assert not p.consulta_temporal


def test_moverse_por_la_secuencia_marca_la_consulta_temporal():
    p = picker.Picker(UNIDADES, prompts.esquema_vacio())
    p.mover_frame(+4)
    assert p.frame_visible == 7
    assert not p.en_start
    assert p.consulta_temporal


def test_volver_a_start_no_borra_que_se_consulto():
    p = picker.Picker(UNIDADES, prompts.esquema_vacio())
    p.mover_frame(+2)
    p.volver_a_start()
    assert p.en_start
    assert p.consulta_temporal, "la desviacion se declara aunque se vuelva"


def test_no_se_sale_de_la_secuencia():
    p = picker.Picker(UNIDADES, prompts.esquema_vacio())
    p.mover_frame(-10)
    assert p.frame_visible == 3, "no hay frame 0"
    p.mover_frame(+100)
    assert p.frame_visible == 3, "no hay frame 22"
    assert not p.consulta_temporal


def test_fuera_de_start_no_se_clica():
    p = picker.Picker(UNIDADES, prompts.esquema_vacio())
    p.mover_frame(+5)
    p.poner_clic(100, 200)
    assert p.clics == []
    p.volver_a_start()
    p.poner_clic(100, 200)
    assert p.clics == [(100, 200)]


def test_la_consulta_temporal_se_guarda_en_el_fichero(sin_disco):
    p = picker.Picker(UNIDADES, prompts.esquema_vacio())
    p.mover_frame(+6)
    p.volver_a_start()
    for fila, columna in CINCO:
        p.poner_clic(fila, columna)
    p.confirmar()
    assert p.datos["unidades"][0]["consulta_temporal"] is True


def test_sin_consulta_temporal_el_campo_queda_en_falso(sin_disco):
    p = picker.Picker(UNIDADES, prompts.esquema_vacio())
    for fila, columna in CINCO:
        p.poner_clic(fila, columna)
    p.confirmar()
    assert p.datos["unidades"][0]["consulta_temporal"] is False


def test_al_avanzar_se_vuelve_al_start_de_la_siguiente(sin_disco):
    p = picker.Picker(UNIDADES, prompts.esquema_vacio())
    p.mover_frame(+5)
    p.avanzar()
    assert p.nombre() == "2-L2"
    assert p.frame_visible == 4
    assert not p.consulta_temporal


def test_el_panel_avisa_cuando_no_se_esta_en_start():
    p = picker.Picker(UNIDADES, prompts.esquema_vacio())
    assert "OTRO FRAME" not in p.texto_lateral()
    p.mover_frame(+3)
    texto = p.texto_lateral()
    assert "OTRO FRAME" in texto
    assert "consulta temporal: SI" in texto


def test_el_brillo_no_toca_lo_que_se_guarda(sin_disco):
    p = picker.Picker(UNIDADES, prompts.esquema_vacio())
    p.factor_brillo = 0.5
    for fila, columna in CINCO:
        p.poner_clic(fila, columna)
    p.confirmar()
    guardado = p.datos["unidades"][0]
    assert "factor_brillo" not in guardado
    assert guardado["clics"] == [list(c) for c in CINCO]
