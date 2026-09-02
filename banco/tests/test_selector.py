"""S1 sobre candidatas sinteticas, donde se sabe cual tiene que salir.

Cada test aisla una pieza del criterio: la puerta, cada una de sus dos guardas,
el orden por z, el desempate por elongacion y el respaldo. La imagen es de dos
niveles, asi que el z de cada candidata no es una opinion.
"""

import numpy as np
import pytest

from surco import modelos, selector

LADO = 256
FONDO, SENAL = 20, 200
BLOQUE = (slice(120, 140), slice(88, 168))
CLIC = (129, 128)
# Dentro del bloque de senal, como los del protocolo real: van en nucleoplasma,
# no en negro. Un negativo sobre fondo oscuro no distinguiria nada.
CLIC_NEGATIVO = (122, 150)


def _imagen() -> np.ndarray:
    imagen = np.full((LADO, LADO), FONDO, dtype=np.uint8)
    imagen[BLOQUE] = SENAL
    return imagen


def _mascara(filas: slice, columnas: slice) -> np.ndarray:
    mascara = np.zeros((LADO, LADO), dtype=bool)
    mascara[filas, columnas] = True
    return mascara


def _candidatas(mascaras, puntuaciones) -> modelos.Candidatas:
    return modelos.Candidatas(np.stack(mascaras), np.array(puntuaciones, dtype=float))


# La que esta sobre la senal y contiene el clic: la respuesta correcta.
BUENA = _mascara(slice(125, 132), slice(100, 160))
# Tambien contiene el clic, pero se come medio fondo: mismo sitio, peor z.
GORDA = _mascara(slice(100, 160), slice(100, 160))
# Ni contiene el clic ni esta sobre la senal.
LEJOS = _mascara(slice(200, 220), slice(200, 220))
# Entera sobre la senal, asi que gana por z, pero se lleva por delante el clic
# negativo: es la candidata que solo la segunda guarda puede parar.
ANCHA = _mascara(slice(120, 135), slice(95, 165))
# Contiene el clic, no toca el negativo, y es mayoritariamente fondo: pasa la
# puerta siempre y pierde por z contra cualquiera que este sobre la senal.
ALTA = _mascara(slice(100, 170), slice(120, 140))


def test_la_puerta_exige_contener_el_clic_cero():
    assert selector.pasa_la_puerta(BUENA, [CLIC], [1])
    assert not selector.pasa_la_puerta(LEJOS, [CLIC], [1])


def test_la_segunda_guarda_solo_existe_si_hay_negativos():
    """Es la diferencia entre P1 o P3 y P2 o P4."""
    assert selector.pasa_la_puerta(ANCHA, [CLIC], [1])
    assert not selector.pasa_la_puerta(ANCHA, [CLIC, CLIC_NEGATIVO], [1, 0])


def test_el_z_separa_la_candidata_ajustada_de_la_gorda():
    gris = _imagen()
    assert selector.z_robusto(gris, BUENA, CLIC) > selector.z_robusto(gris, GORDA, CLIC)


def test_una_candidata_fuera_de_la_ventana_no_tiene_z():
    assert selector.z_robusto(_imagen(), LEJOS, CLIC) == selector.Z_SIN_MUESTRA


def test_s1_corrige_a_s0_cuando_la_de_mas_score_no_pasa_la_puerta():
    candidatas = _candidatas([BUENA, LEJOS], [0.1, 0.9])
    assert int(np.argmax(candidatas.puntuaciones)) == 1  # esto es S0
    indice, respaldo = selector.elegir(candidatas, _imagen(), [CLIC], [1])
    assert (indice, respaldo) == (0, False)


def test_entre_las_que_pasan_gana_el_z_mayor():
    candidatas = _candidatas([GORDA, BUENA], [0.9, 0.1])
    indice, respaldo = selector.elegir(candidatas, _imagen(), [CLIC], [1])
    assert (indice, respaldo) == (1, False)


def test_el_negativo_cambia_la_eleccion_entre_protocolos():
    """La misma pareja, con y sin clic de fondo, da candidatas distintas.

    Es el nivel de mas que S1 tiene en P2 y P4: sin negativos gana ANCHA por z,
    y con ellos la segunda guarda la para y pasa ALTA.
    """
    candidatas = _candidatas([ANCHA, ALTA], [0.9, 0.1])
    gris = _imagen()
    assert selector.z_robusto(gris, ANCHA, CLIC) > selector.z_robusto(gris, ALTA, CLIC)
    sin_negativo, _ = selector.elegir(candidatas, gris, [CLIC], [1])
    con_negativo, _ = selector.elegir(candidatas, gris, [CLIC, CLIC_NEGATIVO], [1, 0])
    assert sin_negativo == 0
    assert con_negativo == 1


def test_si_no_pasa_ninguna_tira_del_respaldo_y_lo_marca():
    candidatas = _candidatas([LEJOS, GORDA], [0.9, 0.1])
    indice, respaldo = selector.elegir(candidatas, _imagen(), [CLIC], [0])
    assert respaldo
    assert indice == 1, "el respaldo se queda con la de mayor z, no con la primera"


def test_la_elongacion_desempata_cuando_el_z_es_identico():
    """Dos candidatas de la misma area y dentro del mismo bloque uniforme.

    Con la imagen de dos niveles y la misma cantidad de pixeles quitada al
    bloque, las dos ven exactamente el mismo fondo local: el z coincide hasta el
    ultimo bit y decide la forma.
    """
    alargada = _mascara(slice(125, 130), slice(100, 120))
    cuadrada = _mascara(slice(125, 135), slice(100, 110))
    clic = (127, 105)
    gris = _imagen()
    assert selector.z_robusto(gris, alargada, clic) == selector.z_robusto(
        gris, cuadrada, clic
    )
    assert selector.elongacion(alargada) > selector.elongacion(cuadrada)

    candidatas = _candidatas([cuadrada, alargada], [0.9, 0.1])
    indice, respaldo = selector.elegir(candidatas, gris, [clic], [1])
    assert (indice, respaldo) == (1, False)


def test_una_candidata_vacia_no_rompe_nada():
    vacia = np.zeros((LADO, LADO), dtype=bool)
    assert selector.elongacion(vacia) == 0.0
    candidatas = _candidatas([vacia, BUENA], [0.9, 0.1])
    indice, respaldo = selector.elegir(candidatas, _imagen(), [CLIC], [1])
    assert (indice, respaldo) == (1, False)


def test_una_imagen_plana_no_da_un_z_infinito_sin_diferencia():
    plana = np.full((LADO, LADO), FONDO, dtype=np.uint8)
    assert selector.z_robusto(plana, BUENA, CLIC) == pytest.approx(0.0)
