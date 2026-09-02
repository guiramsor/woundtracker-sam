"""Tests de F0.

Comprueban que el paquete se importa, que lo que el config declara existe en
disco y que las constantes son coherentes entre si. No leen un solo pixel: eso
es F1.
"""

import pytest

from surco import config


def test_la_rejilla_de_profundidad_esta_clavada():
    """Los cuatro representantes, escritos aqui para que cambiarlos sea ruidoso.

    Es la unica constante del trabajo que puede fallar en silencio: si un
    representante cambiara, los scripts medirian modelos distintos sin que nada
    reventara y los CSV ya reportados dejarian de corresponder. Este test es el
    ruido que falta.
    """
    assert config.REJILLA_PROFUNDIDAD == {
        "microsam_lm": "vit_b",
        "sammed2d": "sam_med2d",
        "tinysam": "vit_t",
        "sam21": "tiny",
    }


def test_la_mejor_celda_es_una_de_la_rejilla():
    """La celda de los overlays y del apartado 7 sale de un modelo medido."""
    mejor = config.MEJOR_CELDA
    assert config.REJILLA_PROFUNDIDAD[mejor["modelo"]] == mejor["checkpoint"]
    assert mejor["modo"] in config.MODOS
    assert mejor["protocolo"] in config.PROTOCOLOS
    assert mejor["selector"] in config.SELECTORES


def test_la_mejor_celda_es_de_verdad_la_mejor():
    """Y ademas la mejor **medida**, que es lo que el test de arriba no ve.

    Esta comprobacion existe porque el fallo ya ocurrio: `MEJOR_CELDA` decia
    M1 x P4 x S1 y siguio diciendolo cuando se midio M4 y el techo subio a
    0.5517. El test anterior pasaba igual, porque M1 x P4 x S1 es una celda
    perfectamente valida; lo que dejo de ser es la mejor. Consecuencia: los
    overlays dibujaban la segunda mejor configuracion y el apartado 7 comparaba
    el techo humano contra ella.

    Necesita `salidas/` poblada, asi que se salta en una copia recien clonada
    en vez de fallar: no es un test del codigo, es un test de que la constante
    sigue describiendo lo que hay medido.
    """
    from surco import verificacion

    try:
        medida = verificacion.mejor_celda_medida()
    except FileNotFoundError as falta:
        pytest.skip(f"hace falta correr F8 y F10 antes: {falta}")

    declarada = config.MEJOR_CELDA
    for clave in ("modelo", "modo", "protocolo", "selector"):
        assert medida[clave] == declarada[clave], (
            f"MEJOR_CELDA dice {declarada[clave]!r} en {clave!r} y lo medido es "
            f"{medida[clave]!r}: la mejor celda ha cambiado y la constante no"
        )


def test_las_rutas_declaradas_existen():
    for ruta in (
        config.DIR_DATOS,
        config.DIR_IMAGENES,
        config.DIR_GT,
        config.DIR_ANOTACIONES,
        config.DIR_SALIDAS,
    ):
        assert ruta.is_dir(), f"no existe el directorio {ruta}"


@pytest.mark.parametrize("serie", config.SERIES)
def test_cada_serie_tiene_sus_frames(serie):
    for frame in range(1, config.N_FRAMES + 1):
        nombre = config.PATRON_IMAGEN.format(
            canal=config.CANAL, serie=serie, frame=frame
        )
        assert (config.DIR_IMAGENES / nombre).is_file(), f"falta {nombre}"


def test_el_gt_cubre_las_series_con_gt_y_solo_esas():
    presentes = {
        int(p.name.removeprefix("serie_"))
        for p in config.DIR_GT.iterdir()
        if p.is_dir()
    }
    assert presentes == set(config.SERIES_CON_GT)
    assert 8 not in presentes


@pytest.mark.parametrize("serie", config.SERIES_CON_GT)
def test_cada_serie_con_gt_tiene_sus_mascaras(serie):
    for frame in range(1, config.N_FRAMES + 1):
        relativa = config.PATRON_GT.format(serie=serie, frame=frame)
        assert (config.DIR_GT / relativa).is_file(), f"falta {relativa}"


def test_los_protocolos_son_subconjuntos_de_los_cinco_clics():
    todos = set(range(config.N_CLICS))
    assert set(config.PROTOCOLOS["P4"]) == todos
    for nombre, indices in config.PROTOCOLOS.items():
        assert set(indices) <= todos, nombre
        assert len(set(indices)) == len(indices), f"{nombre} repite un indice"


def test_positivos_y_negativos_parten_los_cinco_clics():
    positivos = set(config.CLICS_POSITIVOS)
    negativos = set(config.CLICS_NEGATIVOS)
    assert positivos | negativos == set(range(config.N_CLICS))
    assert not positivos & negativos


def test_el_reparto_de_pares_cuadra_con_el_total():
    assert set(config.PARES_POR_SERIE) == set(config.SERIES_CON_GT)
    assert sum(config.PARES_POR_SERIE.values()) == config.N_PARES


def test_la_fragmentacion_se_cuenta_con_ocho_vecinos():
    assert config.CONECTIVIDAD_FRAGMENTACION == 8
    assert config.COLUMNA_FRAGMENTACION == "n_comp_c8"


def test_el_umbral_catastrofico_es_el_de_la_base():
    assert config.UMBRAL_CATASTROFICO == 0.1


def test_las_convenciones_del_bootstrap_estan_declaradas():
    # No salen de la base: estan en el config para que se vean y se cambien.
    assert 0 < config.NIVEL_CONFIANZA < 1
    assert config.N_REMUESTREOS > 0
    assert isinstance(config.SEMILLA_BOOTSTRAP, int)


def test_no_hay_semiancho_global():
    # No es un capricho: un semiancho unico en el config invitaria a usarlo como
    # umbral en F2, y el umbral es el semiancho de cada lesion, medido del GT.
    assert not hasattr(config, "SEMIANCHO_PX")
