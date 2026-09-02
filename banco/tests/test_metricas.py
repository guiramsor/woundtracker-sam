"""Tests de las metricas sobre casos con resultado conocido a mano.

Sin modelos y sin datos reales: mascaras de cuadrados y series de numeros
elegidas para que el resultado se pueda calcular en el margen.
"""

import numpy as np
import pandas as pd
import pytest

from surco import config, metricas


def _cuadrado(lado: int, fila: int = 0, columna: int = 0, tamano: int = 10):
    mascara = np.zeros((lado, lado), dtype=bool)
    mascara[fila : fila + tamano, columna : columna + tamano] = True
    return mascara


# -- 6.1 que se mide --------------------------------------------------------


def test_dice_de_una_mascara_consigo_misma_es_uno():
    cuadrado = _cuadrado(40)
    assert metricas.dice(cuadrado, cuadrado) == 1.0


def test_dice_de_mascaras_disjuntas_es_cero():
    assert metricas.dice(_cuadrado(40), _cuadrado(40, columna=20)) == 0.0


def test_dice_de_un_solape_a_la_mitad():
    # Dos cuadrados de 100 px que comparten 50: 2*50 / (100 + 100) = 0.5.
    uno = _cuadrado(40)
    otro = _cuadrado(40, columna=5)
    assert metricas.dice(uno, otro) == pytest.approx(0.5)


def test_una_prediccion_vacia_da_cero():
    vacia = np.zeros((40, 40), dtype=bool)
    assert metricas.dice(vacia, _cuadrado(40)) == 0.0


def test_un_gt_vacio_no_se_puntua():
    with pytest.raises(ValueError, match="no esta definido"):
        metricas.dice(_cuadrado(40), np.zeros((40, 40), dtype=bool))


def test_iou_sale_del_dice_y_conserva_el_orden():
    assert metricas.iou_desde_dice(1.0) == pytest.approx(1.0)
    assert metricas.iou_desde_dice(0.0) == pytest.approx(0.0)
    assert metricas.iou_desde_dice(0.5) == pytest.approx(1 / 3)
    valores = [0.1, 0.3, 0.55, 0.8]
    ious = [metricas.iou_desde_dice(v) for v in valores]
    assert ious == sorted(ious), "monotona: el ranking no puede cambiar"


# -- 6.2 sobre que frames ---------------------------------------------------


def test_la_ventana_empieza_despues_del_frame_semilla():
    assert metricas.ventana_de_evaluacion(range(3, 22), start=3) == tuple(range(4, 22))


def test_la_ventana_no_incluye_frames_sin_anotar():
    # La lesion existe en 4, 5 y 9: los huecos no entran aunque esten en rango.
    assert metricas.ventana_de_evaluacion([4, 5, 9], start=4) == (5, 9)


def test_la_ventana_de_una_unidad_tardia():
    assert metricas.ventana_de_evaluacion(range(9, 22), start=9) == tuple(range(10, 22))


# -- 6.3 como se agrega -----------------------------------------------------


@pytest.fixture
def tabla_desigual():
    """Dos unidades con muy distinto numero de frames, como s6 contra s5."""
    filas = [{"serie": 1, "etiqueta": 1, "dice": 0.9} for _ in range(3)]
    filas += [{"serie": 5, "etiqueta": 1, "dice": 0.3} for _ in range(27)]
    return pd.DataFrame(filas)


def test_se_promedia_primero_dentro_de_cada_lesion(tabla_desigual):
    unidades = metricas.por_unidad(tabla_desigual)
    assert unidades.loc[(1, 1)] == pytest.approx(0.9)
    assert unidades.loc[(5, 1)] == pytest.approx(0.3)


def test_la_media_entre_unidades_no_es_la_media_plana(tabla_desigual):
    unidades = metricas.por_unidad(tabla_desigual)
    assert metricas.media_entre_unidades(unidades) == pytest.approx(0.6)
    # La media plana sobre los 30 frames daria 0.36: la lesion con 27 frames
    # se lleva nueve veces mas peso sin que nadie lo haya decidido.
    assert tabla_desigual["dice"].mean() == pytest.approx(0.36)


# -- 6.4 como se comparan dos modelos ---------------------------------------


@pytest.fixture
def dos_modelos():
    unidades = pd.MultiIndex.from_tuples(
        [(s, 1) for s in range(1, 18)], names=["serie", "etiqueta"]
    )
    uno = pd.Series([0.60] * 17, index=unidades)
    otro = pd.Series([0.58] * 15 + [0.75, 0.75], index=unidades)
    return uno, otro


def test_la_diferencia_es_unidad_a_unidad(dos_modelos):
    uno, otro = dos_modelos
    diferencias = metricas.diferencia_pareada(uno, otro)
    assert len(diferencias) == 17
    assert diferencias.iloc[0] == pytest.approx(0.02)
    assert diferencias.iloc[-1] == pytest.approx(-0.15)


def test_no_se_comparan_conjuntos_distintos_de_unidades(dos_modelos):
    uno, otro = dos_modelos
    with pytest.raises(ValueError, match="mismas unidades"):
        metricas.diferencia_pareada(uno, otro.iloc[:10])


def test_ganar_en_quince_de_diecisiete_se_ve_aunque_la_media_no(dos_modelos):
    uno, otro = dos_modelos
    diferencias = metricas.diferencia_pareada(uno, otro)
    assert metricas.fraccion_de_unidades_ganadas(diferencias) == pytest.approx(15 / 17)
    # La media de la diferencia es casi cero y sin embargo gana en 15 de 17:
    # esto es lo que no se ve mirando solo si los intervalos se solapan.
    assert abs(diferencias.mean()) < 0.002


def test_el_bootstrap_es_reproducible():
    valores = np.linspace(0.4, 0.8, 17)
    argumentos = dict(n_remuestreos=2000, nivel=0.95, semilla=config.SEMILLA_BOOTSTRAP)
    assert metricas.intervalo_bootstrap(valores, **argumentos) == (
        metricas.intervalo_bootstrap(valores, **argumentos)
    )


def test_el_intervalo_contiene_la_media_y_se_estrecha_al_subir_el_nivel():
    valores = np.linspace(0.4, 0.8, 17)
    bajo, alto = metricas.intervalo_bootstrap(valores, 5000, 0.95, 1)
    assert bajo < valores.mean() < alto
    estrecho = metricas.intervalo_bootstrap(valores, 5000, 0.50, 1)
    assert (estrecho[1] - estrecho[0]) < (alto - bajo)


def test_una_diferencia_siempre_positiva_da_un_intervalo_positivo():
    bajo, _ = metricas.intervalo_bootstrap([0.05] * 17, 2000, 0.95, 1)
    assert bajo > 0


def test_el_bootstrap_remuestrea_unidades_no_frames():
    # Con 17 valores, cada replica tiene 17 elementos: si remuestreara frames
    # el intervalo saldria mucho mas estrecho.
    valores = np.array([0.0] * 8 + [1.0] * 9)
    bajo, alto = metricas.intervalo_bootstrap(valores, 5000, 0.95, 1)
    assert alto - bajo > 0.3


# -- 6.5 como se reporta la calidad -----------------------------------------


def test_la_curva_empieza_arriba_y_acaba_abajo():
    curva = metricas.curva_de_umbrales([0.2, 0.5, 0.9])
    assert curva["umbral"].iloc[0] == 0.0
    assert curva["umbral"].iloc[-1] == 1.0
    assert curva["fraccion"].iloc[0] == pytest.approx(1.0)
    assert curva["fraccion"].iloc[-1] == pytest.approx(0.0)


def test_la_curva_no_sube_nunca():
    curva = metricas.curva_de_umbrales([0.1, 0.4, 0.4, 0.75, 0.9])
    assert (np.diff(curva["fraccion"].to_numpy()) <= 0).all()


def test_el_area_bajo_la_curva_es_exactamente_la_media():
    for valores in ([0.6], [0.4, 0.8], [0.1, 0.25, 0.5, 0.5, 0.99]):
        curva = metricas.curva_de_umbrales(valores)
        assert metricas.area_bajo_curva(curva) == pytest.approx(np.mean(valores))


def test_una_curva_por_encima_de_otra_en_todo_el_rango_gana_pase_lo_que_pase():
    mejor = metricas.curva_de_umbrales([0.5, 0.6, 0.7])
    peor = metricas.curva_de_umbrales([0.2, 0.3, 0.4])
    for umbral in (0.1, 0.25, 0.45, 0.55, 0.65):
        alta = metricas.fraccion_que_supera([0.5, 0.6, 0.7], umbral)
        baja = metricas.fraccion_que_supera([0.2, 0.3, 0.4], umbral)
        assert alta >= baja
    assert metricas.area_bajo_curva(mejor) > metricas.area_bajo_curva(peor)


def test_los_catastroficos_son_el_punto_de_la_misma_curva():
    valores = [0.02, 0.05, 0.4, 0.6, 0.9]
    umbral = config.UMBRAL_CATASTROFICO
    assert metricas.fraccion_que_supera(valores, umbral) == pytest.approx(0.6)
    assert metricas.fraccion_catastrofica(valores, umbral) == pytest.approx(0.4)


def test_la_media_alta_puede_perder_contra_la_que_nunca_se_rompe():
    # El ejemplo del apartado 6.5: 0.55 que nunca baja de 0.45, contra 0.60 que
    # se rompe uno de cada diez frames.
    estable = [0.55] * 9 + [0.45]
    irregular = [0.65] * 9 + [0.05]
    assert np.mean(irregular) > np.mean(estable)
    umbral = config.UMBRAL_CATASTROFICO
    assert metricas.fraccion_catastrofica(estable, umbral) == 0.0
    assert metricas.fraccion_catastrofica(irregular, umbral) == pytest.approx(0.1)


def test_se_reportan_las_dos_curvas(tabla_desigual):
    dos = metricas.curvas(tabla_desigual)
    assert set(dos) == {"unidad", "frame"}


def test_el_area_de_la_curva_por_unidad_es_el_numero_de_la_tabla(tabla_desigual):
    dos = metricas.curvas(tabla_desigual)
    esperada = metricas.media_entre_unidades(metricas.por_unidad(tabla_desigual))
    assert metricas.area_bajo_curva(dos["unidad"]) == pytest.approx(esperada)
    assert metricas.area_bajo_curva(dos["unidad"]) == pytest.approx(0.6)


def test_el_area_de_la_curva_por_frame_no_es_el_numero_de_la_tabla(tabla_desigual):
    dos = metricas.curvas(tabla_desigual)
    assert metricas.area_bajo_curva(dos["frame"]) == pytest.approx(0.36)
    assert metricas.area_bajo_curva(dos["frame"]) != pytest.approx(
        metricas.area_bajo_curva(dos["unidad"])
    )


def test_que_las_dos_curvas_discrepen_dice_donde_estan_los_fallos():
    # Mismos 20 frames y misma media por unidad en los dos casos, pero en uno
    # el fallo esta concentrado en una lesion entera y en el otro repartido.
    concentrado = pd.DataFrame(
        [{"serie": 1, "etiqueta": 1, "dice": 0.05} for _ in range(10)]
        + [{"serie": 2, "etiqueta": 1, "dice": 0.85} for _ in range(10)]
    )
    repartido = pd.DataFrame(
        [
            {"serie": s, "etiqueta": 1, "dice": d}
            for s in (1, 2)
            for d in [0.05] * 5 + [0.85] * 5
        ]
    )
    umbral = config.UMBRAL_CATASTROFICO
    unidades_concentrado = metricas.por_unidad(concentrado).to_numpy()
    unidades_repartido = metricas.por_unidad(repartido).to_numpy()

    # Por frame las dos son identicas: mitad de frames por debajo de 0.1.
    assert metricas.fraccion_catastrofica(concentrado["dice"], umbral) == pytest.approx(
        metricas.fraccion_catastrofica(repartido["dice"], umbral)
    )
    # Por unidad se separan: una pierde media lesion, la otra ninguna.
    assert metricas.fraccion_catastrofica(unidades_concentrado, umbral) == 0.5
    assert metricas.fraccion_catastrofica(unidades_repartido, umbral) == 0.0


def test_la_curva_rechaza_valores_fuera_de_rango():
    with pytest.raises(ValueError, match="entre 0 y 1"):
        metricas.curva_de_umbrales([0.5, 1.4])
