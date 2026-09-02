"""Todas las comparaciones pareadas del trabajo, en una sola tabla.

**Es la otra mitad de los resultados.** `resumen.csv` lleva una fila por
checkpoint medido; esto lleva una fila por **comparacion**: M1 contra M0, M3
contra M1, M4 contra M1, S1 contra S0 tanto en los modelos como en el filtro
clasico, el filtro contra cada modelo y el techo humano contra el modelo. Hasta
ahora esos numeros solo existian impresos por los scripts de tabla, asi que para
citar uno habia que volver a ejecutar un notebook.

Todas se calculan igual, con `metricas.contraste`: diferencia pareada unidad a
unidad, intervalo por bootstrap sobre unidades y cuantas gana. No hay ningun
numero nuevo aqui, solo reunidos.

    conda run -n surco python scripts/contrastes.py
"""

import pandas as pd

from surco import config, metricas, resultados

# Las dos mitades de S1, que el apartado 5.5 mide por separado.
VARIANTES_SELECTOR = {"s1": "S1", "puerta": "solo puerta", "ranking": "solo ranking"}

# El apartado va con `A` delante y no como "5.1" a secas por un motivo practico:
# "5.1" y "7" son numeros validos, asi que pandas leeria la columna como float y
# el apartado 7 volveria como 7.0. Con la letra nunca deja de ser texto.
APARTADOS = {"modo": "A5.1", "prompt": "A5.5", "selector": "A5.5",
             "clasico": "A5.4", "memoria": "A5.1", "concordancia": "A7",
             "cribado": "A5.3"}


# Que tabla de `fuentes()` resuelve cada patron de `resultados`. La
# correspondencia hace falta solo aqui, porque este script lee los CSV una vez
# al principio en vez de por celda.
TABLA_DE = {
    "profundidad_surco-*.csv": "rejilla",
    "selector.csv": "selector",
    "m4_surco-*.csv": "m4",
    "clasicos.csv": "clasicos",
    "selector_clasico.csv": "selector_clasico",
}


def fuentes() -> dict:
    return {
        "rejilla": resultados.unir("profundidad_surco-*.csv"),
        "m4": resultados.unir("m4_surco-*.csv"),
        "selector": pd.read_csv(config.DIR_PROCESO / "selector.csv"),
        "clasicos": pd.read_csv(config.DIR_PROCESO / "clasicos.csv"),
        "selector_clasico": pd.read_csv(config.DIR_PROCESO / "selector_clasico.csv"),
        "cribado": resultados.unir("cribado_surco-*.csv"),
        "m3_suelto": pd.read_csv(config.DIR_PROCESO / "m3_suelto_medsam2.csv"),
        "m1_suelto": pd.read_csv(config.DIR_PROCESO / "m1_suelto_medsam2.csv"),
        "tres_vias": pd.read_csv(config.DIR_PROCESO / "concordancia_tres_vias.csv"),
        "tercero": pd.read_csv(config.DIR_PROCESO / "concordancia_tercer_anotador.csv"),
    }


def _fila(apartado: str, factor: str, a: str, b: str, uno, otro) -> dict | None:
    """Una comparacion, o None si a alguno de los dos lados le faltan datos."""
    if uno is None or otro is None or uno.empty or otro.empty:
        return None
    if set(uno.index) != set(otro.index):
        return None
    contra = metricas.contraste(uno, otro)
    return {
        "apartado": apartado,
        "factor": factor,
        "a": a,
        "b": b,
        **contra.como_fila(),
        "separado": contra.separado,
    }


def de_modo(f: dict) -> list[dict]:
    """El factor modo del apartado 5.1: que aporta cada forma de usar el tiempo."""
    filas = []
    for modelo in sorted(f["rejilla"]["modelo"].unique()):
        for protocolo in config.PROTOCOLOS:
            def en(tabla, modo, columna="dice", **extra):
                return metricas.celda(
                    tabla, columna, modelo=modelo, protocolo=protocolo,
                    modo=modo, **extra
                )

            parejas = [
                ("M1", "M0", en(f["rejilla"], "M1"), en(f["rejilla"], "M0"), "S0"),
                ("M3", "M1", en(f["rejilla"], "M3"), en(f["rejilla"], "M1"), "S0"),
                ("M4", "M1", en(f["m4"], "M4", selector="S0"),
                 en(f["rejilla"], "M1"), "S0"),
                ("M4", "M3", en(f["m4"], "M4", selector="S0"),
                 en(f["rejilla"], "M3"), "S0"),
                ("M4", "M1", en(f["m4"], "M4", selector="S1"),
                 metricas.celda(f["selector"], "dice_s1", modelo=modelo,
                                protocolo=protocolo, modo="M1"), "S1"),
            ]
            for arriba, abajo, uno, otro, sel in parejas:
                filas.append(
                    _fila(APARTADOS["modo"], "modo",
                          f"{modelo} {arriba} x {protocolo} x {sel}",
                          f"{modelo} {abajo} x {protocolo} x {sel}", uno, otro)
                )
    return filas


def de_prompt(f: dict) -> list[dict]:
    """El factor prompt: cuantos clics hacen falta, con el modo fijo en M1."""
    filas = []
    for modelo in sorted(f["rejilla"]["modelo"].unique()):
        base = metricas.celda(f["rejilla"], modelo=modelo, modo="M1", protocolo="P1")
        for protocolo in list(config.PROTOCOLOS)[1:]:
            filas.append(
                _fila(APARTADOS["prompt"], "prompt",
                      f"{modelo} M1 x {protocolo}", f"{modelo} M1 x P1",
                      metricas.celda(f["rejilla"], modelo=modelo, modo="M1",
                                     protocolo=protocolo), base)
            )
    return filas


def de_selector(f: dict) -> list[dict]:
    """El factor selector, y sus dos mitades por separado.

    Se saltan los modelos con una sola candidata: alli S0 y S1 coinciden por
    construccion y la comparacion no tiene dos niveles (apartado 5.5).
    """
    tabla = f["selector"]
    degenerados = tabla.groupby("modelo")["n_candidatas"].max()
    vivos = sorted(degenerados[degenerados > 1].index)
    filas = []
    for modelo in vivos:
        for modo in sorted(tabla["modo"].unique()):
            for protocolo in config.PROTOCOLOS:
                comun = {"modelo": modelo, "modo": modo, "protocolo": protocolo}
                base = metricas.celda(tabla, "dice_s0", **comun)
                for columna, etiqueta in VARIANTES_SELECTOR.items():
                    filas.append(
                        _fila(APARTADOS["selector"], "selector",
                              f"{modelo} {modo} x {protocolo} x {etiqueta}",
                              f"{modelo} {modo} x {protocolo} x S0",
                              metricas.celda(tabla, f"dice_{columna}", **comun), base)
                    )
    return filas


# Las dos parejas del 5.5 sobre el clasico. Las dos van contra **el mismo lado
# derecho**, las tres candidatas con S0, y por eso cada una aisla una cosa: el
# criterio de ordenacion la primera, y trocear la respuesta en tres niveles la
# segunda. Contra la mascara unica, S1 mediria las dos a la vez, que es la
# lectura que el 5.4 advierte que no hay que hacer.
VARIANTES_CLASICO = (
    ("selector sobre el clasico", "S1", "dice_s1"),
    ("candidatas sobre el clasico", "una mascara", "dice_una"),
)


def de_variantes_del_clasico(f: dict) -> list[dict]:
    """Las dos parejas del apartado 5.5 sobre el filtro de cresta.

    Solo sale el filtro de cresta porque es el unico que admite la variante: sus
    tres escalas de sigma dan tres respuestas, mientras que el umbral local y el
    watershed producen una sola mascara.
    """
    tabla = f["selector_clasico"]
    filas = []
    for metodo in sorted(tabla["modelo"].unique()):
        for protocolo in config.PROTOCOLOS:
            comun = {"modelo": metodo, "protocolo": protocolo}
            base = metricas.celda(tabla, "dice_s0", **comun)
            for factor, etiqueta, columna in VARIANTES_CLASICO:
                filas.append(
                    _fila(APARTADOS["selector"], factor,
                          f"{metodo} x {protocolo} x {etiqueta}",
                          f"{metodo} x {protocolo} x S0",
                          metricas.celda(tabla, columna, **comun), base)
                )
    return filas


def de_clasicos(f: dict) -> list[dict]:
    """La hipotesis nula del apartado 5.4, a protocolo fijo contra la fila M1.

    El nombre del clasico lleva la variante porque **hay tres numeros distintos
    detras de `sato x P3`**: la mascara unica, y las tres candidatas con cada
    selector. Estas 48 filas son la de mascara unica, que es como el metodo
    compite en el 5.4.
    """
    filas = []
    for metodo in config.METODOS_CLASICOS:
        for modelo in sorted(f["rejilla"]["modelo"].unique()):
            for protocolo in config.PROTOCOLOS:
                filas.append(
                    _fila(APARTADOS["clasico"], "clasico contra SAM",
                          f"{metodo} x {protocolo} x una mascara",
                          f"{modelo} M1 x {protocolo}",
                          metricas.celda(f["clasicos"], modelo=metodo,
                                         protocolo=protocolo),
                          metricas.celda(f["rejilla"], modelo=modelo, modo="M1",
                                         protocolo=protocolo))
                )
    return filas


def de_memoria(f: dict) -> list[dict]:
    """M3 contra M1 en los dos modelos con banco de memoria, y sus aportes.

    Es la unica comparacion del conjunto que separa si lo que aporta la memoria
    depende del dominio de ajuste o de la mecanica: SAM 2.1 y MedSAM2 son el
    mismo modelo y su fork.
    """
    filas = []
    for protocolo in config.PROTOCOLOS:
        de_sam21 = (
            metricas.celda(f["rejilla"], modelo="sam21", modo="M3", protocolo=protocolo),
            metricas.celda(f["rejilla"], modelo="sam21", modo="M1", protocolo=protocolo),
        )
        m1_medsam2 = (
            metricas.por_unidad(f["cribado"][f["cribado"]["checkpoint"] == "2411"])
            if protocolo == "P1"
            else metricas.celda(f["m1_suelto"], protocolo=protocolo)
        )
        de_medsam2 = (metricas.celda(f["m3_suelto"], protocolo=protocolo), m1_medsam2)
        for nombre, (uno, otro) in (("sam21", de_sam21), ("medsam2", de_medsam2)):
            filas.append(
                _fila(APARTADOS["modo"], "memoria",
                      f"{nombre} M3 x {protocolo}", f"{nombre} M1 x {protocolo}",
                      uno, otro)
            )
        filas.append(
            _fila(APARTADOS["modo"], "memoria contra dominio",
                  f"aporte de medsam2 en {protocolo}",
                  f"aporte de sam21 en {protocolo}",
                  metricas.diferencia_pareada(*de_medsam2),
                  metricas.diferencia_pareada(*de_sam21))
        )
    return filas


def _unidades_de_celda_sam(f: dict, celda: tuple) -> pd.Series:
    """Las 17 medias por unidad de una celda de modelo, de la fuente que toque."""
    patron, columna, filtros = resultados.fuente_de_celda(celda)
    return metricas.celda(f[TABLA_DE[patron]], columna, **filtros)


def _unidades_de_celda_clasica(f: dict, celda: tuple) -> pd.Series:
    patron, columna, filtros = resultados.fuente_de_celda_clasica(celda)
    return metricas.celda(f[TABLA_DE[patron]], columna, **filtros)


def de_techos(f: dict) -> list[dict]:
    """Techo contra techo: lo mejor de cada lado, en las cuatro combinaciones.

    El 5.4 cierra con esta comparacion y sus numeros solo vivian en prosa, que es
    el mismo defecto que `resultados` existe para evitar. Aqui los techos **no se
    escriben a mano**: se le preguntan a ese modulo, que es el unico que sabe que
    las celdas de modelo viven en tres ficheros y las del clasico en dos.

    Cuatro filas y no una, porque cada lado tiene dos techos (SAM con S0 y con
    S1, el clasico con una mascara y con S1) y la pregunta cambia segun cual se
    compare con cual. Publicar solo la pareja que convenga seria elegir la
    conclusion.
    """
    de_modelos = resultados.celdas_de_modelos()
    de_clasicos = resultados.celdas_de_clasicos()
    filas = []
    for selector in config.SELECTORES:
        celda_sam, _ = resultados.mejor(
            {k: v for k, v in de_modelos.items() if k[3] == selector}
        )
        for variante in ("una mascara", "S1"):
            celda_cl, _ = resultados.mejor(
                {k: v for k, v in de_clasicos.items() if k[2] == variante}
            )
            filas.append(
                _fila(APARTADOS["clasico"], "techo contra techo",
                      f"{celda_cl[0]} x {celda_cl[1]} x {celda_cl[2]}",
                      f"{celda_sam[0]} {celda_sam[1]} x {celda_sam[2]} x {celda_sam[3]}",
                      _unidades_de_celda_clasica(f, celda_cl),
                      _unidades_de_celda_sam(f, celda_sam))
            )
    return filas


def _representantes(cribado: pd.DataFrame) -> pd.DataFrame:
    """El checkpoint con mejor Dice de cada modelo, que es la regla del 6.3."""
    por_celda = (
        cribado.groupby(["modelo", "checkpoint", "serie", "etiqueta"])["dice"]
        .mean()
        .groupby(level=["modelo", "checkpoint"])
        .mean()
        .reset_index()
    )
    mejores = por_celda.loc[por_celda.groupby("modelo")["dice"].idxmax()]
    return mejores.set_index("modelo").sort_values("dice", ascending=False)


def de_cribado(f: dict) -> list[dict]:
    """Lo que la eleccion de k = 4 del apartado 5.3 tiene que sobrevivir.

    El 5.3 elige un modelo por familia y no los cuatro de mayor Dice, asi que
    deja fuera a modelos que en el cribado quedaron **por delante de alguno de
    los elegidos**. Esas son las comparaciones incomodas, y por tanto las que hay
    que publicar.

    La regla, y no una lista a mano: entra el representante de **todo** modelo no
    elegido que supere a alguno de los cuatro, contra **cada** uno de los cuatro.
    Escoger cuales se publican seria seleccionar por conveniencia.

    Y se publican como lo que son: diferencias pareadas con su intervalo. Las
    medias del cribado ordenan la tabla, pero no dicen si dos modelos se separan.
    """
    mejores = _representantes(f["cribado"])
    elegidos = config.REJILLA_PROFUNDIDAD
    suelo = min(mejores.loc[m, "dice"] for m in elegidos)

    def unidades(modelo: str) -> pd.Series:
        return metricas.celda(
            f["cribado"], modelo=modelo, checkpoint=mejores.loc[modelo, "checkpoint"]
        )

    filas = []
    for modelo, fila in mejores.iterrows():
        if modelo in elegidos or fila["dice"] <= suelo:
            continue
        for elegido in elegidos:
            filas.append(
                _fila(APARTADOS["cribado"], "no elegido contra elegido",
                      f"{modelo} {mejores.loc[modelo, 'checkpoint']} en el cribado",
                      f"{elegido} {mejores.loc[elegido, 'checkpoint']} en el cribado",
                      unidades(modelo), unidades(elegido))
            )
    return filas


def de_concordancia(f: dict) -> list[dict]:
    """El apartado 7: quien concuerda con quien, y el modelo como tercer anotador."""
    filas = []
    for estrato, grupo in f["tres_vias"].groupby("estrato"):
        for una, otra in (("a1_a2", "a1_gt"), ("a1_a2", "a2_gt")):
            filas.append(
                _fila(APARTADOS["concordancia"], "acuerdo humano", f"{una} ({estrato})",
                      f"{otra} ({estrato})", grupo[una], grupo[otra])
            )
    for estrato, grupo in f["tercero"].groupby("estrato"):
        for columna in ("modelo_a1", "modelo_a2"):
            filas.append(
                _fila(APARTADOS["concordancia"], "modelo como anotador", f"{columna} ({estrato})",
                      f"a1_a2 ({estrato})", grupo[columna], grupo["a1_a2"])
            )
    return filas


def main() -> None:
    pd.set_option("display.width", 220)
    f = fuentes()
    filas = (
        de_modo(f) + de_prompt(f) + de_selector(f) + de_variantes_del_clasico(f)
        + de_clasicos(f) + de_techos(f) + de_memoria(f) + de_cribado(f)
        + de_concordancia(f)
    )
    tabla = pd.DataFrame([fila for fila in filas if fila is not None])
    tabla = tabla.round({"diferencia": 4, "ic_bajo": 4, "ic_alto": 4})
    destino = config.DIR_SALIDAS / "contrastes.csv"
    tabla.to_csv(destino, index=False)

    print(f"{len(tabla)} contrastes en {destino}")
    print(tabla.groupby(["apartado", "factor"])["separado"].agg(["count", "sum"])
          .rename(columns={"count": "contrastes", "sum": "fuera del cero"})
          .to_string())
    print("\n=== LOS DIEZ MAYORES A FAVOR DE A, CON EL INTERVALO FUERA DEL CERO ===")
    print(
        tabla[tabla["separado"]]
        .nlargest(10, "diferencia")
        .to_string(index=False, float_format=lambda v: f"{v:7.4f}")
    )


if __name__ == "__main__":
    main()
