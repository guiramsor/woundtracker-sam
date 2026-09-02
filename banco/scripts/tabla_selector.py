"""La tabla del factor selector: que aporta S1 sobre S0 (apartado 5.5).

Pareado unidad a unidad, que es lo natural aqui: las dos columnas salen del
**mismo conjunto de candidatas**, asi que comparten modelo, prompt, frame y
lesion, y lo unico que cambia es cual se elige.

Se reporta ademas lo que el apartado pregunta expresamente: **si el selector
sigue aportando cuando el prompt ya lleva clics de fondo**. La puerta de S1
tiene dos guardas en P2 y P4 y una sola en P1 y P3, asi que la comparacion entre
protocolos es la respuesta.

**Sobre la validacion dejando una serie fuera.** El S1 reformulado del apartado
5.5 **no tiene ningun umbral libre**: la puerta es contener el clic 0 y no
contener negativos, el orden es z y el desempate elongacion. Con `FRAC_MAX` cayo
el unico parametro que habia que calibrar. Asi que las nueve rondas no calibran
nada, y no se presentan como si lo hicieran: se usan para lo unico que aqui
significan algo, ver si lo que aporta S1 es consistente serie a serie o sale de
una sola.

    conda run -n surco python scripts/tabla_selector.py
"""

import pandas as pd

from surco import config, metricas


# S1 entero y sus dos mitades. `puerta` filtra y deja mandar al score del
# modelo; `ranking` ordena por z sin filtrar. Con el respaldo actuando en buena
# parte de los prompts de P4, sin separarlas no se sabe cual sostiene el
# resultado.
VARIANTES = ("s1", "puerta", "ranking")


def _contraste(grupo: pd.DataFrame, columna: str = "s1") -> metricas.Contraste:
    """Una variante del selector contra S0, sobre las mismas candidatas."""
    return metricas.contraste(
        metricas.por_unidad(grupo, f"dice_{columna}"),
        metricas.por_unidad(grupo, "dice_s0"),
    )


def por_celda(tabla: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for (modelo, modo, protocolo), grupo in tabla.groupby(
        ["modelo", "modo", "protocolo"]
    ):
        contra_s0 = _contraste(grupo)
        filas.append(
            {
                "modelo": modelo,
                "modo": modo,
                "protocolo": protocolo,
                "dice_s0": metricas.media_entre_unidades(
                    metricas.por_unidad(grupo, "dice_s0")
                ),
                "dice_s1": metricas.media_entre_unidades(
                    metricas.por_unidad(grupo, "dice_s1")
                ),
                # El nombre de la columna se conserva: `tabla_selector.csv` ya
                # esta reportado y `por_protocolo` agrega por el.
                "s1_menos_s0": contra_s0.diferencia,
                "ic_bajo": contra_s0.ic_bajo,
                "ic_alto": contra_s0.ic_alto,
                "ganadas": contra_s0.ganadas,
                "coincide": float(grupo["coincide"].mean()),
                "respaldo": float(grupo["respaldo"].mean()),
            }
        )
    return pd.DataFrame(filas)


def degenerados(tabla: pd.DataFrame) -> list[str]:
    """Modelos que proponen una sola candidata: ahi S0 y S1 son la misma cosa.

    Es el mismo caso que M3 y se trata igual (apartado 5.5): la celda no tiene
    dos niveles, tiene uno. Sus ceros no son "S1 no aporta", son "no hay nada
    que elegir", y promediarlos con los demas diluiria el factor con filas que
    no pueden moverse.
    """
    por_modelo = tabla.groupby("modelo")["n_candidatas"].max()
    return sorted(por_modelo[por_modelo == 1].index)


def por_protocolo(celdas: pd.DataFrame) -> pd.DataFrame:
    """La pregunta del apartado: si el clic de fondo deja sin trabajo a S1."""
    return celdas.groupby("protocolo").agg(
        celdas=("s1_menos_s0", "count"),
        s1_menos_s0=("s1_menos_s0", "mean"),
        coincide=("coincide", "mean"),
        respaldo=("respaldo", "mean"),
    )


def descomposicion(tabla: pd.DataFrame) -> pd.DataFrame:
    """S1 entero contra cada una de sus dos mitades, protocolo a protocolo."""
    filas = []
    for protocolo, grupo in tabla.groupby("protocolo"):
        fila = {"protocolo": protocolo, "respaldo": float(grupo["respaldo"].mean())}
        for variante in VARIANTES:
            contra_s0 = _contraste(grupo, variante)
            fila[variante] = contra_s0.diferencia
            fila[f"{variante}_ic"] = (
                f"[{contra_s0.ic_bajo:+.4f}, {contra_s0.ic_alto:+.4f}]"
            )
        filas.append(fila)
    return pd.DataFrame(filas)


def dejando_una_serie_fuera(tabla: pd.DataFrame) -> pd.DataFrame:
    """Lo que aporta S1 en cada serie, evaluado solo sobre esa serie.

    No calibra nada, porque no hay nada que calibrar. Dice si el aporte es
    consistente o si sale de una sola serie, que con n = 17 es la pregunta que
    de verdad se puede contestar.
    """
    filas = []
    for serie in sorted(tabla["serie"].unique()):
        contra_s0 = _contraste(tabla[tabla["serie"] == serie])
        filas.append(
            {
                "serie_fuera": int(serie),
                "unidades": contra_s0.n,
                "s1_menos_s0": contra_s0.diferencia,
                "ganadas": contra_s0.ganadas,
            }
        )
    return pd.DataFrame(filas)


def main() -> None:
    pd.set_option("display.width", 200)
    tabla = pd.read_csv(config.DIR_PROCESO / "selector.csv")
    print(f"{len(tabla)} filas, {tabla['modelo'].nunique()} modelos")

    sin_eleccion = degenerados(tabla)
    if sin_eleccion:
        print(
            f"\n  celda degenerada, una sola candidata: {', '.join(sin_eleccion)}.\n"
            "  S0 y S1 coinciden por construccion, como en M3. Fuera de los\n"
            "  agregados del factor, y declarado en vez de mostrado como un cero."
        )
    vivas = tabla[~tabla["modelo"].isin(sin_eleccion)]

    celdas = por_celda(vivas)
    print("\n=== S1 MENOS S0, PAREADO POR UNIDAD, n = 17 ===")
    print(celdas.to_string(index=False, float_format=lambda v: f"{v:7.4f}"))

    separadas = celdas[(celdas["ic_bajo"] > 0) | (celdas["ic_alto"] < 0)]
    print(
        f"\n  celdas cuyo intervalo no cruza el cero: {len(separadas)} de {len(celdas)}"
    )
    if not separadas.empty:
        print(separadas.to_string(index=False, float_format=lambda v: f"{v:7.4f}"))

    print("\n=== SIGUE APORTANDO CON CLICS DE FONDO? ===")
    print(por_protocolo(celdas).to_string(float_format=lambda v: f"{v:7.4f}"))
    print(
        "  coincide = fraccion de prompts en que S1 elige lo mismo que S0.\n"
        "  respaldo = fraccion en que no sobrevive ninguna candidata a la puerta."
    )

    print("\n=== LAS DOS MITADES DE S1, POR SEPARADO (contra S0) ===")
    print(
        descomposicion(vivas).to_string(index=False, float_format=lambda v: f"{v:7.4f}")
    )
    print(
        "  puerta  = filtra por las guardas y despues manda el score del modelo.\n"
        "  ranking = ordena por z sin filtrar, que es tambien lo que hace el respaldo."
    )

    print("\n=== DEJANDO UNA SERIE FUERA (no calibra: S1 no tiene umbrales) ===")
    print(
        dejando_una_serie_fuera(vivas).to_string(
            index=False, float_format=lambda v: f"{v:7.4f}"
        )
    )

    # No se guarda: es un agregado de `selector.csv`, y `F10_selector.ipynb` lo
    # recalcula llamando a `por_celda()`.


if __name__ == "__main__":
    main()
