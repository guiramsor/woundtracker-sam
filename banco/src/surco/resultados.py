"""De donde sale cada Dice medido, y cual es el mejor de todos.

**Este modulo existe por un defecto que aparecio cinco veces.** Los resultados
del trabajo no estan en un fichero sino en tres, y cada uno cubre una parte de la
rejilla:

- `profundidad_surco-*.csv`: M0, M1 y M3, siempre con **S0**;
- `selector.csv`: los mismos M0 y M1 con **S1**, reordenando las candidatas
  guardadas;
- `m4_surco-*.csv`: **M4**, que se midio dentro del entorno del modelo y trae sus
  dos selectores, porque alli cada uno es una cadena distinta.

Quien quiera saber cual es la mejor celda tiene que mirar las tres. Cada vez que
alguien miro solo una, el resultado salio sesgado sin que nada avisara: el
apartado 7 comparo el techo humano contra la segunda mejor configuracion, y el
5.4 enfrento el filtro clasico en su mejor celda contra SAM en una que ya no lo
era. La respuesta no es acordarse cada vez, es preguntarselo a este modulo.

Le preguntan `verificacion`, `clasicos_contra_sam`, `resumen`, `figuras` y el
test de `MEJOR_CELDA`. Los dos ultimos entraron despues, cuando el mismo defecto
aparecio del otro lado: el mejor **clasico** tambien sale de dos fuentes, porque
`clasicos.csv` solo tiene la variante de una mascara y la de tres candidatas con
selector esta en `selector_clasico.csv`.

**Y `unir()` es la unica implementacion de "junta los CSV por entorno".** Llego a
estar copiada en ocho sitios mas: `verificacion`, `contrastes`, `resumen`,
`clasicos_contra_sam`, `concordancia_entre_anotadoras` y las tres tablas. Once
expresiones en total, porque tres de esos sitios unen dos patrones cada uno.

Y no eran ocho copias iguales, que es lo que las hacia peligrosas: **tres no
filtraban los caidos** -las de `m4_surco-*.csv` en `tabla_m4`, `resumen` y
`concordancia_entre_anotadoras`-. Sobre los ficheros de hoy da igual, porque no
existe ningun `m4_caidos_*.csv` y esta comprobado que las once expresiones
devuelven exactamente las mismas filas. Lo que no da igual es el dia que un
entorno de M4 se caiga a la mitad: entonces tres sitios meterian filas fallidas
en una media y cinco no, y el desacuerdo saldria como un numero raro y no como un
error. Ahora la regla la sabe uno.
"""

from pathlib import Path

import pandas as pd

from surco import config, metricas

# Que columna de Dice y que selector aporta cada fuente.
FUENTES = (
    ("profundidad_surco-*.csv", "dice", "S0"),
    ("selector.csv", "dice_s1", "S1"),
    ("m4_surco-*.csv", "dice", None),  # el selector viene en su propia columna
)


def ficheros_de(patron: str) -> list[Path]:
    """Los CSV que casan con el patron, sin los de filas no medidas.

    Va aparte de `unir` porque dos de las tablas imprimen **cuantos entornos**
    han leido, y ese recuento tiene que salir de la misma lista que la union. Con
    cada una haciendo su glob, la del recuento y la de los datos pueden dejar de
    coincidir sin que nada avise.

    Un `*_caidos_*.csv` es la lista de lo que fallo, no un resultado. Hoy ninguno
    de los patrones en uso llega a casar con uno -`cribado_surco-*.csv` no coge
    `cribado_caidos_surco-tinysam.csv`, porque despues de `cribado_` viene otra
    cosa-, asi que el filtro es una red y no un arreglo. Se queda por lo que
    costaria el dia que un patron se amplie: filas fallidas dentro de una media.
    """
    ficheros = [
        f for f in sorted(config.DIR_PROCESO.glob(patron)) if "caidos" not in f.name
    ]
    if not ficheros:
        raise FileNotFoundError(f"no hay ningun {patron} en salidas/proceso/")
    return ficheros


def unir(patron: str) -> pd.DataFrame:
    """Une los CSV que casen con el patron, saltando los de filas no medidas."""
    if "*" not in patron:
        return pd.read_csv(config.DIR_PROCESO / patron)
    return pd.concat(
        [pd.read_csv(f) for f in ficheros_de(patron)], ignore_index=True
    )


def fuente_de_celda(celda: tuple) -> tuple[str, str, dict]:
    """De que fichero, que columna y con que filtro se lee una celda de modelo.

    Es `FUENTES` visto desde el otro lado: no "que hay en cada fichero" sino
    "a esta celda cual le toca". Vive aqui porque llego a estar escrita entera
    en dos sitios -las figuras del 6.5 y las comparaciones pareadas-, que es la
    misma cadena de tres ramas que hay que tocar el dia que aparezca una cuarta
    fuente.
    """
    modelo, modo, protocolo, selector = celda
    filtros = {"modelo": modelo, "modo": modo, "protocolo": protocolo}
    if modo == "M4":
        return "m4_surco-*.csv", "dice", {**filtros, "selector": selector}
    if selector == "S1":
        return "selector.csv", "dice_s1", filtros
    return "profundidad_surco-*.csv", "dice", filtros


def fuente_de_celda_clasica(celda: tuple) -> tuple[str, str, dict]:
    """Lo mismo para los clasicos, que tienen dos fuentes en vez de tres."""
    metodo, protocolo, variante = celda
    filtros = {"modelo": metodo, "protocolo": protocolo}
    if variante == "una mascara":
        return "clasicos.csv", "dice", filtros
    return "selector_clasico.csv", f"dice_{variante.lower()}", filtros


def celdas_de_modelos() -> dict[tuple[str, str, str, str], float]:
    """Dice medio entre unidades de **cada celda medida**, de las tres fuentes.

    La clave es `(modelo, modo, protocolo, selector)`, que es lo que identifica
    una celda sin ambiguedad.
    """
    celdas: dict[tuple[str, str, str, str], float] = {}
    for patron, columna, selector_fijo in FUENTES:
        tabla = unir(patron)
        claves = ["modelo", "modo", "protocolo"]
        if selector_fijo is None:
            claves.append("selector")
        for valores, grupo in tabla.groupby(claves):
            selector = selector_fijo or valores[3]
            celdas[(valores[0], valores[1], valores[2], selector)] = (
                metricas.media_entre_unidades(metricas.por_unidad(grupo, columna))
            )
    return celdas


def mejor(celdas: dict) -> tuple[tuple, float]:
    """La celda con mas Dice, desempatando **a favor de S0**.

    El desempate importa cuando un modelo propone una sola candidata: alli S0 y
    S1 dan literalmente lo mismo, y sin esto saldria S1 por el orden alfabetico
    de la etiqueta, sugiriendo que el selector aporto algo que no podia aportar.
    """
    return max(celdas.items(), key=lambda par: (round(par[1], 6), par[0][-1] == "S0"))


def mejor_celda_de_modelo() -> dict:
    """La mejor celda del trabajo, mirando las tres fuentes."""
    (modelo, modo, protocolo, selector), dice = mejor(celdas_de_modelos())
    return {
        "modelo": modelo,
        "modo": modo,
        "protocolo": protocolo,
        "selector": selector,
        "dice": dice,
    }


def celdas_de_clasicos() -> dict[tuple[str, str, str], float]:
    """Lo mismo para los tres metodos clasicos, que tienen dos fuentes.

    `clasicos.csv` los mide con una sola mascara, que es como compiten en el
    apartado 5.4; `selector_clasico.csv` mide el filtro de cresta con tres
    candidatas y los dos selectores encima.
    """
    celdas: dict[tuple[str, str, str], float] = {}
    de_una = unir("clasicos.csv")
    for (metodo, protocolo), grupo in de_una.groupby(["modelo", "protocolo"]):
        celdas[(metodo, protocolo, "una mascara")] = metricas.media_entre_unidades(
            metricas.por_unidad(grupo)
        )
    ruta = config.DIR_PROCESO / "selector_clasico.csv"
    if ruta.exists():
        con_selector = pd.read_csv(ruta)
        for (metodo, protocolo), grupo in con_selector.groupby(["modelo", "protocolo"]):
            for columna, etiqueta in (("dice_s0", "S0"), ("dice_s1", "S1")):
                celdas[(metodo, protocolo, etiqueta)] = metricas.media_entre_unidades(
                    metricas.por_unidad(grupo, columna)
                )
    return celdas


def mejor_celda_de_clasico() -> dict:
    """El mejor metodo clasico, mirando sus dos fuentes."""
    (metodo, protocolo, variante), dice = mejor(celdas_de_clasicos())
    return {
        "metodo": metodo,
        "protocolo": protocolo,
        "variante": variante,
        "dice": dice,
    }
