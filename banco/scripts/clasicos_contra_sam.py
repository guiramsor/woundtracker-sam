"""La hipotesis nula, contrastada: los clasicos contra SAM (apartado 5.4).

Pareado unidad a unidad y **a protocolo fijo**, que es la unica forma de que la
comparacion signifique algo: darle cinco clics a uno y uno al otro no compara
metodos, compara prompts.

Contra la fila **M1** de la rejilla, porque los clasicos son de clic fijo: se
vuelven a aplicar en cada frame con los mismos clics y sin memoria.

Y aparte, contra **la mejor celda de toda la rejilla**, sea del modo y protocolo
que sea. Esa comparacion no es pareada en prompt y se etiqueta como lo que es:
el mejor SAM del banco contra el mejor clasico, para saber si el techo de SAM
esta por encima del techo clasico.

    conda run -n surco python scripts/clasicos_contra_sam.py
"""

import pandas as pd

from surco import config, metricas, resultados


def cargar() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        resultados.unir("profundidad_surco-*.csv"),
        pd.read_csv(config.DIR_PROCESO / "clasicos.csv"),
    )


def _celda(tabla: pd.DataFrame, **filtros) -> pd.Series:
    return metricas.celda(tabla, **filtros)


def comparar(uno: pd.Series, otro: pd.Series, etiqueta: str) -> dict:
    """El contraste comun, con los nombres de columna que ya tiene el CSV."""
    contra = metricas.contraste(uno, otro)
    return {
        "comparacion": etiqueta,
        "diferencia": contra.diferencia,
        "ic_bajo": contra.ic_bajo,
        "ic_alto": contra.ic_alto,
        "unidades_ganadas": contra.ganadas,
        "de": contra.n,
    }


def a_protocolo_fijo(rejilla: pd.DataFrame, clasicos: pd.DataFrame) -> pd.DataFrame:
    """Cada clasico contra cada modelo en M1, con el mismo prompt los dos."""
    filas = []
    for metodo in config.METODOS_CLASICOS:
        for modelo in sorted(rejilla["modelo"].unique()):
            for protocolo in config.PROTOCOLOS:
                clasico = _celda(clasicos, modelo=metodo, protocolo=protocolo)
                sam = _celda(rejilla, modelo=modelo, modo="M1", protocolo=protocolo)
                if clasico.empty or sam.empty:
                    continue
                filas.append(
                    comparar(clasico, sam, f"{metodo} menos {modelo} en M1 x {protocolo}")
                )
    return pd.DataFrame(filas)


def dice_por_unidad_de(celda: tuple) -> pd.Series:
    """El Dice por unidad de una celda de modelo, de la fuente que le toque.

    Ninguna de las tres fuentes cubre la rejilla entera, asi que hay que saber
    de cual sale cada celda: M4 trae sus dos selectores en `m4_surco-*.csv`, S1
    de M0 y M1 esta en `selector.csv`, y el resto en la rejilla de F8.
    """
    modelo, modo, protocolo, selector = celda
    if modo == "M4":
        return metricas.celda(
            resultados.unir("m4_surco-*.csv"), modelo=modelo,
            protocolo=protocolo, selector=selector,
        )
    if selector == "S1":
        return metricas.celda(
            resultados.unir("selector.csv"), "dice_s1", modelo=modelo,
            modo=modo, protocolo=protocolo,
        )
    return metricas.celda(
        resultados.unir("profundidad_surco-*.csv"), modelo=modelo,
        modo=modo, protocolo=protocolo,
    )


def mejores(rejilla: pd.DataFrame, clasicos: pd.DataFrame) -> None:
    """El techo de cada lado, y la diferencia entre los dos.

    Los techos salen de `surco.resultados`, que mira **las tres fuentes**. Antes
    se sacaban de una sola cada uno, y eso enfrentaba al filtro clasico en su
    mejor celda contra SAM en una que ya no lo era: un sesgo contra SAM por
    accidente, no por diseno.
    """
    celdas = resultados.celdas_de_modelos()
    de_s0 = {c: d for c, d in celdas.items() if c[3] == "S0"}
    de_s1 = {c: d for c, d in celdas.items() if c[3] == "S1"}
    clave_s0, dice_s0 = resultados.mejor(de_s0)
    clave_s1, dice_s1 = resultados.mejor(de_s1)

    mejor_clasico_una = resultados.mejor(
        {c: d for c, d in resultados.celdas_de_clasicos().items() if c[2] == "una mascara"}
    )
    clave_clasico, dice_clasico = mejor_clasico_una

    print("\n=== EL TECHO DE CADA LADO ===")
    print(f"  SAM con S0: {' '.join(clave_s0):31s} {dice_s0:7.4f}")
    print(f"  SAM con S1: {' '.join(clave_s1):31s} {dice_s1:7.4f}")
    print(f"  clasico:    {' '.join(clave_clasico[:2]):31s} {dice_clasico:7.4f}")

    uno = _celda(clasicos, modelo=clave_clasico[0], protocolo=clave_clasico[1])
    _imprimir(comparar(uno, dice_por_unidad_de(clave_s0),
                       "mejor clasico menos mejor SAM con S0"))
    mejor_s1 = dice_por_unidad_de(clave_s1)
    _imprimir(comparar(uno, mejor_s1, "mejor clasico sin S1 menos mejor SAM con S1"))

    # El clasico tambien recibe tres candidatas y el mismo selector, asi que su
    # techo se mueve igual y hay que compararlos con el mismo trato.
    ruta_clasico = config.DIR_PROCESO / "selector_clasico.csv"
    if not ruta_clasico.exists():
        return
    con_selector = pd.read_csv(ruta_clasico)
    marcas = {
        protocolo: metricas.media_entre_unidades(
            metricas.por_unidad(grupo, "dice_s1")
        )
        for protocolo, grupo in con_selector.groupby("protocolo")
    }
    protocolo, dice = max(marcas.items(), key=lambda par: par[1])
    print(f"  clasico con S1: {'sato ' + protocolo:27s} {dice:7.4f}")
    mejor_clasico = metricas.por_unidad(
        con_selector[con_selector["protocolo"] == protocolo], "dice_s1"
    )
    _imprimir(
        comparar(mejor_clasico, mejor_s1, "mejor clasico con S1 menos mejor SAM con S1")
    )
    print("  (prompts distintos: es techo contra techo, no una comparacion pareada)")


def _imprimir(fila: dict) -> None:
    print(
        f"  {fila['comparacion']}: {fila['diferencia']:+.4f} "
        f"[{fila['ic_bajo']:+.4f}, {fila['ic_alto']:+.4f}], "
        f"gana en {fila['unidades_ganadas']} de {fila['de']}"
    )


def main() -> None:
    pd.set_option("display.width", 200)
    rejilla, clasicos = cargar()
    tabla = a_protocolo_fijo(rejilla, clasicos)

    print("=== A PROTOCOLO FIJO, CONTRA M1: positivo = gana el clasico ===")
    gana = tabla[tabla["ic_bajo"] > 0]
    print(f"  el clasico gana con el intervalo entero por encima del cero en "
          f"{len(gana)} de las {len(tabla)} comparaciones")
    if not gana.empty:
        print(gana.to_string(index=False, float_format=lambda v: f"{v:7.4f}"))

    print("\n  las diez mayores a favor del clasico:")
    print(
        tabla.sort_values("diferencia", ascending=False)
        .head(10)
        .to_string(index=False, float_format=lambda v: f"{v:7.4f}")
    )

    mejores(rejilla, clasicos)
    # No se guarda: es un agregado de `clasicos.csv`, la rejilla y el selector,
    # y `F9_clasicos.ipynb` lo recalcula llamando a `a_protocolo_fijo()`.


if __name__ == "__main__":
    main()
