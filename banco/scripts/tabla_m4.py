"""M4 contra M1: si el encadenado de mascara arregla el fallo del clic fijo.

**M1 es el modo que M4 tiene que superar**, y no M0. El fallo que M4 ataca es el
del clic fijo, que F2 midio que se sale de la lesion en 183 de los 295 pares y en
15 de las 17 unidades: la mascara de `t-1` esta donde estaba la lesion en `t-1`,
asi que arrastra la posicion que el clic ya no tiene. Compararlo contra M0 diria
solo que re-promptear ayuda, que ya se sabia.

Pareado unidad a unidad y a protocolo fijo, con el bootstrap del apartado 6.4.
**Y por selector**, porque M4 x S0 y M4 x S1 son dos cadenas distintas: lo que se
reencadena es lo que eligio el selector activo.

El M1 de referencia sale de donde ya estaba medido, sin volver a calcularlo: el
de S0 de la rejilla de F8 y el de S1 de la tabla del selector de F10.

    conda run -n surco python scripts/tabla_m4.py
"""

import pandas as pd

from surco import config, metricas, prompts, resultados


def cargar() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    m4 = resultados.unir("m4_surco-*.csv")
    rejilla = resultados.unir("profundidad_surco-*.csv")
    m1_s0 = rejilla[rejilla["modo"] == "M1"].rename(columns={"dice": "dice_m1"})
    seleccion = pd.read_csv(config.DIR_PROCESO / "selector.csv")
    m1_s1 = seleccion[seleccion["modo"] == "M1"].rename(columns={"dice_s1": "dice_m1"})
    return m4, {"S0": m1_s0, "S1": m1_s1}


def _pareada(uno: pd.Series, otro: pd.Series) -> dict:
    """El contraste comun, con los nombres de columna que ya tiene el CSV."""
    contra = metricas.contraste(uno, otro)
    return {
        "m4_menos_m1": contra.diferencia,
        "ic_bajo": contra.ic_bajo,
        "ic_alto": contra.ic_alto,
        "ganadas": contra.ganadas,
        "de": contra.n,
    }


def comparar(m4: pd.DataFrame, referencias: dict) -> pd.DataFrame:
    filas = []
    for (modelo, nombre_selector, protocolo), grupo in m4.groupby(
        ["modelo", "selector", "protocolo"]
    ):
        referencia = referencias[nombre_selector]
        propia = referencia[
            (referencia["modelo"] == modelo) & (referencia["protocolo"] == protocolo)
        ]
        if propia.empty:
            continue
        de_m4 = metricas.por_unidad(grupo)
        de_m1 = metricas.por_unidad(propia, "dice_m1")
        fila = {
            "modelo": modelo,
            "selector": nombre_selector,
            "protocolo": protocolo,
            "dice_m1": metricas.media_entre_unidades(de_m1),
            "dice_m4": metricas.media_entre_unidades(de_m4),
        }
        fila.update(_pareada(de_m4, de_m1))
        filas.append(fila)
    return pd.DataFrame(filas)


def censo_del_selector(m4: pd.DataFrame) -> pd.DataFrame:
    """S1 menos S0 dentro de M4: la parte del censo del 5.5 que falta en F10.

    El factor selector se mide en F10 reordenando las candidatas que F8 dejo
    guardadas, y eso solo alcanza a M0 y M1: son 24 celdas. En M4 el selector
    **cambia la cadena**, asi que sus dos variantes se corrieron dentro de los
    entornos de modelo y viven en `m4_surco-*.csv`. Sin este censo esas 12 celdas
    quedan medidas y sin contar, que es como el 5.5 llego a decir «8 de las 24»
    cuando el censo entero es de 36.

    SAM-Med2D sale del recuento por el mismo motivo que en el 5.5: propone una
    sola candidata, asi que S0 y S1 coinciden por construccion y su celda no
    puede moverse. Se marca en vez de filtrarse, para que el cero se vea.
    """
    filas = []
    for (modelo, protocolo), grupo in m4.groupby(["modelo", "protocolo"]):
        s0 = metricas.por_unidad(grupo[grupo["selector"] == "S0"])
        s1 = metricas.por_unidad(grupo[grupo["selector"] == "S1"])
        if s0.empty or s1.empty:
            continue
        contra = metricas.contraste(s1, s0)
        filas.append(
            {
                "modelo": modelo,
                "protocolo": protocolo,
                "dice_s0": metricas.media_entre_unidades(s0),
                "dice_s1": metricas.media_entre_unidades(s1),
                "s1_menos_s0": contra.diferencia,
                "ic_bajo": contra.ic_bajo,
                "ic_alto": contra.ic_alto,
                "ganadas": contra.ganadas,
                "de": contra.n,
                "degenerada": bool(grupo["n_candidatas"].max() == 1),
            }
        )
    return pd.DataFrame(filas)


def m4_contra_m3(m4: pd.DataFrame) -> pd.DataFrame:
    """Los dos modos temporales, dentro del unico modelo que tiene los dos.

    SAM 2.1 es el unico de la rejilla con banco de memoria y con prompt de
    mascara, asi que es el unico sitio donde M3 y M4 se pueden comparar sin
    cambiar de modelo. Los dos se comparan en **S0**, porque M3 devuelve una
    sola mascara y sin puntuacion: alli el selector es degenerado.
    """
    rejilla = pd.read_csv(config.DIR_PROCESO / "profundidad_surco-sam21.csv")
    de_m3 = rejilla[rejilla["modo"] == "M3"]
    de_m4 = m4[(m4["modelo"] == "sam21") & (m4["selector"] == "S0")]
    filas = []
    for protocolo in config.PROTOCOLOS:
        uno = metricas.por_unidad(de_m4[de_m4["protocolo"] == protocolo])
        otro = metricas.por_unidad(de_m3[de_m3["protocolo"] == protocolo])
        if uno.empty or otro.empty:
            continue
        fila = {
            "protocolo": protocolo,
            "dice_m3": metricas.media_entre_unidades(otro),
            "dice_m4": metricas.media_entre_unidades(uno),
        }
        fila.update(_pareada(uno, otro))
        filas.append(fila)
    return pd.DataFrame(filas).rename(columns={"m4_menos_m1": "m4_menos_m3"})


def por_posicion_en_la_cadena(m4: pd.DataFrame, referencias: dict) -> pd.DataFrame:
    """Dice segun cuantos frames lleva encadenados, contra el M1 del mismo sitio.

    Distingue las dos formas de fallar de un encadenado, que piden arreglos
    distintos. Un **desplazamiento constante** significa que la mascara previa
    ayuda o estorba lo mismo en el primer paso que en el decimoctavo. Una
    **caida monotona con la posicion** significa deriva acumulada: cada paso
    hereda el error del anterior y lo amplifica, que es el fallo clasico del
    encadenado y el motivo por el que M3 existe.
    """
    starts = {
        (u["serie"], u["etiqueta"]): u["start"] for u in prompts.cargar()["unidades"]
    }
    m4 = m4.assign(
        paso=[
            fila.frame - starts[(fila.serie, fila.etiqueta)]
            for fila in m4.itertuples()
        ]
    )
    m1 = referencias["S0"].assign(
        paso=[
            fila.frame - starts[(fila.serie, fila.etiqueta)]
            for fila in referencias["S0"].itertuples()
        ]
    )
    tramos = {"1": (1, 1), "2-5": (2, 5), "6-11": (6, 11), "12+": (12, 99)}
    filas = []
    for modelo, grupo in m4[m4["selector"] == "S0"].groupby("modelo"):
        propio_m1 = m1[m1["modelo"] == modelo]
        fila = {"modelo": modelo}
        for nombre, (bajo, alto) in tramos.items():
            dentro = grupo[grupo["paso"].between(bajo, alto)]
            base = propio_m1[propio_m1["paso"].between(bajo, alto)]
            fila[f"m4_{nombre}"] = float(dentro["dice"].mean())
            fila[f"m1_{nombre}"] = float(base["dice_m1"].mean())
        filas.append(fila)
    return pd.DataFrame(filas)


def main() -> None:
    pd.set_option("display.width", 200)
    m4, referencias = cargar()
    print(f"{len(m4)} filas de M4, {m4['modelo'].nunique()} modelos")

    tabla = comparar(m4, referencias)
    print("\n=== M4 MENOS M1, PAREADO POR UNIDAD, n = 17 ===")
    print(tabla.to_string(index=False, float_format=lambda v: f"{v:7.4f}"))

    separadas = tabla[(tabla["ic_bajo"] > 0) | (tabla["ic_alto"] < 0)]
    print(f"\n  celdas cuyo intervalo no cruza el cero: {len(separadas)} de {len(tabla)}")
    if not separadas.empty:
        print(separadas.to_string(index=False, float_format=lambda v: f"{v:7.4f}"))

    print("\n=== M4 POR SELECTOR Y PROTOCOLO, media de los modelos ===")
    print(
        tabla.groupby(["selector", "protocolo"])[["dice_m1", "dice_m4", "m4_menos_m1"]]
        .mean()
        .to_string(float_format=lambda v: f"{v:7.4f}")
    )

    print("\n=== EL SELECTOR DENTRO DE M4: S1 MENOS S0 (apartado 5.5) ===")
    censo = censo_del_selector(m4)
    vivas = censo[~censo["degenerada"]]
    print(vivas.to_string(index=False, float_format=lambda v: f"{v:7.4f}"))
    fuera = vivas[(vivas["ic_bajo"] > 0) | (vivas["ic_alto"] < 0)]
    degeneradas = ", ".join(sorted(censo[censo["degenerada"]]["modelo"].unique()))
    print(
        f"\n  celdas vivas en M4: {len(vivas)}, con el intervalo fuera del cero "
        f"{len(fuera)}, {int((fuera['s1_menos_s0'] > 0).sum())} a favor de S1"
    )
    print(f"  una sola candidata, fuera del recuento: {degeneradas}")

    print("\n=== LOS DOS MODOS TEMPORALES, DENTRO DE SAM 2.1 (S0) ===")
    print(
        m4_contra_m3(m4).to_string(index=False, float_format=lambda v: f"{v:7.4f}")
    )

    print("\n=== SEGUN LA POSICION EN LA CADENA (S0, media de los protocolos) ===")
    print(
        por_posicion_en_la_cadena(m4, referencias).to_string(
            index=False, float_format=lambda v: f"{v:7.4f}"
        )
    )
    print(
        "  paso = frames encadenados desde la semilla. Un desplazamiento constante\n"
        "  es un sesgo del prompt de mascara; una caida con el paso es deriva."
    )

    # No se guarda: es un agregado de los `m4_surco-*.csv` y de la rejilla, y
    # `F8_profundidad.ipynb` lo recalcula llamando a `comparar()`.


if __name__ == "__main__":
    main()
