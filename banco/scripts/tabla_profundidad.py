"""La rejilla de profundidad: une los CSV por entorno y la presenta (F8).

Agregacion en dos pasos del apartado 6.3, como en el cribado: primero dentro de
cada lesion y despues entre las 17. n = 17, no 295.

**La celda de M2 sale vacia y con su motivo**, no ausente. El apartado 5.1 la
declara no implementable por dependencia no disponible: si la tabla se limitara
a no tenerla, se leeria como que no dio tiempo.

Las comparaciones que la rejilla contesta van pareadas unidad a unidad y con el
bootstrap del apartado 6.4, porque todos los modelos y todos los modos ven
exactamente las mismas lesiones.

    conda run -n surco python scripts/tabla_profundidad.py
"""

import pandas as pd

from surco import config, metricas, resultados

MODOS_MEDIDOS = ("M0", "M1", "M3")

PATRON = "profundidad_surco-*.csv"


def cargar() -> pd.DataFrame:
    ficheros = resultados.ficheros_de(PATRON)
    tabla = resultados.unir(PATRON)
    print(f"{len(ficheros)} entornos, {len(tabla)} filas")
    return tabla


def rejilla(tabla: pd.DataFrame) -> pd.DataFrame:
    """Dice medio entre unidades por modelo, modo y protocolo."""
    filas = []
    for (modelo, modo, protocolo), grupo in tabla.groupby(
        ["modelo", "modo", "protocolo"]
    ):
        filas.append(
            {
                "modelo": modelo,
                "modo": modo,
                "protocolo": protocolo,
                "dice": metricas.media_entre_unidades(metricas.por_unidad(grupo)),
            }
        )
    return (
        pd.DataFrame(filas)
        .pivot(index=["modelo", "modo"], columns="protocolo", values="dice")
        .sort_index()
    )


def celda(tabla: pd.DataFrame, modelo: str, modo: str, protocolo: str) -> pd.Series:
    return metricas.celda(tabla, modelo=modelo, modo=modo, protocolo=protocolo)


def comparar(tabla: pd.DataFrame, una: tuple, otra: tuple) -> str:
    """Diferencia pareada entre dos celdas de la rejilla, con su intervalo."""
    uno, otro = celda(tabla, *una), celda(tabla, *otra)
    if uno.empty or otro.empty:
        return ""
    contra = metricas.contraste(uno, otro)
    return f"  {'/'.join(una)} menos {'/'.join(otra)}: {contra}"


def diferencias_de_modo(tabla: pd.DataFrame) -> None:
    """Que aporta el tiempo: M1 contra M0, y M3 contra M1 donde exista."""
    print("\n=== EL FACTOR MODO, A PROTOCOLO FIJO ===")
    for modelo in sorted(tabla["modelo"].unique()):
        propias = tabla[tabla["modelo"] == modelo]
        print(f" {modelo}")
        for protocolo in config.PROTOCOLOS:
            linea = comparar(tabla, (modelo, "M1", protocolo), (modelo, "M0", protocolo))
            if linea:
                print(linea)
        if "M3" in set(propias["modo"]):
            for protocolo in config.PROTOCOLOS:
                linea = comparar(
                    tabla, (modelo, "M3", protocolo), (modelo, "M1", protocolo)
                )
                if linea:
                    print(linea)


def diferencias_de_protocolo(tabla: pd.DataFrame) -> None:
    """Cuantos clics hacen falta: cada protocolo contra P1, en M1."""
    print("\n=== EL FACTOR PROMPT, EN M1 ===")
    for modelo in sorted(tabla["modelo"].unique()):
        print(f" {modelo}")
        for protocolo in list(config.PROTOCOLOS)[1:]:
            linea = comparar(tabla, (modelo, "M1", protocolo), (modelo, "M1", "P1"))
            if linea:
                print(linea)


def main() -> None:
    pd.set_option("display.width", 200)
    tabla = cargar()

    print("\n=== REJILLA DE PROFUNDIDAD: Dice medio entre unidades, S0, n = 17 ===")
    print(rejilla(tabla).to_string(float_format=lambda v: f"{v:7.4f}"))
    print(
        "\n  M2: celda vacia por dependencia no disponible (apartado 5.1). No se ha\n"
        "  medido porque con estos datos no se puede medir: el nucleo no es\n"
        "  segmentable en w1CSU488 y w2TRANS solo existe en t1."
    )
    ausentes = [
        modelo
        for modelo in sorted(tabla["modelo"].unique())
        if "M3" not in set(tabla[tabla["modelo"] == modelo]["modo"])
    ]
    print(f"  M3: no lo soportan {', '.join(ausentes)}; la comparacion vive dentro")

    diferencias_de_modo(tabla)
    diferencias_de_protocolo(tabla)

    # La rejilla no se guarda en disco. Es un agregado de los
    # `profundidad_surco-*.csv`, se recalcula en un segundo, y el unico sitio
    # que la enseña es `F8_profundidad.ipynb`, que llama a `rejilla()` en vez de
    # leer el fichero. Guardarla solo servia para tener dos copias del mismo
    # numero y una de ellas envejeciendo sin avisar.


if __name__ == "__main__":
    main()
