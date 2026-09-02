"""M1 contra M3 en SAM 2.1 y en MedSAM2: la memoria contra el dominio.

Los dos son **el mismo modelo y su fork**: misma mecanica de banco de memoria,
distinto dominio de ajuste. Medir M1 contra M3 en los dos separa si lo que
aporta la memoria temporal depende del dominio o de la memoria en si, y es la
unica comparacion del conjunto que puede contestar eso (apartado 5.1).

Todo pareado unidad a unidad y con el bootstrap del apartado 6.4, porque los dos
modelos ven exactamente las mismas 17 lesiones.

**Solo se puede hacer donde exista M1 de los dos.** El de SAM 2.1 sale de la
rejilla de F8, en los cuatro protocolos; el de MedSAM2 sale del cribado, que fijo
P1 para todos. Los protocolos sin las dos mitades se listan y no se rellenan.

    conda run -n surco python scripts/memoria_por_dominio.py
"""

import pandas as pd

from surco import config, metricas


def _por_unidad(tabla: pd.DataFrame, **filtros) -> pd.Series:
    return metricas.celda(tabla, **filtros)


def cargar() -> dict[str, dict[str, pd.Series]]:
    """Dice por unidad de cada celda que hace falta, por modelo y modo."""
    rejilla = pd.read_csv(config.DIR_PROCESO / "profundidad_surco-sam21.csv")
    suelto = pd.read_csv(config.DIR_PROCESO / "m3_suelto_medsam2.csv")
    suelto_m1 = pd.read_csv(config.DIR_PROCESO / "m1_suelto_medsam2.csv")
    cribado = pd.read_csv(config.DIR_PROCESO / "cribado_surco-medsam2.csv")
    # Como texto los dos lados: el CSV suelto solo tiene "2411" y pandas lo lee
    # como entero, mientras que el del cribado convive con "latest" y se queda
    # en texto. Comparados tal cual, el filtro no casa con nada.
    mismo = cribado["checkpoint"].astype(str) == str(suelto["checkpoint"].iloc[0])
    cribado = cribado[mismo]
    if cribado.empty:
        raise ValueError("el cribado no tiene el checkpoint de la medida suelta")

    celdas: dict[str, dict[str, pd.Series]] = {"sam21": {}, "medsam2": {}}
    for protocolo in config.PROTOCOLOS:
        for modo in ("M1", "M3"):
            serie = _por_unidad(rejilla, modo=modo, protocolo=protocolo)
            if not serie.empty:
                celdas["sam21"][f"{modo}/{protocolo}"] = serie
        serie = _por_unidad(suelto, modo="M3", protocolo=protocolo)
        if not serie.empty:
            celdas["medsam2"][f"M3/{protocolo}"] = serie
        serie = _por_unidad(suelto_m1, modo="M1", protocolo=protocolo)
        if not serie.empty:
            celdas["medsam2"][f"M1/{protocolo}"] = serie
    # El M1 x P1 de MedSAM2 sale del cribado, que fijo esa configuracion para
    # todos, y por eso no esta en la medida suelta: se lee de donde ya estaba.
    celdas["medsam2"]["M1/P1"] = metricas.por_unidad(cribado)
    return celdas


def _linea(etiqueta: str, uno: pd.Series, otro: pd.Series) -> str:
    return f"  {etiqueta:34s} {metricas.contraste(uno, otro)}"


def main() -> None:
    celdas = cargar()
    aportes: dict[str, dict[str, pd.Series]] = {}

    print("=== LO QUE APORTA LA MEMORIA: M3 menos M1, pareado, n = 17 ===")
    for modelo, propias in celdas.items():
        aportes[modelo] = {}
        print(f" {modelo}")
        for protocolo in config.PROTOCOLOS:
            uno, otro = f"M3/{protocolo}", f"M1/{protocolo}"
            if uno not in propias or otro not in propias:
                print(f"  {protocolo}: sin M1 medido, no hay comparacion")
                continue
            aportes[modelo][protocolo] = metricas.diferencia_pareada(
                propias[uno], propias[otro]
            )
            print(_linea(f"M3 menos M1 en {protocolo}", propias[uno], propias[otro]))

    comunes = sorted(set(aportes["sam21"]) & set(aportes["medsam2"]))
    print("\n=== EL APORTE DE UNO MENOS EL DEL OTRO ===")
    print(
        "  Si la memoria aportara lo mismo en los dos, esta diferencia seria cero:\n"
        "  lo que quede es lo que pone el dominio de ajuste encima de la mecanica."
    )
    for protocolo in comunes:
        print(
            _linea(
                f"aporte medsam2 menos sam21 en {protocolo}",
                aportes["medsam2"][protocolo],
                aportes["sam21"][protocolo],
            )
        )
    if not comunes:
        print("  ninguna: falta M1 de MedSAM2 en todos los protocolos")


if __name__ == "__main__":
    main()
