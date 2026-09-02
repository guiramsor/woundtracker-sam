"""Analisis parcial de una anotadora contra el GT del autor (apartado 7).

**Tres casos por separado, nunca en un solo numero**, porque miden cosas
distintas:

1. **Concordancia de trazado.** Dice entre la ROI de la anotadora y la del GT,
   solo sobre las lesiones que las dos encontraron. Es el techo humano.
2. **Falsos negativos humanos.** Lesiones del GT que la anotadora no encontro.
   Un Dice bajo y una lesion no vista no son el mismo problema: el primero es
   precision de trazado, el segundo es deteccion.
3. **Falsos positivos humanos.** ROIs de la anotadora que no emparejan con
   ninguna del GT.

Y **por estrato**, temprano y tardio separados, sin agregar (apartado 6.5): en
los frames tardios el surco esta fragmentado y un desacuerdo alto ahi puede
venir de ambiguedad del protocolo y no de falta de precision.

Las imagenes marcadas como no identificable se cuentan aparte: son una respuesta
valida del protocolo, no un hueco.

    conda run -n surco python scripts/concordancia_anotadora.py --anotadora anotadora_1
"""

import argparse

import numpy as np
import pandas as pd
import roifile

from surco import config, datos, metricas

MARCA_NO_IDENTIFICABLE = "NO_IDENTIFICABLE"


def _mascara_de_roi(roi, forma: tuple[int, int]) -> np.ndarray:
    """Rasteriza una ROI de Fiji sobre una mascara del tamano de la imagen."""
    from matplotlib.path import Path

    coordenadas = roi.coordinates()
    if coordenadas is None or len(coordenadas) < 3:
        return np.zeros(forma, dtype=bool)
    alto, ancho = forma
    y, x = np.mgrid[:alto, :ancho]
    puntos = np.column_stack([x.ravel(), y.ravel()])
    dentro = Path(coordenadas).contains_points(puntos)
    return dentro.reshape(forma)


def rois_de(ruta, forma: tuple[int, int]) -> list[np.ndarray]:
    return [_mascara_de_roi(roi, forma) for roi in roifile.roiread(str(ruta))]


def emparejar(humanas: list[np.ndarray], gt: dict[int, np.ndarray]) -> dict:
    """Empareja cada lesion del GT con la ROI humana que mas se le solapa.

    Una ROI humana solo puede emparejar con una lesion, y se exige solape no
    nulo: sin eso, "emparejar" seria elegir la menos mala.
    """
    libres = set(range(len(humanas)))
    parejas, no_encontradas = {}, []
    for etiqueta, mascara in gt.items():
        mejor, mejor_dice = None, 0.0
        for indice in libres:
            valor = metricas.dice(humanas[indice], mascara)
            if valor > mejor_dice:
                mejor, mejor_dice = indice, valor
        if mejor is None:
            no_encontradas.append(etiqueta)
        else:
            parejas[etiqueta] = (mejor, mejor_dice)
            libres.discard(mejor)
    return {"parejas": parejas, "no_encontradas": no_encontradas, "sobrantes": libres}


def analizar(carpeta, correspondencia: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    filas, no_identificables = [], []
    for fichero, grupo in correspondencia.groupby("fichero"):
        serie = int(grupo["serie"].iloc[0])
        frame = int(grupo["frame"].iloc[0])
        base = fichero.replace(".tif", "")

        marca = list(carpeta.glob(f"{base}*{MARCA_NO_IDENTIFICABLE}*"))
        zip_roi = carpeta / f"{base}.zip"
        if marca and not zip_roi.exists():
            no_identificables.append((fichero, sorted(grupo["unidad"])))
            continue
        if not zip_roi.exists():
            no_identificables.append((fichero, ["SIN ENTREGAR"]))
            continue

        gt_frame = datos.cargar_gt(serie, frame)
        # **Todas** las lesiones del frame, no solo las que la tabla de
        # correspondencia asigna a este fichero. La anotadora traza lo que ve, y
        # un frame puede contener lesiones que aqui solo entran por su otro
        # estrato. Comparar contra el subconjunto convertiria esas ROIs
        # legitimas en falsos positivos inventados.
        etiquetas_del_frame = [
            int(e) for e in np.unique(gt_frame) if e != config.ETIQUETA_FONDO
        ]
        gt = {e: gt_frame == e for e in etiquetas_del_frame}
        humanas = rois_de(zip_roi, gt_frame.shape)
        resultado = emparejar(humanas, gt)

        for etiqueta, fila in zip(grupo["etiqueta"], grupo.itertuples()):
            etiqueta = int(etiqueta)
            emparejada = resultado["parejas"].get(etiqueta)
            filas.append(
                {
                    "fichero": fichero,
                    "unidad": f"{serie}-L{etiqueta}",
                    "estrato": fila.estrato,
                    "frame": frame,
                    "encontrada": emparejada is not None,
                    "dice": emparejada[1] if emparejada else np.nan,
                    "rois_humanas": len(humanas),
                    "sobrantes_en_el_fichero": len(resultado["sobrantes"]),
                }
            )
    return pd.DataFrame(filas), no_identificables


def main(argv: list[str] | None = None) -> None:
    analizador = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    analizador.add_argument("--anotadora", default="anotadora_1")
    argumentos = analizador.parse_args(argv)

    carpeta = config.DIR_ANOTACIONES / argumentos.anotadora
    correspondencia = pd.read_csv(
        config.DIR_PROCESO / "concordancia" / "correspondencia.csv"
    )
    tabla, no_identificables = analizar(carpeta, correspondencia)

    pd.set_option("display.width", 200)
    print(f"=== {argumentos.anotadora} ===")
    print(f"pares esperados: {len(correspondencia)}   analizados: {len(tabla)}")

    print("\n=== 1. CONCORDANCIA DE TRAZADO, por estrato ===")
    encontradas = tabla[tabla["encontrada"]]
    for estrato, grupo in encontradas.groupby("estrato"):
        print(
            f"  {estrato:<9} n={len(grupo):2d}  Dice medio {grupo['dice'].mean():.4f}"
            f"  mediana {grupo['dice'].median():.4f}"
            f"  min {grupo['dice'].min():.4f}  max {grupo['dice'].max():.4f}"
        )
    print("\n  por unidad:")
    print(
        encontradas.pivot_table(index="unidad", columns="estrato", values="dice")
        .to_string(float_format=lambda v: f"{v:7.4f}")
    )

    print("\n=== 2. FALSOS NEGATIVOS HUMANOS: lesiones del GT no encontradas ===")
    perdidas = tabla[~tabla["encontrada"]]
    if perdidas.empty:
        print("  ninguna")
    else:
        print(perdidas[["unidad", "estrato", "fichero"]].to_string(index=False))

    print("\n=== 3. FALSOS POSITIVOS HUMANOS: ROIs sin pareja en el GT ===")
    sobrantes = (
        tabla.groupby("fichero")[["rois_humanas", "sobrantes_en_el_fichero"]]
        .first()
        .query("sobrantes_en_el_fichero > 0")
    )
    print("  ninguna" if sobrantes.empty else sobrantes.to_string())

    print("\n=== NO IDENTIFICABLE ===")
    if not no_identificables:
        print("  ninguna")
    for fichero, unidades in no_identificables:
        print(f"  {fichero}: {', '.join(map(str, unidades))}")

    cruzar_con_fragmentacion(tabla)

    destino = config.DIR_PROCESO / f"concordancia_{argumentos.anotadora}.csv"
    tabla.to_csv(destino, index=False)
    print(f"\ntabla en {destino}")


def cruzar_con_fragmentacion(tabla: pd.DataFrame) -> None:
    """Si el desacuerdo humano se explica por el GT partido, eso es un resultado.

    El surco fragmentado obliga a decidir donde acaba la lesion, y ahi dos
    personas discrepan sobre la definicion y no sobre la imagen. Si el Dice cae
    con el numero de componentes, el desacuerdo no mide falta de precision sino
    ambiguedad del protocolo, que es una conclusion distinta y accionable:
    se arregla escribiendo mejor el protocolo, no anotando con mas cuidado.
    """
    inventario_df = pd.read_csv(config.DIR_PROCESO / "inventario_gt.csv")
    partes = tabla["unidad"].str.split("-L", expand=True)
    tabla = tabla.assign(serie=partes[0].astype(int), etiqueta=partes[1].astype(int))
    columna = config.COLUMNA_FRAGMENTACION
    junto = tabla.merge(
        inventario_df[["serie", "etiqueta", "frame", columna, "area_px2"]],
        on=["serie", "etiqueta", "frame"],
        how="left",
    ).dropna(subset=["dice"])

    print("\n=== DICE DE LA ANOTADORA CONTRA LA FRAGMENTACION DEL GT ===")
    print(
        junto.groupby(columna)["dice"]
        .agg(["count", "mean", "median"])
        .to_string(float_format=lambda v: f"{v:7.4f}")
    )
    for metodo in ("spearman", "pearson"):
        rho = junto["dice"].corr(junto[columna], method=metodo)
        print(f"  correlacion {metodo:<9}: {rho:+.3f}")

    print("\n  por estrato:")
    for estrato, grupo in junto.groupby("estrato"):
        rho = grupo["dice"].corr(grupo[columna], method="spearman")
        print(
            f"    {estrato:<9} n={len(grupo):2d}  componentes medias "
            f"{grupo[columna].mean():.2f}  rho={rho:+.3f}"
        )

    print("\n  las cinco mayores caidas entre estratos:")
    ancho = junto.pivot_table(index="unidad", columns="estrato", values="dice")
    componentes = junto.pivot_table(
        index="unidad", columns="estrato", values=columna
    )
    if {"temprano", "tardio"} <= set(ancho.columns):
        caida = (ancho["temprano"] - ancho["tardio"]).rename("caida_dice")
        resumen = pd.concat(
            [
                caida,
                componentes.rename(columns=lambda c: f"comp_{c}"),
                ancho.rename(columns=lambda c: f"dice_{c}"),
            ],
            axis=1,
        ).sort_values("caida_dice", ascending=False)
        print(resumen.head(5).to_string(float_format=lambda v: f"{v:7.3f}"))


if __name__ == "__main__":
    main()
