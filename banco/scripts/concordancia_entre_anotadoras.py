"""El techo humano del apartado 7: las dos anotadoras entre si, y contra el GT.

**El estadistico principal es el Dice entre las dos anotadoras, sin pasar por el
GT del autor.** Es el unico numero del trabajo que no depende de quien lo
escribe: mide cuanto coinciden dos personas que han visto la misma imagen y han
seguido el mismo protocolo, y por debajo de el no tiene sentido pedirle nada a un
modelo.

Y las tres comparaciones juntas, A1-GT, A2-GT y A1-A2, porque contestan una
pregunta que ninguna contesta sola: **si las dos anotadoras concuerdan mas entre
si que con el GT, el atipico es el GT**, y eso hay que decirlo.

**Nada se agrega entre estratos** (apartado 6.5). En los frames tardios el surco
esta fragmentado, y ahi el desacuerdo mide ambiguedad del protocolo y no falta de
precision.

    conda run -n surco python scripts/concordancia_entre_anotadoras.py
"""

import numpy as np
import pandas as pd

from surco import config, datos, metricas, resultados, selector

from concordancia_anotadora import MARCA_NO_IDENTIFICABLE, emparejar, rois_de

ANOTADORAS = ("anotadora_1", "anotadora_2")
# La celda de la rejilla con la que se compara el techo humano: el mejor modelo
# del trabajo (apartado 5.5), con su selector puesto. Vive en el config.
MEJOR = config.MEJOR_CELDA
# Las tres columnas con las que se localiza esa celda dentro de `selector.csv`,
# que no tiene columna de checkpoint ni de selector: alli el selector son dos
# columnas de Dice, `dice_s0` y `dice_s1`.
CLAVES_DE_CELDA = ("modelo", "modo", "protocolo")


def carpeta_de(anotadora: str):
    return config.DIR_ANOTACIONES / anotadora


def entrega_de(anotadora: str, fichero: str, forma) -> list | None:
    """Las ROIs de esa anotadora en ese fichero, o None si no identificable."""
    carpeta = carpeta_de(anotadora)
    base = fichero.replace(".tif", "")
    ruta = carpeta / f"{base}.zip"
    if ruta.exists():
        return rois_de(ruta, forma)
    if list(carpeta.glob(f"{base}*{MARCA_NO_IDENTIFICABLE}*")):
        return None
    raise FileNotFoundError(f"{anotadora} no entrego nada para {fichero}")


def entre_anotadoras(correspondencia: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Dice A1 contra A2 emparejando por solape, **sin tocar el GT**.

    El emparejamiento es el mismo de siempre, por maximo solape y exigiendo que
    no sea nulo. Aqui no hay una referencia y otra prediccion: se empareja A2
    contra A1 porque hay que fijar un orden, y el Dice es simetrico.
    """
    filas, incidencias = [], []
    for fichero, grupo in correspondencia.groupby("fichero"):
        serie, frame = int(grupo["serie"].iloc[0]), int(grupo["frame"].iloc[0])
        forma = datos.cargar_gt(serie, frame).shape
        entregas = {a: entrega_de(a, fichero, forma) for a in ANOTADORAS}
        if any(rois is None for rois in entregas.values()):
            faltan = [a for a, rois in entregas.items() if rois is None]
            incidencias.append((fichero, faltan))
            continue
        una, otra = entregas["anotadora_1"], entregas["anotadora_2"]
        resultado = emparejar(otra, dict(enumerate(una)))
        for indice, (pareja, dice) in resultado["parejas"].items():
            filas.append(
                {
                    "fichero": fichero,
                    "serie": serie,
                    "frame": frame,
                    "estrato": grupo["estrato"].iloc[0],
                    "roi_a1": indice,
                    "roi_a2": pareja,
                    "dice_a1_a2": dice,
                }
            )
        for indice in resultado["no_encontradas"]:
            incidencias.append((fichero, [f"ROI {indice} de A1 sin pareja en A2"]))
        for indice in resultado["sobrantes"]:
            incidencias.append((fichero, [f"ROI {indice} de A2 sin pareja en A1"]))
    return pd.DataFrame(filas), incidencias


def por_lesion(correspondencia: pd.DataFrame) -> pd.DataFrame:
    """Las tres comparaciones ancladas en cada lesion del GT.

    Aqui si se usa el GT, y a proposito: es lo que permite poner las tres
    medidas sobre la misma lesion y ver cual de las tres se separa.
    """
    filas = []
    for fichero, grupo in correspondencia.groupby("fichero"):
        serie, frame = int(grupo["serie"].iloc[0]), int(grupo["frame"].iloc[0])
        gt_frame = datos.cargar_gt(serie, frame)
        etiquetas = [
            int(e) for e in np.unique(gt_frame) if e != config.ETIQUETA_FONDO
        ]
        gt = {e: gt_frame == e for e in etiquetas}
        entregas = {a: entrega_de(a, fichero, gt_frame.shape) for a in ANOTADORAS}
        if any(rois is None for rois in entregas.values()):
            continue
        parejas = {
            a: emparejar(rois, gt)["parejas"] for a, rois in entregas.items()
        }
        for fila in grupo.itertuples():
            etiqueta = int(fila.etiqueta)
            una = parejas["anotadora_1"].get(etiqueta)
            otra = parejas["anotadora_2"].get(etiqueta)
            if una is None or otra is None:
                continue
            filas.append(
                {
                    "unidad": f"{serie}-L{etiqueta}",
                    "serie": serie,
                    "etiqueta": etiqueta,
                    "frame": frame,
                    "estrato": fila.estrato,
                    "a1_gt": una[1],
                    "a2_gt": otra[1],
                    "a1_a2": metricas.dice(
                        entregas["anotadora_1"][una[0]], entregas["anotadora_2"][otra[0]]
                    ),
                }
            )
    return pd.DataFrame(filas)


def _resumen(grupo: pd.DataFrame, columna: str) -> str:
    valores = grupo[columna]
    return (
        f"n={len(valores):2d}  media {valores.mean():.4f}  "
        f"mediana {valores.median():.4f}  min {valores.min():.4f}  "
        f"max {valores.max():.4f}"
    )


def _pareada(grupo: pd.DataFrame, una: str, otra: str) -> str:
    """Aqui las dos columnas ya vienen pareadas por fila, no por unidad.

    Cada fila es una lesion en un frame, y las dos medidas se hicieron sobre
    ella, asi que se restan directamente en vez de agregarlas antes.
    """
    return str(metricas.contraste(grupo[una], grupo[otra]))


def contra_la_fragmentacion(tabla: pd.DataFrame) -> None:
    """Si el acuerdo entre las dos cae con el GT partido, el hallazgo se sostiene.

    Con la anotadora 1 contra el GT salio rho = -0.562. Repetirlo **entre dos
    personas independientes** quita de en medio al autor: si el desacuerdo sigue
    a la fragmentacion tambien ahi, no es que una anotadora se aparte del GT, es
    que el objeto es ambiguo cuando esta roto.
    """
    inventario_df = pd.read_csv(config.DIR_PROCESO / "inventario_gt.csv")
    columna = config.COLUMNA_FRAGMENTACION
    junto = tabla.merge(
        inventario_df[["serie", "etiqueta", "frame", columna]],
        on=["serie", "etiqueta", "frame"],
        how="left",
    )
    print("\n=== 4. EL ACUERDO ENTRE LAS DOS, CONTRA LA FRAGMENTACION DEL GT ===")
    print(
        junto.groupby(columna)["a1_a2"]
        .agg(["count", "mean", "median"])
        .to_string(float_format=lambda v: f"{v:7.4f}")
    )
    for metodo in ("spearman", "pearson"):
        print(f"  correlacion {metodo:<9}: {junto['a1_a2'].corr(junto[columna], method=metodo):+.3f}")
    for estrato, grupo in junto.groupby("estrato"):
        rho = grupo["a1_a2"].corr(grupo[columna], method="spearman")
        print(
            f"    {estrato:<9} n={len(grupo):2d}  componentes medias "
            f"{grupo[columna].mean():.2f}  rho={rho:+.3f}"
        )


def mascara_del_modelo(serie: int, etiqueta: int, frame: int):
    """La mascara que el mejor modelo produce para esa lesion en ese frame.

    Se recalcula desde las candidatas que guardo F8, con el mismo selector, en
    vez de leerse de `selector.csv`: alli solo esta el Dice contra el GT, y aqui
    hace falta la mascara para compararla con las ROIs humanas. Lo hace
    `selector.mascara_guardada`, que es tambien lo que dibuja los overlays.
    """
    return selector.mascara_guardada(MEJOR, serie, etiqueta, frame)


def _mejor_solape(mascara, rois: list) -> float:
    """Dice contra la ROI humana que mas se le solapa, o 0 si no toca ninguna.

    Es el mismo emparejamiento por solape con el que se compararon las dos
    anotadoras entre si: el modelo entra como un tercer anotador y se le mide
    igual, sin que su mascara pase por el GT en ningun momento.
    """
    valores = [metricas.dice(mascara, roi) for roi in rois]
    return max(valores) if valores else 0.0


def como_tercer_anotador(correspondencia: pd.DataFrame) -> pd.DataFrame:
    """El modelo tratado como una anotadora mas, por solape y sin GT."""
    filas = []
    for fichero, grupo in correspondencia.groupby("fichero"):
        serie, frame = int(grupo["serie"].iloc[0]), int(grupo["frame"].iloc[0])
        forma = datos.cargar_gt(serie, frame).shape
        entregas = {a: entrega_de(a, fichero, forma) for a in ANOTADORAS}
        if any(rois is None for rois in entregas.values()):
            continue
        for fila in grupo.itertuples():
            etiqueta = int(fila.etiqueta)
            mascara = mascara_del_modelo(serie, etiqueta, frame)
            if mascara is None:
                continue
            filas.append(
                {
                    "unidad": f"{serie}-L{etiqueta}",
                    "serie": serie,
                    "etiqueta": etiqueta,
                    "frame": frame,
                    "estrato": fila.estrato,
                    "modelo_a1": _mejor_solape(mascara, entregas["anotadora_1"]),
                    "modelo_a2": _mejor_solape(mascara, entregas["anotadora_2"]),
                }
            )
    return pd.DataFrame(filas)


def dice_de_la_mejor_celda() -> pd.DataFrame:
    """El Dice contra el GT de la mejor celda, por par (unidad, frame).

    Sale de un sitio u otro segun el modo, y por eso no basta con filtrar
    `selector.csv`: alli solo estan M0 y M1, porque en esos modos el selector se
    aplica **despues** sobre las candidatas guardadas. M4 se midio dentro del
    entorno del modelo, con una cadena por selector, y vive en sus propios CSV.
    """
    if MEJOR["modo"] == "M4":
        m4 = resultados.unir("m4_surco-*.csv")
        propias = m4[
            (m4["modelo"] == MEJOR["modelo"]) & (m4["protocolo"] == MEJOR["protocolo"])
        ]
        anchas = propias.pivot_table(
            index=["serie", "etiqueta", "frame"], columns="selector", values="dice"
        ).reset_index()
        return anchas.rename(columns={"S0": "dice_s0", "S1": "dice_s1"})

    seleccion = pd.read_csv(config.DIR_PROCESO / "selector.csv")
    for columna in CLAVES_DE_CELDA:
        seleccion = seleccion[seleccion[columna] == MEJOR[columna]]
    return seleccion[["serie", "etiqueta", "frame", "dice_s0", "dice_s1"]]


def contra_el_mejor_modelo(tabla: pd.DataFrame) -> None:
    """El techo humano frente al mejor modelo, sobre las mismas unidades.

    **En diferencias de Dice**, nunca en porcentaje: Dice no es una escala donde
    dividir signifique algo.

    Hay una asimetria que conviene tener presente al leerlo. `a1_a2` mide cuanto
    coinciden dos personas entre si; `dice_s1` mide cuanto coincide el modelo con
    el GT del autor. Las comparables entre si sin matices son `a1_gt`, `a2_gt` y
    `dice_s1`, que las tres miden acuerdo **con el mismo GT**.
    """
    junto = tabla.merge(
        dice_de_la_mejor_celda(), on=["serie", "etiqueta", "frame"], how="left"
    )
    nombre = f"{MEJOR['modelo']} {MEJOR['modo']} x {MEJOR['protocolo']}"
    print(f"\n=== 6. EL TECHO HUMANO FRENTE A {nombre.upper()} ===")
    for estrato, grupo in junto.groupby("estrato"):
        disponibles = grupo.dropna(subset=["dice_s1"])
        print(f"\n  {estrato} (n={len(disponibles)})")
        for columna, etiqueta in (
            ("a1_a2", "techo humano, A1 contra A2"),
            ("a1_gt", "A1 contra GT"),
            ("a2_gt", "A2 contra GT"),
            ("dice_s0", "modelo con S0"),
            ("dice_s1", "modelo con S1"),
        ):
            print(f"    {etiqueta:<28} {disponibles[columna].mean():.4f}")
        for columna, etiqueta in (
            ("a1_a2", "techo humano"),
            ("a1_gt", "A1 contra GT"),
            ("a2_gt", "A2 contra GT"),
        ):
            print(
                f"    modelo con S1 menos {etiqueta:<16} "
                f"{_pareada(disponibles, 'dice_s1', columna)}"
            )


def main() -> None:
    pd.set_option("display.width", 200)
    correspondencia = pd.read_csv(
        config.DIR_PROCESO / "concordancia" / "correspondencia.csv"
    )

    directo, incidencias = entre_anotadoras(correspondencia)
    print("=== 2. EL TECHO HUMANO: A1 CONTRA A2, SIN PASAR POR EL GT ===")
    for estrato, grupo in directo.groupby("estrato"):
        print(f"  {estrato:<9} {_resumen(grupo, 'dice_a1_a2')}")
    print("\n  por fichero:")
    print(
        directo.groupby(["estrato", "fichero"])["dice_a1_a2"]
        .agg(["count", "mean"])
        .to_string(float_format=lambda v: f"{v:7.4f}")
    )
    print("\n  incidencias del emparejamiento:")
    if not incidencias:
        print("    ninguna")
    for fichero, detalle in incidencias:
        print(f"    {fichero}: {', '.join(map(str, detalle))}")

    tabla = por_lesion(correspondencia)
    print("\n=== 3. LAS TRES COMPARACIONES SOBRE LA MISMA LESION ===")
    for estrato, grupo in tabla.groupby("estrato"):
        print(f"\n  {estrato} (n={len(grupo)})")
        for columna, etiqueta in (
            ("a1_gt", "A1 contra GT"),
            ("a2_gt", "A2 contra GT"),
            ("a1_a2", "A1 contra A2"),
        ):
            print(f"    {etiqueta:<14} {_resumen(grupo, columna)}")
        print(f"    A1-A2 menos A1-GT: {_pareada(grupo, 'a1_a2', 'a1_gt')}")
        print(f"    A1-A2 menos A2-GT: {_pareada(grupo, 'a1_a2', 'a2_gt')}")
    print("\n  por unidad:")
    print(
        tabla.pivot_table(
            index="unidad", columns="estrato", values=["a1_gt", "a2_gt", "a1_a2"]
        ).to_string(float_format=lambda v: f"{v:7.4f}")
    )

    contra_la_fragmentacion(tabla)

    print("\n=== 5. LA IMAGEN MAS DIFICIL: conc_s06_t10 ===")
    for anotadora in ANOTADORAS:
        carpeta = carpeta_de(anotadora)
        zip_roi = carpeta / "conc_s06_t10.zip"
        marca = list(carpeta.glob(f"conc_s06_t10*{MARCA_NO_IDENTIFICABLE}*"))
        if marca and not zip_roi.exists():
            print(f"  {anotadora}: marcada como no identificable")
        elif zip_roi.exists():
            forma = datos.cargar_gt(6, 10).shape
            print(f"  {anotadora}: entrego {len(rois_de(zip_roi, forma))} ROIs")

    tercero = como_tercer_anotador(correspondencia).merge(
        tabla[["unidad", "frame", "a1_a2", "a1_gt", "a2_gt"]],
        on=["unidad", "frame"],
        how="inner",
    )
    print("\n=== 7. EL MODELO COMO TERCER ANOTADOR: TODO POR SOLAPE, SIN GT ===")
    for estrato, grupo in tercero.groupby("estrato"):
        print(f"\n  {estrato} (n={len(grupo)})")
        for columna, etiqueta in (
            ("a1_a2", "A1 contra A2"),
            ("modelo_a1", "modelo contra A1"),
            ("modelo_a2", "modelo contra A2"),
        ):
            print(f"    {etiqueta:<18} {_resumen(grupo, columna)}")
        print(f"    modelo-A1 menos A1-A2: {_pareada(grupo, 'modelo_a1', 'a1_a2')}")
        print(f"    modelo-A2 menos A1-A2: {_pareada(grupo, 'modelo_a2', 'a1_a2')}")
    print("\n  por unidad:")
    print(
        tercero.pivot_table(
            index="unidad", columns="estrato",
            values=["a1_a2", "modelo_a1", "modelo_a2"],
        ).to_string(float_format=lambda v: f"{v:7.4f}")
    )
    tercero.to_csv(config.DIR_PROCESO / "concordancia_tercer_anotador.csv", index=False)

    contra_el_mejor_modelo(tabla)

    directo.to_csv(config.DIR_PROCESO / "concordancia_a1_a2.csv", index=False)
    tabla.to_csv(config.DIR_PROCESO / "concordancia_tres_vias.csv", index=False)


if __name__ == "__main__":
    main()
