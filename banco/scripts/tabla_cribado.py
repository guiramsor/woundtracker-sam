"""La tabla de cribado: une los CSV por entorno y agrega como manda la base.

Agregacion en dos pasos del apartado 6.3: primero dentro de cada lesion,
despues entre las 17. n = 17, no 312.

Intervalos por bootstrap remuestreando **unidades** (apartado 6.4), y las dos
curvas del apartado 6.5, por unidad y por frame, con sus areas, que no coinciden
y no tienen por que.

**La agregacion por familia es provisional.** El representante de cada modelo es
su checkpoint con mejor Dice en el cribado, y esta primera pasada solo ha medido
el checkpoint por defecto de cada uno. Hasta que se midan los demas tamanos, el
representante es el defecto por falta de alternativas, no por haber ganado.

    conda run -n surco python scripts/tabla_cribado.py
"""

import numpy as np
import pandas as pd

from surco import config, metricas, resultados

PATRON = "cribado_surco-*.csv"


def cargar() -> pd.DataFrame:
    """Une los CSV por entorno y comprueba que son de esta corrida.

    El patron excluye los agregados que este mismo script escribe: si
    `cribado_tabla.csv` casara con el, la segunda ejecucion se leeria a si misma.
    La union la hace `resultados`, que es el unico sitio que sabe que un fichero
    de caidos no es un resultado; lo de aqui es la comprobacion, que si es propia
    de esta fase: una fila sin `lado_entrada` es de una corrida anterior.
    """
    ficheros = resultados.ficheros_de(PATRON)
    tabla = resultados.unir(PATRON)
    faltan = tabla["lado_entrada"].isna().sum() if "lado_entrada" in tabla else len(tabla)
    if faltan:
        raise ValueError(
            f"{faltan} filas sin lado_entrada: hay CSV de una corrida anterior. "
            "Vuelve a correr esos entornos en vez de mezclar."
        )
    print(f"{len(ficheros)} entornos, {len(tabla)} pares medidos")
    return tabla


def resumen_por_fila(tabla: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for (modelo, checkpoint), grupo in tabla.groupby(["modelo", "checkpoint"]):
        por_unidad = metricas.por_unidad(grupo)
        valores = por_unidad.to_numpy()
        bajo, alto = metricas.intervalo_bootstrap(
            valores,
            config.N_REMUESTREOS,
            config.NIVEL_CONFIANZA,
            config.SEMILLA_BOOTSTRAP,
        )
        curva_unidad = metricas.curva_de_umbrales(valores)
        curva_frame = metricas.curva_de_umbrales(grupo["dice"].to_numpy())
        filas.append(
            {
                "modelo": modelo,
                "checkpoint": checkpoint,
                "familia": grupo["familia"].iloc[0],
                "n_unidades": len(por_unidad),
                "dice_unidad": metricas.media_entre_unidades(por_unidad),
                "ic_bajo": bajo,
                "ic_alto": alto,
                "dice_frame": float(grupo["dice"].mean()),
                "area_curva_unidad": metricas.area_bajo_curva(curva_unidad),
                "area_curva_frame": metricas.area_bajo_curva(curva_frame),
                "catastroficas_unidad": metricas.fraccion_catastrofica(
                    valores, config.UMBRAL_CATASTROFICO
                ),
                "catastroficos_frame": metricas.fraccion_catastrofica(
                    grupo["dice"].to_numpy(), config.UMBRAL_CATASTROFICO
                ),
                "lado": int(grupo["lado_entrada"].iloc[0]),
                "candidatas": int(grupo["n_candidatas"].median()),
            }
        )
    return pd.DataFrame(filas).sort_values("dice_unidad", ascending=False)


def por_familia(resumen: pd.DataFrame) -> pd.DataFrame:
    """Un representante por modelo, y despues media de representantes."""
    representantes = resumen.loc[resumen.groupby("modelo")["dice_unidad"].idxmax()]
    return (
        representantes.groupby("familia")
        .agg(
            modelos=("modelo", "count"),
            dice_medio=("dice_unidad", "mean"),
            mejor=("dice_unidad", "max"),
            peor=("dice_unidad", "min"),
        )
        .sort_values("dice_medio", ascending=False)
    )


def comparar_dos(tabla: pd.DataFrame, uno: str, otro: str) -> None:
    """Diferencia pareada unidad a unidad entre dos filas (apartado 6.4)."""
    series = {nombre: metricas.celda(tabla, modelo=nombre) for nombre in (uno, otro)}
    if any(serie.empty for serie in series.values()):
        return
    print(f"  {uno} menos {otro}: {metricas.contraste(series[uno], series[otro])}")


def main() -> None:
    pd.set_option("display.width", 220)
    tabla = cargar()
    resumen = resumen_por_fila(tabla)

    print("\n=== TABLA DE CRIBADO: M1 x P1 x S0, n = 17 unidades ===")
    columnas = [
        "modelo", "checkpoint", "familia", "lado", "candidatas",
        "dice_unidad", "ic_bajo", "ic_alto", "catastroficas_unidad",
    ]
    print(resumen[columnas].to_string(index=False, float_format=lambda v: f"{v:7.4f}"))

    print("\n=== EL ORDEN, CONTRA LA RESOLUCION ===")
    print(
        resumen.groupby("lado")["dice_unidad"]
        .agg(["count", "mean", "min", "max"])
        .to_string(float_format=lambda v: f"{v:7.4f}")
    )
    rho = resumen["dice_unidad"].corr(resumen["lado"], method="spearman")
    print(f"  correlacion de Spearman entre Dice y lado de entrada: {rho:+.3f}")
    print(
        "  la pareja que aisla la resolucion es l contra xl de EfficientViT-SAM,\n"
        "  que comparte el resto de la arquitectura:"
    )
    evit = resumen[resumen["modelo"] == "efficientvitsam"]
    if not evit.empty:
        print(
            evit[["checkpoint", "lado", "dice_unidad"]]
            .sort_values("checkpoint")
            .to_string(index=False, float_format=lambda v: f"{v:7.4f}")
        )

    print("\n=== COMPROBACION: el area de cada curva es su media ===")
    iguales_unidad = np.allclose(resumen["area_curva_unidad"], resumen["dice_unidad"])
    iguales_frame = np.allclose(resumen["area_curva_frame"], resumen["dice_frame"])
    print(f"  curva por unidad: {iguales_unidad}   curva por frame: {iguales_frame}")
    print(
        "  las dos medias difieren en "
        f"{(resumen['dice_unidad'] - resumen['dice_frame']).abs().mean():.4f} de media: "
        "por eso son dos curvas y no una"
    )

    print("\n=== POR FAMILIA (provisional: un solo checkpoint por modelo) ===")
    print(por_familia(resumen).to_string(float_format=lambda v: f"{v:7.4f}"))

    print("\n=== DIFERENCIAS PAREADAS QUE CONTESTAN LA HIPOTESIS ===")
    for uno, otro in (
        ("microsam_lm", "sam21"),
        ("microsam_lm", "microsam_em"),
        ("microsam_lm", "medsam2"),
        ("sam1", "sam21"),
    ):
        comparar_dos(tabla, uno, otro)

    try:
        coste = resultados.unir("coste_*.csv").sort_values("coste_por_imagen_s")
    except FileNotFoundError:
        coste = None  # no se ha corrido comprobar_carga en ningun entorno
    if coste is not None:
        print("\n=== COSTE, dos columnas (apartado 6.6) ===")
        print(coste.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
        coste.to_csv(config.DIR_PROCESO / "tabla_coste.csv", index=False)

    caidos = sorted(config.DIR_PROCESO.glob("cribado_caidos_*.csv"))
    if caidos:
        print("\n=== FILAS NO MEDIDAS ===")
        print(
            pd.concat([pd.read_csv(f) for f in caidos], ignore_index=True)
            .to_string(index=False)
        )

    resumen.to_csv(config.DIR_PROCESO / "tabla_cribado.csv", index=False)
    print(f"\ntabla en {config.DIR_PROCESO / 'tabla_cribado.csv'}")


if __name__ == "__main__":
    main()
