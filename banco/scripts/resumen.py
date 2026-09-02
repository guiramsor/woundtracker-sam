"""Una tabla unica para compartir: `salidas/resumen.csv` y su LEEME.

**No calcula ningun numero nuevo.** Une lo que ya dejaron el cribado, la rejilla
de profundidad, M4, el selector y los metodos clasicos, y lo ordena para que la
estructura de familias se vea al abrirlo.

Una fila por checkpoint: los 28 que midio el cribado mas los tres metodos
clasicos. Las columnas de coste van al final y con prefijo `COSTE_`, para que no
se lean como calidad: un modelo rapido y malo sigue siendo malo.

    conda run -n surco python scripts/resumen.py
"""

import pandas as pd

from surco import config, metricas, resultados

# Configuracion del cribado, que es la unica en la que se midieron los 28
# checkpoints: la misma para todos (apartado 5.3).
CRIBADO = "M1 x P1 x S0"

# Los cuatro que pasaron a profundidad, con su representante (apartado 5.3).
FINALISTAS = config.REJILLA_PROFUNDIDAD


def _mejor_celda_de_la_rejilla() -> dict:
    """Por modelo, su mejor Dice en toda la rejilla y donde lo consigue.

    Las tres fuentes y el desempate a favor de S0 los sabe `resultados`, y se
    le preguntan en vez de repetirlos: este fichero llego a barrerlas por su
    cuenta, que era el quinto sitio del defecto que ese modulo existe para
    cerrar.
    """
    por_modelo: dict[str, dict] = {}
    for clave, dice in resultados.celdas_de_modelos().items():
        por_modelo.setdefault(clave[0], {})[clave] = dice

    mejores = {}
    for modelo, suyas in por_modelo.items():
        (_, modo, protocolo, selector), dice = resultados.mejor(suyas)
        mejores[modelo] = (dice, f"{modo} x {protocolo} x {selector}")
    return mejores


def filas_de_modelos() -> pd.DataFrame:
    cribado = pd.read_csv(config.DIR_PROCESO / "tabla_cribado.csv")
    coste = pd.read_csv(config.DIR_PROCESO / "tabla_coste.csv")
    mejores = _mejor_celda_de_la_rejilla()

    junto = cribado.merge(coste, on=["modelo", "checkpoint"], how="left")
    es_finalista = junto.apply(
        lambda f: FINALISTAS.get(f["modelo"]) == f["checkpoint"], axis=1
    )
    return pd.DataFrame(
        {
            "familia": junto["familia"],
            "modelo": junto["modelo"],
            "checkpoint": junto["checkpoint"],
            "lado_entrada_px": junto["lado"],
            "n_candidatas": junto["candidatas"],
            "dice_cribado": junto["dice_unidad"].round(4),
            # Del Dice ya redondeado, para que la columna se pueda comprobar
            # con una calculadora desde la de al lado.
            "iou_cribado": junto["dice_unidad"]
            .round(4)
            .map(metricas.iou_desde_dice)
            .round(4),
            "ic_bajo": junto["ic_bajo"].round(4),
            "ic_alto": junto["ic_alto"].round(4),
            "unidades_catastroficas": junto["catastroficas_unidad"].round(4),
            "paso_a_profundidad": es_finalista.map({True: "si", False: "no"}),
            "mejor_dice_rejilla": [
                round(mejores[m][0], 4) if e else None
                for m, e in zip(junto["modelo"], es_finalista)
            ],
            "configuracion_del_mejor": [
                mejores[m][1] if e else None
                for m, e in zip(junto["modelo"], es_finalista)
            ],
            "COSTE_por_imagen_s": junto["coste_por_imagen_s"].round(4),
            "COSTE_por_prompt_adicional_s": junto["coste_prompt_adicional_s"].round(4),
        }
    )


def filas_de_clasicos() -> pd.DataFrame:
    """Los tres metodos clasicos, en P1, que es el protocolo del cribado.

    Se toma su celda de P1 y no la mejor que tienen, para que la columna
    `dice_cribado` signifique lo mismo en todas las filas de la tabla.
    """
    clasicos = pd.read_csv(config.DIR_PROCESO / "tabla_clasicos.csv")
    en_p1 = clasicos[clasicos["protocolo"] == "P1"]
    return pd.DataFrame(
        {
            "familia": "clasico",
            "modelo": en_p1["metodo"],
            "checkpoint": "-",
            "lado_entrada_px": None,  # no reescalan: trabajan sobre el frame
            "n_candidatas": 1,
            "dice_cribado": en_p1["dice"].round(4),
            "iou_cribado": en_p1["dice"].round(4).map(metricas.iou_desde_dice).round(4),
            "ic_bajo": en_p1["ic_bajo"].round(4),
            "ic_alto": en_p1["ic_alto"].round(4),
            "unidades_catastroficas": en_p1["catastroficas"].round(4),
            "paso_a_profundidad": "si",
            "mejor_dice_rejilla": None,
            "configuracion_del_mejor": None,
            "COSTE_por_imagen_s": None,
            "COSTE_por_prompt_adicional_s": None,
        }
    )


def _mejor_clasico(tabla: pd.DataFrame) -> pd.DataFrame:
    """La mejor celda de cada metodo clasico, preguntando a `resultados`.

    Antes salia de `tabla_clasicos.csv` a secas, que solo tiene la variante de
    una mascara: el filtro de cresta aparecia con 0.4165 cuando con el selector
    llega a 0.4312. Era el mismo defecto que sesgaba los techos de SAM, en la
    otra direccion, y corregir uno y no el otro habria hecho que esta tabla
    favoreciera a SAM por seleccion de errores.

    **La variante se escribe tal cual, sin traducir "una mascara" a S0.** El
    numero seria el mismo, porque con una sola candidata no hay nada que elegir,
    pero la etiqueta afirmaria que corrio un selector que no existio y taparia
    que solo el filtro de cresta admite la variante de tres candidatas.
    """
    celdas = resultados.celdas_de_clasicos()
    por_metodo = {}
    for metodo in {clave[0] for clave in celdas}:
        suyas = {c: d for c, d in celdas.items() if c[0] == metodo}
        (_, protocolo, variante), dice = resultados.mejor(suyas)
        por_metodo[metodo] = (round(dice, 4), f"M1 x {protocolo} x {variante}")
    es_clasico = tabla["familia"] == "clasico"
    tabla.loc[es_clasico, "mejor_dice_rejilla"] = [
        por_metodo[m][0] for m in tabla.loc[es_clasico, "modelo"]
    ]
    tabla.loc[es_clasico, "configuracion_del_mejor"] = [
        por_metodo[m][1] for m in tabla.loc[es_clasico, "modelo"]
    ]
    return tabla


def ordenar(tabla: pd.DataFrame) -> pd.DataFrame:
    """Por familia en el orden del config, y dentro de cada una por Dice."""
    orden = {familia: i for i, familia in enumerate(config.FAMILIAS)}
    return (
        tabla.assign(_orden=tabla["familia"].map(orden))
        .sort_values(["_orden", "dice_cribado"], ascending=[True, False])
        .drop(columns="_orden")
        .reset_index(drop=True)
    )


def main() -> None:
    pd.set_option("display.width", 250)
    tabla = ordenar(
        _mejor_clasico(
            pd.concat([filas_de_modelos(), filas_de_clasicos()], ignore_index=True)
        )
    )
    destino = config.DIR_SALIDAS / "resumen.csv"
    tabla.to_csv(destino, index=False)
    (config.DIR_SALIDAS / "LEEME_resumen.md").write_text(LEEME, encoding="utf-8")

    print(tabla.to_string(index=False))
    print(f"\n{len(tabla)} filas en {destino}")
    print(f"y el LEEME en {config.DIR_SALIDAS / 'LEEME_resumen.md'}")


LEEME = """# Cómo leer `resumen.csv`

Una fila por checkpoint: los 28 que se midieron en el cribado más los tres métodos clásicos.

## Qué significa cada columna

- **familia**: el grupo del modelo según a qué se adaptó: `linaje` (SAM original y SAM 2.1),
  `eficiencia` (destilados y podados), `microscopia`, `radiologia` y `clasico` (sin
  aprendizaje). Comparar entre familias es el resultado principal del trabajo.
- **modelo** y **checkpoint**: el modelo y el tamaño concreto de pesos. Un modelo puede
  tener varios.
- **lado_entrada_px**: el lado al que ese modelo reescala la imagen antes de procesarla. Con
  un surco de unos 17 px de ancho sobre 1024, a 512 quedan 8 y a 256 quedan 4. Vacío en los
  clásicos, que no reescalan.
- **n_candidatas**: cuántas máscaras propone por cada prompt. Con una sola, el selector no
  tiene nada que elegir.
- **dice_cribado**: la medida principal de calidad, de 0 a 1. Todos los checkpoints se
  midieron en **la misma configuración**, `M1 x P1 x S0`, para que la columna sea comparable
  fila a fila.
- **iou_cribado**: el mismo resultado en IoU, que es `Dice / (2 - Dice)`. Va porque la
  literatura lo usa, pero **no aporta información**: es monótona en el Dice, así que el orden
  de las filas es idéntico en las dos columnas (apartado 6.1).
- **ic_bajo** e **ic_alto**: el intervalo de confianza del 95 % **del Dice**. Dos filas cuyos
  intervalos se solapan mucho no están claramente separadas.
- **unidades_catastroficas**: fracción de las 17 lesiones en las que el Dice no llega a 0.1,
  es decir donde el modelo falla del todo en vez de fallar un poco. Cuanto más bajo, mejor.
- **paso_a_profundidad**: si ese checkpoint pasó de la primera pasada a la rejilla completa.
  Pasaron cuatro, uno por familia, más los tres clásicos.
- **mejor_dice_rejilla**: para los que pasaron, su mejor resultado probando todas las
  combinaciones, no solo la del cribado.
- **configuracion_del_mejor**: con qué combinación lo consigue, en el formato
  `modo x protocolo x selector`.
- **COSTE_por_imagen_s** y **COSTE_por_prompt_adicional_s**: segundos. Van al final y con
  prefijo a propósito: **son coste, no calidad**. La primera incluye procesar la imagen; la
  segunda es lo que cuesta un clic más sobre esa misma imagen ya procesada.

## Los tres factores que aparecen en `configuracion_del_mejor`

**Modo**, qué se le da al modelo en cada frame:

- **M0**: se segmenta una vez en el primer frame y esa máscara se copia al resto. Es el suelo.
- **M1**: se vuelve a segmentar cada frame con los mismos clics, sin memoria.
- **M3**: propagación con memoria de vídeo. Solo lo soportan los modelos que la tienen.
- **M4**: la máscara del frame anterior entra como pista, junto con los clics.

**Protocolo**, cuántos clics pone el biólogo. Se anotan cinco una sola vez y los cuatro
protocolos son subconjuntos: **P1** un clic en el centro; **P2** ese más uno de fondo; **P3**
tres clics sobre el surco; **P4** los cinco.

**Selector**, cuál de las máscaras propuestas se acepta: **S0** la que el propio modelo
puntúa más alto, que es su comportamiento por defecto; **S1** un criterio propio que además
mira dónde están los clics y qué forma tiene la máscara.

En dos filas de método clásico el selector aparece como **`una mascara`**, y no es un
descuido: el umbral local y el watershed **producen una sola máscara**, así que no hay nada
que seleccionar y ningún selector llegó a correr. Solo el filtro de cresta admite la variante
de tres candidatas, porque sus tres escalas de sigma dan tres respuestas. Poner `S0` ahí
habría hecho invisible esa diferencia entre los tres métodos.

## Cómo se calculó el Dice

- **Se compara** la máscara predicha contra la trazada a mano, lesión a lesión.
- **Sobre qué frames**: desde el frame siguiente al del impacto hasta el último anotado. El
  frame del impacto no entra, porque ahí la máscara es la entrada y no la salida.
- **Cómo se promedia**: primero dentro de cada lesión y después entre las 17. **El tamaño de
  muestra es n = 17 lesiones**, no los 295 pares imagen-lesión, porque los frames de una misma
  lesión son casi idénticos entre sí y contarlos por separado estrecharía los intervalos de
  forma artificial.
"""


if __name__ == "__main__":
    main()
