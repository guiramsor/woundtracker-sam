"""Comprueba que `salidas/` esta completa y cuadra con la memoria.

Sirve para una sola pregunta: **si esta copia del proyecto tiene los resultados
que dice tener**. Recorre los CSV que dejan los scripts, recalcula un puñado de
numeros ancla y los compara contra los que estan escritos en la memoria.

**Los valores esperados estan aqui copiados a mano desde la memoria**, y eso es
deliberado: si se derivaran de los mismos CSV que se quieren comprobar, la
comprobacion pasaria siempre y no diria nada. Son la memoria mirando al codigo, y
si un dia dejan de cuadrar, o se ha movido un resultado o se ha movido el texto.

No cubre todos los numeros del trabajo, solo uno o dos por fase, elegidos porque
resumen la fase entera: si el Dice de micro-sam LM en el cribado cuadra, es que
el cribado entero se leyo bien.
"""

import numpy as np
import pandas as pd

from surco import config, metricas, resultados

# Tolerancia de la comparacion. Los numeros de la memoria estan a cuatro
# decimales, asi que media unidad del ultimo decimal es lo que cabe esperar de
# haberlos redondeado al escribirlos.
TOLERANCIA = 5e-5


def _leer(nombre: str) -> pd.DataFrame:
    """Busca arriba y despues en `proceso/`.

    Las dos tablas finales viven en la raiz de `salidas/` y el material
    intermedio debajo, asi que aqui se mira en los dos sitios en vez de tener
    que recordar cual es cual.
    """
    for ruta in (config.DIR_SALIDAS / nombre, config.DIR_PROCESO / nombre):
        if ruta.exists():
            return pd.read_csv(ruta)
    raise FileNotFoundError(f"falta {nombre} en salidas/ ni en salidas/proceso/")


def _dice(tabla: pd.DataFrame, columna: str = "dice", **filtros) -> float:
    filtro = pd.Series(True, index=tabla.index)
    for clave, valor in filtros.items():
        filtro &= tabla[clave] == valor
    return metricas.media_entre_unidades(metricas.por_unidad(tabla[filtro], columna))


# ---------------------------------------------------------------------------
# Una comprobacion por fase. Cada entrada es (apartado, que se mide, esperado,
# como se recalcula).
# ---------------------------------------------------------------------------


def _f1() -> list[tuple]:
    unidades = _leer("unidades.csv")
    inventario = _leer("inventario_gt.csv")
    return [
        ("F1", "unidades", config.N_UNIDADES, len(unidades)),
        ("F1", "pares (unidad, frame)", config.N_PARES, len(inventario)),
        (
            "F1",
            "frames anotados",
            config.N_FRAMES_ANOTADOS,
            inventario.groupby(["serie", "frame"]).ngroups,
        ),
    ]


def _f2() -> list[tuple]:
    pares = _leer("deriva.csv")
    unidades = _leer("deriva_por_unidad.csv")
    return [
        ("F2", "pares con el clic fuera", 183, int((~pares["clic_dentro"]).sum())),
        ("F2", "pares medidos", 295, len(pares)),
        ("F2", "unidades que se salen", 15, int(unidades["sale_alguna_vez"].sum())),
    ]


def _f7() -> list[tuple]:
    cribado = resultados.unir("cribado_surco-*.csv")
    resumen = _leer("tabla_cribado.csv")
    rho = resumen["dice_unidad"].corr(resumen["lado"], method="spearman")
    return [
        ("F7", "filas medidas", 28, resumen.shape[0]),
        # Con el checkpoint fijado: el cribado midio los tres tamanos de
        # micro-sam LM, y sin fijarlo esto promediaria los tres.
        ("F7", "micro-sam LM vit_b", 0.3389,
         _dice(cribado, modelo="microsam_lm", checkpoint="vit_b")),
        ("F7", "SAM-Med2D sam_med2d", 0.3284,
         _dice(cribado, modelo="sammed2d", checkpoint="sam_med2d")),
        ("F7", "Spearman Dice contra lado", -0.039, round(float(rho), 3)),
    ]


def _f8() -> list[tuple]:
    rejilla = resultados.unir("profundidad_surco-*.csv")
    return [
        ("F8", "micro-sam LM M1 x P4", 0.4429,
         _dice(rejilla, modelo="microsam_lm", modo="M1", protocolo="P4")),
        ("F8", "SAM 2.1 M3 x P4", 0.4269,
         _dice(rejilla, modelo="sam21", modo="M3", protocolo="P4")),
    ]


def _f9() -> list[tuple]:
    clasicos = _leer("clasicos.csv")
    return [
        ("F9", "Sato x P3", 0.4165, _dice(clasicos, modelo="sato", protocolo="P3")),
        ("F9", "watershed x P3", 0.1299,
         _dice(clasicos, modelo="watershed", protocolo="P3")),
        # El 3 y el "2 6 10" van escritos, no sacados de `config.SIGMAS_SATO`:
        # con el valor esperado derivado del config, esta comprobacion seguiria
        # al config en vez de vigilarlo, que es justo lo contrario de lo que
        # este modulo dice hacer.
        ("F9", "escalas de Sato en paso 4", 3,
         len(_leer("robustez_sigma.csv").query("paso == 4")["sigmas"].iloc[0].split())),
        ("F9", "el paso adoptado son esas escalas", 3, len(config.SIGMAS_SATO)),
    ]


def _f10() -> list[tuple]:
    seleccion = _leer("selector.csv")
    filtro = {"modelo": "sam21", "modo": "M1", "protocolo": "P4"}
    diferencias = metricas.diferencia_pareada(
        metricas.por_unidad(
            seleccion[np.logical_and.reduce([seleccion[k] == v for k, v in filtro.items()])],
            "dice_s1",
        ),
        metricas.por_unidad(
            seleccion[np.logical_and.reduce([seleccion[k] == v for k, v in filtro.items()])],
            "dice_s0",
        ),
    )
    return [
        ("F10", "micro-sam LM M1 x P4 x S1", 0.5264,
         _dice(seleccion, "dice_s1", modelo="microsam_lm", modo="M1", protocolo="P4")),
        ("F10", "SAM 2.1 M1 x P4, S1 menos S0", 0.1480, float(diferencias.mean())),
    ]


def _m4() -> list[tuple]:
    m4 = resultados.unir("m4_surco-*.csv")
    return [
        ("M4", "SAM 2.1 S0 x P4", 0.5134,
         _dice(m4, modelo="sam21", selector="S0", protocolo="P4")),
        ("M4", "micro-sam LM S1 x P4", 0.5517,
         _dice(m4, modelo="microsam_lm", selector="S1", protocolo="P4")),
        ("M4", "SAM-Med2D S0 x P3", 0.0808,
         _dice(m4, modelo="sammed2d", selector="S0", protocolo="P3")),
        # El +0.1063 que sostiene el titular del 5.5: sin S1, micro-sam LM se
        # queda en 0.4454, por debajo del techo de S0, y no seria la mejor celda
        # del trabajo. Se ancla la diferencia y no solo sus dos extremos porque
        # es la diferencia lo que esta escrito.
        ("M4", "micro-sam LM S1 menos S0 en P4", 0.1063,
         _dice(m4, modelo="microsam_lm", selector="S1", protocolo="P4")
         - _dice(m4, modelo="microsam_lm", selector="S0", protocolo="P4")),
    ]


def _perdida_jpeg() -> list[tuple]:
    """La perdida por JPEG, medida sobre el checkpoint que hace la comparacion.

    Es el ancla que faltaba. La primera version del script llamaba a
    `modelos.crear("sam21")` sin checkpoint, asi que midio sobre `large` mientras
    la comparacion M1 contra M3 la corre `tiny`: un numero escrito que dejo de
    coincidir con el que se calcula, y nada aviso.

    No se ancla la diferencia pareada. Vale -0.00003 y TOLERANCIA es 5e-5, o sea
    mayor que el propio valor: esa fila daria True pasara lo que pasara. Se
    anclan las dos cosas que si discriminan, sobre que checkpoint se midio y la
    diferencia por pixel, que es propiedad del codec y no se mueve al cambiar de
    modelo.
    """
    tabla = _leer("perdida_jpeg.csv")
    medido = set(tabla.get("checkpoint", pd.Series(dtype=str)).unique())
    return [
        ("A5.1", "perdida jpeg sobre el checkpoint de la rejilla", 1,
         int(medido == {config.REJILLA_PROFUNDIDAD["sam21"]})),
        ("A5.1", "perdida jpeg, diferencia por pixel", 0.093,
         round(float(tabla["dif_pixel_media"].mean()), 3)),
    ]


def _apartado7() -> list[tuple]:
    tres = _leer("concordancia_tres_vias.csv")
    tercero = _leer("concordancia_tercer_anotador.csv")
    # El techo sale de `concordancia_tres_vias.csv`, que son las dos imagenes por
    # unidad que fija el diseno. `concordancia_a1_a2.csv` trae ademas las lesiones
    # que comparten imagen con una unidad muestreada, asi que seis unidades
    # aparecen alli tres veces: promediar sus 37 pares en plano las pesa de mas,
    # contra la regla del apartado 6.3, y daba 0.5664 y 0.4604.
    por_estrato = tres.groupby("estrato")["a1_a2"].mean()
    return [
        ("A7", "techo humano temprano", 0.5894, float(por_estrato["temprano"])),
        ("A7", "techo humano tardio", 0.5058, float(por_estrato["tardio"])),
        ("A7", "A1 contra GT, tardio", 0.6870,
         float(tres[tres["estrato"] == "tardio"]["a1_gt"].mean())),
        # Con la mejor celda, que desde que se midio M4 es M4 x P4 x S1. Antes
        # esto valia 0.4340, que era el mismo numero para M1 x P4 x S1.
        ("A7", "modelo contra A1, tardio", 0.5410,
         float(tercero[tercero["estrato"] == "tardio"]["modelo_a1"].mean())),
    ]


def mejor_celda_medida() -> dict:
    """La celda con mas Dice de todo lo medido, barriendo las tres fuentes.

    Delega en `surco.resultados`, que es el unico sitio que sabe cuantas fuentes
    hay. Se conserva este nombre porque es el que usan `_mejor_celda` de aqui
    abajo y `tests/test_config.py`.
    """
    return resultados.mejor_celda_de_modelo()


def _mejor_celda() -> list[tuple]:
    """Que la celda declarada en el config sea de verdad la mejor medida."""
    medida = mejor_celda_medida()
    declarada = config.MEJOR_CELDA
    iguales = all(
        medida[clave] == declarada[clave]
        for clave in ("modelo", "modo", "protocolo", "selector")
    )
    return [
        ("config", "MEJOR_CELDA es la mejor medida", 1, int(iguales)),
        ("config", "y su Dice", 0.5517, medida["dice"]),
    ]


def _techo_s0() -> list[tuple]:
    """El techo de SAM con S0 del 5.4, que estuvo afirmado en dos sitios a la vez.

    La tabla de techos decia 0.5134 y el parrafo de arriba seguia diciendo 0.4429,
    que es el maximo de **la rejilla de F8** y no el de todo lo medido con S0. Los
    dos numeros eran correctos por separado, cada uno de su fuente, y por eso no
    salto nada: lo que faltaba era una linea que preguntara **cual es el maximo
    con S0 mirando las tres**, que es lo que esta hace.
    """
    celdas = resultados.celdas_de_modelos()
    clave, dice = resultados.mejor({c: d for c, d in celdas.items() if c[3] == "S0"})
    return [
        ("A5.4", "el techo de S0 es SAM 2.1 M4 x P4", 1,
         int(clave == ("sam21", "M4", "P4", "S0"))),
        ("A5.4", "y su Dice", 0.5134, dice),
    ]


def _contrastes() -> list[tuple]:
    """La tabla de comparaciones pareadas, contra lo escrito en la memoria.

    Se comprueba aparte de las fases porque `contrastes.csv` las reune todas: si
    una de estas cuadra, es que la reunion no ha movido nada por el camino.
    """
    tabla = _leer("contrastes.csv")

    def diferencia(a: str, b: str) -> float:
        fila = tabla[(tabla["a"] == a) & (tabla["b"] == b)]
        if fila.empty:
            raise FileNotFoundError(f"falta el contraste {a!r} contra {b!r}")
        return float(fila["diferencia"].iloc[0])

    return [
        ("contrastes", "filas", 232, len(tabla)),
        ("contrastes", "SAM 2.1 M4 menos M1 en P4", 0.1829,
         diferencia("sam21 M4 x P4 x S0", "sam21 M1 x P4 x S0")),
        # El nombre del clasico lleva la variante desde que se vio que habia tres
        # numeros distintos detras de "sato x P3". Esta linea decia el nombre
        # corto, y cambiarlo la rompio: es lo que tenia que pasar, porque si no
        # se entera es que no estaba comprobando contra que se compara.
        ("contrastes", "Sato de una mascara menos SAM 2.1 en P3", 0.1773,
         diferencia("sato x P3 x una mascara", "sam21 M1 x P3")),
        # Las dos piezas del 5.5, cada una contra el mismo lado derecho: la
        # primera aisla el criterio y la segunda el trocear la respuesta. En P4
        # las dos van en direcciones contrarias, y por eso se vigilan las dos.
        ("contrastes", "Sato S1 menos S0 en P3", 0.0313,
         diferencia("sato x P3 x S1", "sato x P3 x S0")),
        ("contrastes", "Sato una mascara menos S0 en P4", 0.0617,
         diferencia("sato x P4 x una mascara", "sato x P4 x S0")),
        ("contrastes", "A1-A2 menos A1-GT, tardio", -0.1811,
         diferencia("a1_a2 (tardio)", "a1_gt (tardio)")),
    ]


COMPROBACIONES = (
    _f1, _f2, _f7, _f8, _f9, _f10, _m4, _perdida_jpeg, _apartado7, _contrastes,
    _mejor_celda, _techo_s0,
)


def comprobar() -> pd.DataFrame:
    """Una fila por numero ancla, con lo esperado, lo obtenido y si cuadra."""
    filas = []
    for funcion in COMPROBACIONES:
        try:
            for apartado, que, esperado, obtenido in funcion():
                filas.append(
                    {
                        "apartado": apartado,
                        "que": que,
                        "memoria": esperado,
                        "salidas": round(float(obtenido), 4),
                        "cuadra": abs(float(obtenido) - esperado) <= TOLERANCIA,
                    }
                )
        except FileNotFoundError as falta:
            filas.append(
                {
                    "apartado": funcion.__name__.strip("_").upper(),
                    "que": str(falta),
                    "memoria": None,
                    "salidas": None,
                    "cuadra": False,
                }
            )
    return pd.DataFrame(filas)


def resumen(tabla: pd.DataFrame) -> str:
    fallan = tabla[~tabla["cuadra"]]
    if fallan.empty:
        return f"Las {len(tabla)} comprobaciones cuadran: la copia esta completa."
    return (
        f"{len(fallan)} de {len(tabla)} NO cuadran. O falta correr algo, o un "
        "resultado se ha movido respecto a lo escrito en la memoria."
    )
