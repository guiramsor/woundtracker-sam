"""El factor selector del apartado 5.5: S0 contra S1.

**S0** es el comportamiento por defecto de SAM, quedarse con la candidata de
mayor puntuacion del modelo, y ya vive en `Candidatas.mejor()`. Aqui esta S1, el
criterio propio.

**S1 esta reformulado, y la reformulacion es un resultado.** Como estaba
definido en el trabajo anterior, su puerta tenia dos guardas: descartar la
candidata si `area / area_nucleo > FRAC_MAX`, y descartarla si no contiene el
clic. La primera necesita delimitar el nucleo, y F6 midio que en w1CSU488 no se
puede: es la misma cadena que tumbo a M2. Con ella cae tambien el z tal como
estaba, que se calculaba contra el nucleoplasma.

Lo que queda, todo medible sin segmentar el nucleo:

1. **Puerta.** Sobrevive la candidata que **contiene el clic 0** y que **no
   contiene ningun clic negativo**. La segunda guarda solo existe en P2 y P4,
   que son los unicos protocolos con clics de fondo: **S1 tiene mas niveles en
   unos protocolos que en otros**, y eso es parte de lo que se mide.
2. **Ranking.** z descendente, y desempate por elongacion descendente, que es la
   forma del surco.
3. **Respaldo.** Si no sobrevive ninguna, la de mayor z, con bandera.

El z es robusto, de mediana y MAD, y se calcula **contra el fondo local de la
ventana del clic**: la misma region de `config.LADO_REGION_CLIC_PX` que usan los
metodos clasicos del apartado 5.4, y no contra el nucleoplasma.

Reordenar candidatas ya calculadas es aritmetica: no consume GPU y no hay que
volver a correr ningun modelo.
"""

import numpy as np

from surco import clasicos, config, geometria, modelos

# Sin candidata dentro de la ventana no hay nada que puntuar, y la candidata
# tiene que ordenarse por detras de cualquiera que si la tenga.
Z_SIN_MUESTRA = -np.inf


def _mad(valores: np.ndarray) -> float:
    """Desviacion absoluta mediana, sin el factor de escala a sigma.

    El 1.4826 se omite a proposito: multiplica todos los z por la misma
    constante, y el z aqui solo ordena. Meterlo daria la falsa impresion de que
    el numero es comparable con una sigma gaussiana.
    """
    return float(np.median(np.abs(valores - np.median(valores))))


def z_robusto(gris: np.ndarray, candidata: np.ndarray, punto) -> float:
    """Cuanto se separa la candidata del fondo local, en unidades de MAD.

    El fondo local son los pixeles de la ventana del clic que la candidata deja
    fuera. Si la ventana es plana el MAD es cero: entonces cualquier diferencia
    es infinita y una diferencia nula es cero, que es lo que dicen los datos sin
    necesidad de inventar un suelo.
    """
    ventana, (fila0, columna0) = clasicos.recortar(gris, punto)
    lado = ventana.shape[0]
    dentro = candidata[fila0 : fila0 + lado, columna0 : columna0 + lado]
    if not dentro.any() or dentro.all():
        return Z_SIN_MUESTRA
    diferencia = float(np.median(ventana[dentro])) - float(np.median(ventana[~dentro]))
    dispersion = _mad(ventana[~dentro])
    if dispersion == 0:
        return 0.0 if diferencia == 0 else float(np.sign(diferencia) * np.inf)
    return diferencia / dispersion


def elongacion(candidata: np.ndarray) -> float:
    """Eje mayor entre eje menor. Una candidata vacia no tiene forma."""
    if not candidata.any():
        return 0.0
    return float(geometria.medir(candidata)["elongacion"])


def contiene(candidata: np.ndarray, punto) -> bool:
    return bool(candidata[int(punto[0]), int(punto[1])])


def pasa_la_puerta(candidata: np.ndarray, puntos, etiquetas) -> bool:
    """Contiene el clic 0 y no contiene ningun clic negativo.

    El clic 0 y no "algun positivo": es el unico que existe en los cuatro
    protocolos, asi que es el unico con el que la puerta significa lo mismo en
    toda la rejilla.
    """
    if not contiene(candidata, puntos[0]):
        return False
    return not any(
        contiene(candidata, punto)
        for punto, etiqueta in zip(puntos, etiquetas)
        if etiqueta == 0
    )


def rasgos_de(candidatas, gris: np.ndarray, puntos, etiquetas) -> list[dict]:
    """Un diccionario por candidata: lo que la puerta y el ranking necesitan."""
    return [
        {
            "indice": indice,
            "pasa": pasa_la_puerta(mascara, puntos, etiquetas),
            "z": z_robusto(gris, mascara, puntos[0]),
            "elongacion": elongacion(mascara),
        }
        for indice, mascara in enumerate(candidatas.mascaras)
    ]


def _por_z(rasgo: dict) -> tuple[float, float]:
    """El orden de S1: z descendente, desempate por elongacion descendente."""
    return rasgo["z"], rasgo["elongacion"]


def elegir(candidatas, gris: np.ndarray, puntos, etiquetas) -> tuple[int, bool]:
    """El indice que elige S1, y si ha tenido que tirar del respaldo.

    Devuelve indice y bandera en vez de la mascara para que quien llame pueda
    contar cuantas veces actua el respaldo, que es una de las cosas que el
    apartado 5.5 quiere saber.
    """
    rasgos = rasgos_de(candidatas, gris, puntos, etiquetas)
    supervivientes = [rasgo for rasgo in rasgos if rasgo["pasa"]]
    respaldo = not supervivientes
    mejor = max(rasgos if respaldo else supervivientes, key=_por_z)
    return mejor["indice"], respaldo


def elegir_solo_puerta(candidatas, gris: np.ndarray, puntos, etiquetas) -> int:
    """S1 sin su ranking: la puerta filtra y despues manda el score del modelo.

    Es una de las dos mitades en que se descompone S1 para saber cual sostiene
    el resultado. Si no sobrevive ninguna, el respaldo es S0, que es lo que
    haria el pipeline sin selector.
    """
    rasgos = rasgos_de(candidatas, gris, puntos, etiquetas)
    supervivientes = [rasgo["indice"] for rasgo in rasgos if rasgo["pasa"]]
    if not supervivientes:
        return int(np.argmax(candidatas.puntuaciones))
    return max(supervivientes, key=lambda indice: candidatas.puntuaciones[indice])


def elegir_solo_ranking(candidatas, gris: np.ndarray, puntos, etiquetas) -> int:
    """S1 sin su puerta: manda el z sobre todas las candidatas.

    La otra mitad. Es tambien, por construccion, lo que hace el respaldo de S1
    cuando la puerta se queda sin supervivientes.
    """
    rasgos = rasgos_de(candidatas, gris, puntos, etiquetas)
    return max(rasgos, key=_por_z)["indice"]


def mascara_de(candidatas, gris: np.ndarray, puntos, etiquetas):
    """La mascara que elige S1, con la bandera de respaldo."""
    indice, respaldo = elegir(candidatas, gris, puntos, etiquetas)
    return candidatas.mascaras[indice], respaldo


# ---------------------------------------------------------------------------
# Releer lo que F8 guardo
# ---------------------------------------------------------------------------
# Las candidatas se guardaron empaquetadas a bits sobre el ultimo eje, asi que
# la forma se conserva salvo el ancho, que se recupera del config. Vive aqui y
# no en cada script para que la rejilla, el estudio de concordancia y las
# figuras lean exactamente lo mismo de la misma forma.


def desempaquetar(guardado: dict, clave: str) -> modelos.Candidatas:
    """Deshace el `np.packbits` de F8 sobre una entrada del fichero."""
    mascaras = np.unpackbits(guardado[clave], axis=-1, count=config.LADO_PX)
    return modelos.Candidatas(
        mascaras.astype(bool), np.asarray(guardado[f"{clave}_score"], dtype=float)
    )


def ruta_de_candidatas(
    nombre: str, checkpoint: str, serie: int, etiqueta: int, prefijo: str = ""
):
    """El fichero de candidatas de una unidad. El patron vive en el config."""
    return config.DIR_CANDIDATAS / config.PATRON_CANDIDATAS.format(
        prefijo=prefijo, nombre=nombre, checkpoint=checkpoint,
        serie=serie, etiqueta=etiqueta,
    )


def como_se_guardo(modo: str, selector_activo: str) -> tuple[str, str]:
    """Prefijo del fichero y nombre del modo dentro de el, para esa celda.

    M4 no se guardo como los demas y hay que saberlo para releerlo: va en
    ficheros con prefijo `m4_`, y **una cadena por selector**, asi que su clave
    lleva el selector pegado (`M4S0`, `M4S1`) en vez de un `M4` a secas. No es
    un capricho del formato: en M4 lo que se reencadena es lo que eligio el
    selector, asi que S0 y S1 son dos corridas distintas y no dos lecturas de
    las mismas candidatas.
    """
    if modo == "M4":
        return "m4_", f"M4{selector_activo}"
    return "", modo


def candidatas_guardadas(
    nombre: str, checkpoint: str, serie: int, etiqueta: int, modo: str,
    protocolo: str, frame: int, selector_activo: str = "S0",
) -> modelos.Candidatas | None:
    """Las candidatas de un prompt concreto, o None si esa celda no se midio."""
    prefijo, modo_guardado = como_se_guardo(modo, selector_activo)
    ruta = ruta_de_candidatas(nombre, checkpoint, serie, etiqueta, prefijo)
    clave = f"{modo_guardado}_{protocolo}_t{frame:02d}"
    if not ruta.exists():
        return None
    with np.load(ruta) as fichero:
        if clave not in fichero:
            return None
        return desempaquetar(dict(fichero), clave)


def elegir_con(nombre: str, candidatas, gris: np.ndarray, puntos, etiquetas) -> int:
    """El indice que elige el selector que se le pida, S0 o S1.

    S0 es quedarse con la de mayor puntuacion del modelo y no necesita mirar la
    imagen; S1 es el criterio de este modulo. Tenerlos detras de la misma firma
    es lo que permite recorrer los dos selectores en un bucle en vez de
    escribir el `if` en cada sitio que los compara.
    """
    if nombre == "S0":
        return int(np.argmax(candidatas.puntuaciones))
    if nombre == "S1":
        return elegir(candidatas, gris, puntos, etiquetas)[0]
    raise KeyError(f"selector desconocido: {nombre!r}; hay {config.SELECTORES}")


def mascara_guardada(celda: dict, serie: int, etiqueta: int, frame: int):
    """La mascara que una celda de la rejilla produjo para esa lesion y frame.

    `celda` es un diccionario como `config.MEJOR_CELDA`, con modelo,
    checkpoint, modo, protocolo y selector. Reconstruye desde las candidatas
    que guardo F8, asi que no ejecuta ningun modelo.

    Volver a aplicar el selector sobre las candidatas guardadas devuelve el
    mismo indice que se eligio al medirlas, incluso en M4: alli la cadena ya
    esta fijada por el fichero que se lee, y el selector es una funcion pura de
    esas candidatas, la imagen y los clics.
    """
    from surco import prompts

    candidatas = candidatas_guardadas(
        celda["modelo"], celda["checkpoint"], serie, etiqueta,
        celda["modo"], celda["protocolo"], frame, celda["selector"],
    )
    if candidatas is None:
        return None
    unidad = next(
        u for u in prompts.cargar()["unidades"]
        if (u["serie"], u["etiqueta"]) == (serie, etiqueta)
    )
    puntos, etiquetas_clic = prompts.puntos_y_etiquetas(unidad, celda["protocolo"])
    gris = modelos.preparar_frame(serie, frame)[:, :, 0]
    indice = elegir_con(celda["selector"], candidatas, gris, puntos, etiquetas_clic)
    return candidatas.mascaras[indice]
