"""Corrección de registro: normaliza el orden de los dos clics de fondo.

**Ya aplicado, y por eso vive en `scripts/unicos/`.** Su resultado esta congelado
en `anotaciones/clics.json`, que es la unica entrada humana del trabajo y esta
versionada. No forma parte del pipeline de reproduccion: se conserva porque
**documenta por que los indices de ese fichero estan como estan**, que es una
pregunta que se le puede hacer al dato de entrada y que sin esto no tendria
respuesta. Lo que hizo lo comprueba `scripts/verificar_clics.py`.

**No es reanotar.** Los puntos estan donde el anotador los puso; lo que estaba
mal era su orden en el fichero. El protocolo del apartado 4.1 fija que el clic
4 va en el lado que queda mas arriba en la imagen y el 5 en el opuesto, asi que
la fila del indice 3 tiene que ser menor que la del indice 4.

Importa mas de lo que parece. El protocolo P2 es `(0, 3)`: usa un unico
negativo, el del indice 3. Con el orden invertido, esas unidades meterian en P2
el negativo del lado contrario que el resto, y P2 dejaria de comparar lo mismo
en todas.

Es idempotente: solo intercambia donde la regla se incumple, asi que ejecutarlo
dos veces no deshace la correccion.

    conda run -n surco python scripts/unicos/corregir_orden_fondo.py
"""

from surco import prompts

INDICE_ARRIBA = 3
INDICE_ABAJO = 4


def desordenada(unidad: dict) -> bool:
    """Si el clic de fondo del indice 3 queda mas abajo que el del indice 4."""
    fila_arriba = unidad["clics"][INDICE_ARRIBA][0]
    fila_abajo = unidad["clics"][INDICE_ABAJO][0]
    return fila_arriba > fila_abajo


def intercambiar(unidad: dict) -> dict:
    corregida = dict(unidad)
    clics = [list(c) for c in unidad["clics"]]
    clics[INDICE_ARRIBA], clics[INDICE_ABAJO] = (
        clics[INDICE_ABAJO],
        clics[INDICE_ARRIBA],
    )
    corregida["clics"] = clics
    return corregida


def main() -> None:
    datos = prompts.cargar()
    tocadas = []
    unidades = []
    for unidad in datos["unidades"]:
        if desordenada(unidad):
            tocadas.append(f"{unidad['serie']}-L{unidad['etiqueta']}")
            unidades.append(intercambiar(unidad))
        else:
            unidades.append(unidad)

    if not tocadas:
        print("nada que corregir: la regla de desempate se cumple en todas")
        return

    datos["unidades"] = unidades
    prompts.guardar(datos)
    print(f"orden de fondo intercambiado en {len(tocadas)} unidades: {', '.join(tocadas)}")
    print("los puntos son los mismos; solo cambia cual es el indice 3 y cual el 4")


if __name__ == "__main__":
    main()
