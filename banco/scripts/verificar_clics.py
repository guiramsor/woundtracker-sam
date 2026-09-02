"""Auditoria del fichero de clics contra el protocolo del apartado 4.

Solo comprueba y reporta: **no corrige nada**. La anotacion es entrada humana y
se congela tal como se hizo.

Que se comprueba, unidad por unidad, sobre el frame `start`:

- el clic 0 cae dentro del GT;
- los clics 1 y 2 estan sobre el eje mayor de la lesion, uno a cada lado del
  centro, y a distancias parecidas de el;
- los clics 3 y 4 apuntan en la direccion perpendicular al eje, uno a cada
  lado, y fuera del surco;
- el clic 3 esta mas arriba en la imagen que el 4, que es la regla de desempate;
- los cinco clics son distintos entre si.

La posicion a lo largo del eje se da como fraccion del recorrido del surco, de
0 a 1, para poder leerla contra el "a un cuarto" y "a tres cuartos" del
protocolo: el clic 0 deberia salir cerca de 0.5, y el 1 y el 2 cerca de 0.25 y
0.75 en algun orden.

**Que los clics de fondo caigan dentro del nucleo se mide y no se comprueba**, y
la diferencia importa. El apartado 4.1 los pide en nucleoplasma, asi que como
requisito del protocolo es legitimo; lo que no se sostiene es el arbitro. El
nucleo aqui se saca con un umbral de Otsu sobre `w1CSU488` (sin parametros
libres, quedandose con la componente que contiene al clic 0) y el apartado 5.5
midio que en este canal esa deteccion **no funciona**: da componentes de 4.882 a
181.155 px2, un factor de 37 entre nucleos que deberian ser comparables, y en
las series 4, 7 y 9 mete en la misma componente las dos lesiones, que estan en
nucleos distintos. Es la misma dependencia que dejo a M2 sin implementar y que
tumbo la guarda anti-nucleo de S1.

Asi que un `False` en `c3_en_nucleo` o `c4_en_nucleo` no dice que el clic este
mal puesto: dice que Otsu no encontro ahi el nucleo. Contarlo como desviacion
del protocolo seria auditar la anotacion con una regla que el propio trabajo
declara inservible, y de paso acusar a la unica entrada humana del banco. Las
dos columnas se siguen escribiendo en el CSV, porque son justamente la evidencia
del riesgo del 5.5, pero fuera del recuento de fallos.

Se ejecuta con:

    conda run -n surco python scripts/verificar_clics.py
"""

import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu
from skimage.measure import label as etiquetar

from surco import config, datos, deriva, prompts


def mascara_del_nucleo(imagen: np.ndarray, clic: tuple[int, int]) -> np.ndarray:
    """La componente clara que contiene al clic, por umbral de Otsu."""
    claro = imagen > threshold_otsu(imagen)
    componentes = etiquetar(claro, connectivity=2)
    cual = componentes[clic]
    return componentes == cual if cual else np.zeros_like(claro)


def posiciones_en_el_eje(
    mascara: np.ndarray, eje: np.ndarray, origen: np.ndarray, puntos: list
) -> list[float]:
    """Proyeccion sobre el eje, reescalada al recorrido del surco: 0 a 1."""
    filas, columnas = np.nonzero(mascara)
    proyeccion_gt = (np.column_stack([filas, columnas]) - origen) @ eje
    inicio, fin = proyeccion_gt.min(), proyeccion_gt.max()
    return [
        float(((np.array(p) - origen) @ eje - inicio) / (fin - inicio)) for p in puntos
    ]


def _proyectar(punto, origen, eje) -> float:
    return float((np.array(punto) - origen) @ eje)


def revisar_unidad(unidad: dict) -> dict:
    """Todas las comprobaciones de una unidad, en una fila."""
    serie, etiqueta, start = unidad["serie"], unidad["etiqueta"], unidad["start"]
    clics = [tuple(c) for c in unidad["clics"]]
    mascara = datos.cargar_gt(serie, start) == etiqueta
    imagen = datos.cargar_frame(serie, start)
    nucleo = mascara_del_nucleo(imagen, clics[0])

    eje_mayor, eje_menor = deriva.ejes_pca(mascara)
    origen = deriva.centroide(mascara)
    posiciones = posiciones_en_el_eje(mascara, eje_mayor, origen, clics[:3])
    perpendiculares = [_proyectar(c, origen, eje_menor) for c in clics[3:]]
    paralelas = [_proyectar(c, origen, eje_mayor) for c in clics[3:]]

    return {
        "unidad": f"{serie}-L{etiqueta}",
        "start": start,
        "excepcion": unidad["excepcion"],
        # Sale del fichero, no se mide: lo marco quien anoto. Viaja al CSV para
        # que la desviacion se pueda leer sin abrir el JSON a mano.
        "consulta_temporal": unidad["consulta_temporal"],
        "c0_dentro_gt": bool(mascara[clics[0]]),
        "pos_c0": posiciones[0],
        "pos_c1": posiciones[1],
        "pos_c2": posiciones[2],
        "c1_c2_lados_opuestos": (posiciones[1] - 0.5) * (posiciones[2] - 0.5) < 0,
        "asimetria_c1_c2": abs(abs(posiciones[1] - 0.5) - abs(posiciones[2] - 0.5)),
        "c1_dentro_gt": bool(mascara[clics[1]]),
        "c2_dentro_gt": bool(mascara[clics[2]]),
        "c3_perp_domina": abs(perpendiculares[0]) > abs(paralelas[0]),
        "c4_perp_domina": abs(perpendiculares[1]) > abs(paralelas[1]),
        "c3_c4_lados_opuestos": perpendiculares[0] * perpendiculares[1] < 0,
        "c3_fuera_gt": not mascara[clics[3]],
        "c4_fuera_gt": not mascara[clics[4]],
        "c3_en_nucleo": bool(nucleo[clics[3]]),
        "c4_en_nucleo": bool(nucleo[clics[4]]),
        "regla_desempate": clics[3][0] < clics[4][0],
        "cinco_distintos": len(set(clics)) == config.N_CLICS,
        "perp_c3_px": perpendiculares[0],
        "perp_c4_px": perpendiculares[1],
        "area_nucleo_px": int(nucleo.sum()),
        # Un clic de fondo en nucleoplasma tiene que quedar claramente por
        # encima del fondo extranuclear. Es un contraste, no un umbral.
        "int_c3": int(imagen[clics[3]]),
        "int_c4": int(imagen[clics[4]]),
        "int_mediana_imagen": int(np.median(imagen)),
        "int_mediana_surco": int(np.median(imagen[mascara])),
    }


# Lo que de verdad audita la anotacion: todo se decide contra el GT, contra la
# geometria de la propia lesion o contra los clics entre si, que son cosas que
# estan medidas y no estimadas.
COMPROBACIONES = [
    "c0_dentro_gt",
    "c1_c2_lados_opuestos",
    "c1_dentro_gt",
    "c2_dentro_gt",
    "c3_perp_domina",
    "c4_perp_domina",
    "c3_c4_lados_opuestos",
    "c3_fuera_gt",
    "c4_fuera_gt",
    "regla_desempate",
    "cinco_distintos",
]

# Se miden y se reportan, pero no cuentan como fallo: su arbitro es la deteccion
# de nucleo por Otsu, que el apartado 5.5 midio que no funciona en w1CSU488. Un
# False aqui es un fallo del detector, no de quien puso el clic.
DIAGNOSTICOS = ["c3_en_nucleo", "c4_en_nucleo"]


def _diagnostico_del_nucleo(tabla: pd.DataFrame) -> None:
    """Lo que mide el detector de nucleo, que no es la calidad de la anotacion.

    Va aparte y con este nombre a proposito. Estas dos columnas son la evidencia
    del riesgo del apartado 5.5 (que en `w1CSU488` el nucleo no se puede
    delimitar) y no un examen de donde puso los clics el autor.
    """
    print("\n=== EL NUCLEO POR OTSU: DIAGNOSTICO, NO COMPROBACION ===")
    print("un False dice que Otsu no encontro nucleo ahi, no que el clic este mal")
    columnas = ["unidad", *DIAGNOSTICOS, "int_c3", "int_c4", "int_mediana_imagen",
                "int_mediana_surco", "area_nucleo_px"]
    print(tabla[columnas].to_string(index=False))

    fuera = int((~tabla[DIAGNOSTICOS]).to_numpy().sum())
    menor, mayor = int(tabla["area_nucleo_px"].min()), int(tabla["area_nucleo_px"].max())
    print(f"\n  clics de fondo que Otsu deja fuera del nucleo: {fuera} de "
          f"{2 * len(tabla)}, en {int((~tabla[DIAGNOSTICOS]).any(axis=1).sum())} unidades")
    print(f"  la componente va de {menor} a {mayor} px2, un factor de "
          f"{mayor / menor:.0f} entre nucleos que deberian ser comparables")

    # Dos unidades de la misma serie se miden sobre el mismo frame y con el mismo
    # umbral, asi que compartir area es compartir componente: Otsu no separa esos
    # dos nucleos.
    juntas = [
        serie
        for serie, grupo in tabla.groupby(tabla["unidad"].str.split("-").str[0])
        if len(grupo) > 1 and grupo["area_nucleo_px"].nunique() < len(grupo)
    ]
    print(f"  series donde las dos lesiones caen en la misma componente, estando "
          f"en nucleos distintos: {', '.join(sorted(juntas, key=int))}")
    print("  Es la dependencia que dejo a M2 sin implementar y tumbo la guarda")
    print("  anti-nucleo de S1 (apartados 5.1 y 5.5).")


def main() -> None:
    datos_clics = prompts.cargar()
    tabla = pd.DataFrame([revisar_unidad(u) for u in datos_clics["unidades"]])
    tabla["fallos"] = (~tabla[COMPROBACIONES]).sum(axis=1)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)

    print(f"unidades en el fichero: {len(tabla)}\n")
    print("=== COMPROBACIONES (False = se desvia) ===")
    print(tabla[["unidad", *COMPROBACIONES, "fallos"]].to_string(index=False))

    print("\n=== POSICION SOBRE EL EJE, como fraccion del recorrido ===")
    print("referencia del protocolo: c0 en 0.50, c1 y c2 en 0.25 y 0.75")
    continuas = ["unidad", "pos_c0", "pos_c1", "pos_c2", "asimetria_c1_c2",
                 "perp_c3_px", "perp_c4_px"]
    print(tabla[continuas].to_string(index=False, float_format=lambda v: f"{v:7.3f}"))

    desviadas = tabla[tabla["fallos"] > 0]
    print(f"\n=== UNIDADES QUE SE DESVIAN: {len(desviadas)} de {len(tabla)} ===")
    for _, fila in desviadas.iterrows():
        fallan = [c for c in COMPROBACIONES if not fila[c]]
        print(f"  {fila['unidad']:<6} {', '.join(fallan)}")
    if desviadas.empty:
        print("  ninguna: los cinco clics de las 17 siguen el protocolo del apartado 4")

    print("\n=== LO QUE DECLARO QUIEN ANOTO ===")
    for columna, que in (
        ("excepcion", "hubo que deslizar el clic 1 o el 2 buscando senal (clausula 4.2)"),
        ("consulta_temporal", "hubo que mirar otros frames para localizar la lesion"),
    ):
        cuales = tabla[tabla[columna]]["unidad"].tolist()
        print(f"  {columna:<18} {', '.join(cuales) if cuales else 'ninguna':<14} {que}")

    _diagnostico_del_nucleo(tabla)

    tabla.to_csv(config.DIR_PROCESO / "verificacion_clics.csv", index=False)
    print(f"\ntabla completa en {config.DIR_PROCESO / 'verificacion_clics.csv'}")


if __name__ == "__main__":
    main()
