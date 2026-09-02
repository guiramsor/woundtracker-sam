# Cómo leer `resumen.csv`

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
