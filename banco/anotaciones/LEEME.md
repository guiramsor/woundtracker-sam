# Anotaciones del estudio de concordancia (apartado 7)

**Datos primarios irrecuperables.** Lo que devuelvan las anotadoras no se puede regenerar:
si se pierde, hay que volver a pedirlo. Por eso esta carpeta se versiona, igual que el
fichero de clics de F3, y no vive en `salidas/`, que es todo derivado y desechable.

## Qué va aquí

Un `.zip` del ROI Manager de Fiji por cada imagen anotada, con el mismo nombre que la
imagen. Para la imagen `conc_s06_t10.tif`, el fichero es `conc_s06_t10.zip`.

```
anotaciones/
  anotadora_1/conc_s01_t04.zip
  anotadora_2/conc_s01_t04.zip
  ...
```

Son 21 ficheros por anotadora, no 34: las unidades de una misma serie comparten frame, y
en esos casos el `.zip` lleva varias ROIs, una por lesión. La correspondencia entre
fichero y pares `(unidad, frame)` está en `salidas/proceso/concordancia/correspondencia.csv`, que
sí se puede regenerar con `scripts/concordancia_seleccion.py`.

Las carpetas van numeradas y no con nombres: en la memoria las anotadoras se describen por
cualificación y por el procedimiento, no por su relación con el autor.

## Regla de "no identificable"

**Si en una imagen no se consigue identificar ningún surco, no se dibuja nada y se anota
como "no identificable".** Es una respuesta válida del protocolo, no un fallo ni un hueco.

Distingue las dos cosas que de otro modo se confundirían: no ver la lesión, y verla y
trazarla distinto. Sin esta regla, una anotadora insegura dibuja algo por no dejarlo en
blanco, y ese trazo entra en el Dice como si fuera una discrepancia de precisión cuando en
realidad es una de detección.

Importa especialmente en las imágenes tenues. La más débil de las 21 es
`conc_s06_t10.tif`, la unidad 6-L1, con el surco a 13 niveles de gris por encima del
fondo; su estrato tardío no mejora, se queda en 18. Va a las anotadoras tal cual, porque
quitarla inflaría el techo humano al retirar justo el caso difícil.

## Cómo se reporta

**Por estrato, temprano y tardío por separado, nunca agregado en un solo número.**

En los frames tardíos la fragmentación del surco es máxima, así que un desacuerdo alto ahí
puede venir de ambigüedad del protocolo (dónde acaba un surco roto en varios trozos) y no
de falta de precisión de las anotadoras. Promediarlo con el estrato temprano deflactaría
el techo humano entero, y el techo humano es justamente lo que da sentido a un Dice de
0.53. Que humanos y modelo fallen en los mismos frames es un resultado, pero solo se ve si
los estratos se miran por separado.

El estadístico es el Dice por pares entre anotadores, promediado. Nada de kappa.

## Lo que no va aquí

Máscaras. No hace falta convertir las ROIs a máscara para archivarlas: el `.zip` de Fiji
es el dato primario y la conversión es derivada. Y el GT del autor no se toca ni se
mezcla con esto: estas anotaciones sirven solo para estimar el techo humano.
