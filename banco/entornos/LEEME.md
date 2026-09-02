# Los entornos, uno por modelo

Un entorno conda por modelo, porque sus versiones de torch y de sus repositorios **no conviven
entre sí**. Construirlos y comprobar que todos cargan es F5 y F6 (`notebooks/GUIA.ipynb`).

    conda env create -f entornos/<modelo>.yml
    conda run -n surco-<modelo> python scripts/comprobar_carga.py

Los pesos van fuera del repositorio, en `SURCO_PESOS` (por defecto `E:\pesos_sam`), y los
repositorios de terceros que no se instalan por pip, en `SURCO_REPOS`.

## Dos recetas son de modelos que quedaron fuera

`sam3.yml` y `fastsam.yml` **no** forman parte de la comparativa: los dos modelos quedaron
excluidos por incumplir el criterio de inclusión del apartado 5.2. Sus recetas se conservan a
propósito, porque **sin ellas la exclusión no sería reproducible**: son lo que hace falta para
volver a levantar el entorno y repetir la comprobación.

    conda run -n surco-sam3    python scripts/entornos/verificar_exclusion_sam3.py
    conda run -n surco-fastsam python scripts/entornos/verificar_exclusion_fastsam.py

## Tres entornos llevan parche

`surco-edgesam`, `surco-efficientvitsam` y `surco-sam3` necesitan un parche a un paquete de
terceros antes de poder usarse. Son scripts versionados e idempotentes, y se aplican antes de
nada:

    conda run -n surco-edgesam python scripts/entornos/parchear_edgesam.py
    conda run -n surco-edgesam python scripts/entornos/verificar_edgesam.py

Los dos primeros llevan verificación al lado y el tercero no. **Qué parchea cada uno, por qué
no se puede instalar la dependencia y por qué ninguno toca nada evaluable, en la cabecera de
cada script.**

## Un checkpoint no se ejecuta aquí

TinySAM `w8a8` carga pero falla al inferir con torch 2.11 (apartado 5.2). No está excluido del
trabajo: está declarado **no ejecutable en el entorno de referencia**.
