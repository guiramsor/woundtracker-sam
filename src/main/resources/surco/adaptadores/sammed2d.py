"""Adaptador de SAM-Med2D.

Como TinySAM, **no se instala**: su repositorio no trae setup.py ni
pyproject.toml ni requirements.txt, asi que se clona a `SURCO_REPOS` y este
adaptador mete esa ruta en `sys.path` dentro de `cargar()`.

Cuatro cosas leidas del repositorio que lo separan del molde:

1. **Su lado de entrada es 256**, el mas bajo del conjunto con diferencia. Con
   un surco de unos 17 px de ancho sobre 1024, a 256 queda en unos 4 px. Eso no
   es un detalle de configuracion: es casi una prediccion sobre su resultado, y
   por eso la columna de lado de entrada del apartado 5.2 existe.
2. El constructor recibe un **objeto de argumentos** con `image_size`,
   `encoder_adapter` y `sam_checkpoint`, no parametros sueltos.
3. El predictor es `SammedPredictor`, de `predictor_sammed`, y no el
   `SamPredictor` generico que trae el mismo arbol.
4. Publica **dos ficheros de pesos para el mismo codigo**, que se distinguen por
   `encoder_adapter`: `sam_med2d` lo activa y es el modelo que da nombre al
   trabajo; `ft_sam` lo desactiva y es un SAM afinado solo en el decodificador.
   Se tratan como dos checkpoints de la misma entrada, igual que `vit_t` y
   `w8a8` en TinySAM.

Cuidado con el nombre del paquete: dentro del clon, el modulo se llama
`segment_anything`, igual que el paquete oficial de SAM 1. Con un entorno por
modelo no hay choque, pero si algun dia conviven habria que resolverlo.
"""

import contextlib
import sys
from argparse import Namespace

import numpy as np

from surco import config, modelos


@contextlib.contextmanager
def _torch_load_permisivo(torch):
    """Relaja `weights_only` mientras corre el constructor del repositorio.

    Sus checkpoints son de entrenamiento: llevan el estado del optimizador
    ademas de los pesos, y contienen un `dict` que `weights_only=True` se niega
    a desencurtir. torch lo puso a True por defecto despues de que ese codigo se
    escribiera, asi que su `torch.load(f)` ya no funciona.

    Se relaja **solo durante la construccion** y se restaura despues, en vez de
    parchear su fichero: asi su logica de carga corre exactamente como esta
    escrita, incluida la rama que interpola los embeddings de posicion, y no hay
    codigo ajeno modificado que mantener.

    Desencurtir con `weights_only=False` ejecuta pickle arbitrario. Es la misma
    confianza que ya se le da al fichero al bajarlo de los enlaces oficiales del
    repositorio, la misma que con el `w8a8` de TinySAM, pero conviene que se vea
    escrito y acotado a este bloque.
    """
    original = torch.load

    def permisivo(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = permisivo
    try:
        yield
    finally:
        torch.load = original

LADO_ENTRADA = 256

# (fichero de pesos, adaptador del codificador activo)
CHECKPOINTS = {
    "sam_med2d": ("sam-med2d_b.pth", True),
    "ft_sam": ("ft-sam_b.pth", False),
}


@modelos.registrar
class SAMMed2D(modelos.ModeloConMascara):
    nombre = "sammed2d"
    familia = "radiologia"
    acepta_negativos = True
    checkpoints = tuple(CHECKPOINTS)
    checkpoint_por_defecto = "sam_med2d"
    entorno = "surco-sammed2d"
    lado_entrada = LADO_ENTRADA

    def __init__(self, checkpoint: str | None = None) -> None:
        super().__init__(checkpoint)
        self._predictor = None
        self.dispositivo: str | None = None

    @property
    def ruta_repositorio(self):
        return config.DIR_REPOS / "SAM-Med2D"

    @property
    def ruta_pesos(self):
        return config.DIR_PESOS / CHECKPOINTS[self.checkpoint][0]

    def _preparar_ruta(self) -> None:
        if not self.ruta_repositorio.is_dir():
            raise FileNotFoundError(
                f"falta el clon en {self.ruta_repositorio}. Se obtiene con "
                "git clone --depth 1 https://github.com/OpenGVLab/SAM-Med2D.git"
            )
        ruta = str(self.ruta_repositorio)
        if ruta not in sys.path:
            sys.path.insert(0, ruta)

    def cargar(self) -> None:
        self.olvidar_imagen()
        import torch

        self._preparar_ruta()
        from segment_anything import sam_model_registry
        from segment_anything.predictor_sammed import SammedPredictor

        if not self.ruta_pesos.exists():
            raise FileNotFoundError(
                f"faltan los pesos en {self.ruta_pesos}. Estan en los enlaces de "
                "Google Drive del README de OpenGVLab/SAM-Med2D"
            )
        self.dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        argumentos = Namespace(
            image_size=LADO_ENTRADA,
            encoder_adapter=CHECKPOINTS[self.checkpoint][1],
            sam_checkpoint=str(self.ruta_pesos),
        )
        with _torch_load_permisivo(torch):
            red = sam_model_registry["vit_b"](argumentos)
        red.to(self.dispositivo)
        red.eval()
        self._predictor = SammedPredictor(red)

    def segmentar(
        self,
        imagen_rgb: np.ndarray,
        puntos: list[tuple[int, int]],
        etiquetas: list[int],
    ) -> modelos.Candidatas:
        if self._predictor is None:
            raise RuntimeError("hay que llamar a cargar() antes de segmentar")

        if self.hay_que_codificar(imagen_rgb):
            self._predictor.set_image(imagen_rgb)
        # Aqui todo va en (fila, columna); SAM espera (x, y) = (columna, fila).
        coordenadas = np.array([[col, fila] for fila, col in puntos], dtype=float)
        mascaras, puntuaciones, _ = self._predictor.predict(
            point_coords=coordenadas,
            point_labels=np.array(etiquetas, dtype=int),
            multimask_output=True,
        )
        return modelos.Candidatas(
            np.asarray(mascaras, dtype=bool), np.asarray(puntuaciones, dtype=float)
        )

    def segmentar_con_mascara(
        self,
        imagen_rgb: np.ndarray,
        puntos: list[tuple[int, int]],
        etiquetas: list[int],
        mascara_previa: np.ndarray | None = None,
    ) -> tuple[modelos.Candidatas, np.ndarray]:
        # Sus logits son de 64 x 64, no de 256: es el lado de entrada de 256 el
        # que arrastra tambien la resolucion de la mascara que se reencadena.
        return modelos.encadenar_por_predict(
            self, self._predictor, imagen_rgb, puntos, etiquetas, mascara_previa
        )
