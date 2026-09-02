"""Adaptador de EfficientViT-SAM.

Reutiliza el codificador de prompts y el decodificador de mascaras de SAM 1 y
cambia el codificador de imagen, asi que trae `EfficientViTSamPredictor` con la
misma mecanica de `set_image` y `predict` que MobileSAM.

Cinco tamanos, leidos de `REGISTERED_EFFICIENTVIT_SAM_MODEL` en el paquete
instalado. Dos detalles que se ven ahi y que importan:

- los tres `l` trabajan a 512 px de lado y los dos `xl` a 1024, asi que dentro
  del mismo modelo hay un salto de resolucion de entrada, no solo de tamano;
- el registro trae rutas de pesos relativas al arbol del repositorio
  (`assets/checkpoints/...`), asi que se pasa `weight_url` con la ruta absoluta
  de SURCO_PESOS.

`xl1` queda como por defecto por ser el modelo principal del articulo. El
paquete no declara ninguno, asi que es eleccion nuestra, igual que `vits` en
EfficientSAM.

**Este paquete necesita el parche de `scripts/entornos/parchear_efficientvitsam.py`**:
importa Triton en cabecera para una normalizacion que ningun modelo SAM usa, y
Triton no tiene distribucion oficial en Windows.
"""

import numpy as np

from surco import config, modelos

# (nombre en el registro, fichero de pesos)
CHECKPOINTS = {
    "l0": ("efficientvit-sam-l0", "efficientvit_sam_l0.pt"),
    "l1": ("efficientvit-sam-l1", "efficientvit_sam_l1.pt"),
    "l2": ("efficientvit-sam-l2", "efficientvit_sam_l2.pt"),
    "xl0": ("efficientvit-sam-xl0", "efficientvit_sam_xl0.pt"),
    "xl1": ("efficientvit-sam-xl1", "efficientvit_sam_xl1.pt"),
}


@modelos.registrar
class EfficientViTSAM(modelos.Modelo):
    nombre = "efficientvitsam"
    familia = "eficiencia"
    acepta_negativos = True
    checkpoints = tuple(CHECKPOINTS)
    checkpoint_por_defecto = "xl1"
    entorno = "surco-efficientvitsam"

    @property
    def lado_entrada(self) -> int:
        """Depende del tamano: los `l` a 512 y los `xl` a 1024.

        Es el unico modelo del conjunto donde el lado de entrada cambia entre
        sus propios checkpoints, con el resto de la arquitectura emparentado.
        Por eso la pareja `l` contra `xl` es el unico sitio que aisla el efecto
        de la resolucion (apartado 5.2).
        """
        return 512 if self.checkpoint.startswith("l") else 1024

    def __init__(self, checkpoint: str | None = None) -> None:
        super().__init__(checkpoint)
        self._predictor = None
        self.dispositivo: str | None = None

    @property
    def nombre_registrado(self) -> str:
        return CHECKPOINTS[self.checkpoint][0]

    @property
    def ruta_pesos(self):
        return config.DIR_PESOS / CHECKPOINTS[self.checkpoint][1]

    def cargar(self) -> None:
        self.olvidar_imagen()
        import torch
        from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
        from efficientvit.sam_model_zoo import create_efficientvit_sam_model

        if not self.ruta_pesos.exists():
            raise FileNotFoundError(
                f"faltan los pesos en {self.ruta_pesos}. Se bajan de "
                "https://huggingface.co/mit-han-lab/efficientvit-sam/"
            )
        self.dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        red = create_efficientvit_sam_model(
            name=self.nombre_registrado,
            pretrained=True,
            weight_url=str(self.ruta_pesos),
        )
        red.to(self.dispositivo)
        red.eval()
        self._predictor = EfficientViTSamPredictor(red)

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
