"""Adaptador de EfficientSAM.

Se aparta del molde de MobileSAM en tres cosas, y las tres estan leidas del
paquete instalado:

1. **No hay `SamPredictor`.** Se llama al modelo directamente, y hace codificador
   y decodificador de una vez, asi que no existe el `set_image` que en los demas
   permite reutilizar el embedding entre prompts.
2. **Los constructores traen la ruta de pesos cableada y relativa**
   (`weights/efficient_sam_vitt.pt`), asi que solo funcionan si el directorio de
   trabajo es el del repositorio. Se usa el constructor general
   `build_efficient_sam`, que si acepta una ruta absoluta, con los dos
   parametros de arquitectura que cada tamano fija.
3. **La salida viene sin ordenar** y con una dimension de consulta de mas:
   logits `[B, consultas, mascaras, alto, ancho]` y IoU `[B, consultas,
   mascaras]`. Las candidatas salen de ahi tal cual, sin ordenar, porque
   ordenarlas es cosa del selector S0 y no del adaptador.

`vits` no es el checkpoint "por defecto" segun el codigo, que no declara
ninguno: es el modelo principal del articulo, EfficientSAM-S. Queda declarado
como eleccion nuestra, igual que `large` en SAM 2.1.
"""

import numpy as np

from surco import config, modelos

# (dimension del patch embed, cabezas del codificador, fichero de pesos), tal
# como los fijan build_efficient_sam_vitt y build_efficient_sam_vits.
CHECKPOINTS = {
    "vitt": (192, 3, "efficient_sam_vitt.pt"),
    "vits": (384, 6, "efficient_sam_vits.pt"),
}


@modelos.registrar
class EfficientSAM(modelos.Modelo):
    nombre = "efficientsam"
    familia = "eficiencia"
    acepta_negativos = True
    checkpoints = tuple(CHECKPOINTS)
    checkpoint_por_defecto = "vits"
    entorno = "surco-efficientsam"
    lado_entrada = 1024

    def __init__(self, checkpoint: str | None = None) -> None:
        super().__init__(checkpoint)
        self._red = None
        self.dispositivo: str | None = None

    @property
    def ruta_pesos(self):
        return config.DIR_PESOS / CHECKPOINTS[self.checkpoint][2]

    def cargar(self) -> None:
        import torch
        from efficient_sam.build_efficient_sam import build_efficient_sam

        if not self.ruta_pesos.exists():
            raise FileNotFoundError(
                f"faltan los pesos en {self.ruta_pesos}. Estan en el arbol del "
                "repositorio yformer/EfficientSAM, bajo weights/, y el de vits "
                "viene comprimido"
            )
        dimension, cabezas, _ = CHECKPOINTS[self.checkpoint]
        self.dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        self._red = build_efficient_sam(
            encoder_patch_embed_dim=dimension,
            encoder_num_heads=cabezas,
            checkpoint=str(self.ruta_pesos),
        )
        self._red.to(self.dispositivo)
        self._red.eval()

    def segmentar(
        self,
        imagen_rgb: np.ndarray,
        puntos: list[tuple[int, int]],
        etiquetas: list[int],
    ) -> modelos.Candidatas:
        import torch

        if self._red is None:
            raise RuntimeError("hay que llamar a cargar() antes de segmentar")

        # El modelo espera [B, 3, alto, ancho] en coma flotante de 0 a 1; la
        # normalizacion por media y desviacion la lleva dentro.
        tensor = (
            torch.from_numpy(imagen_rgb).permute(2, 0, 1).float().div(255).unsqueeze(0)
        )
        # Aqui todo va en (fila, columna); espera (x, y) = (columna, fila).
        coordenadas = torch.tensor(
            [[[[col, fila] for fila, col in puntos]]], dtype=torch.float
        )
        marcas = torch.tensor([[list(etiquetas)]], dtype=torch.float)

        with torch.no_grad():
            logits, iou = self._red(
                tensor.to(self.dispositivo),
                coordenadas.to(self.dispositivo),
                marcas.to(self.dispositivo),
            )
        # [B, consultas, mascaras, alto, ancho] -> [mascaras, alto, ancho]
        mascaras = (logits[0, 0] >= 0).cpu().numpy()
        return modelos.Candidatas(
            np.asarray(mascaras, dtype=bool),
            np.asarray(iou[0, 0].cpu().numpy(), dtype=float),
        )
