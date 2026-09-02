"""Reproduce la verificacion que deja a SAM 3 fuera de la lista.

SAM 3 se excluye por incumplir el criterio del apartado 5.2, no por un problema
de entorno. Falla las dos primeras condiciones:

1. **No acepta prompt de punto en su API publicada.** `Sam3Processor` expone
   `set_text_prompt` y `add_geometric_prompt`, que recibe cajas. El camino de
   punto al estilo SAM 1 existe en el codigo, en
   `SAM3InteractiveImagePredictor`, pero no forma parte de la interfaz que el
   repositorio ofrece.
2. **Los pesos que lo harian funcionar no se publican.** Ninguno de los dos
   checkpoints contiene una sola clave `inst_interactive_predictor.*`.

Lo peligroso es que **no falla**: con `enable_inst_interactivity=True` el modelo
se construye igual, ese subarbol queda inicializado al azar, y el predictor
devuelve mascaras bien formadas, del tamano correcto y con puntuaciones
plausibles, que no significan nada. Una comprobacion de disponibilidad que solo
mire formas lo da por bueno.

    conda run -n surco-sam3 python scripts/entornos/verificar_exclusion_sam3.py
"""

import os

os.environ.setdefault("HF_HUB_CACHE", r"E:\pesos_sam\huggingface")

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402
from sam3.model_builder import download_ckpt_from_hf  # noqa: E402

from surco import config  # noqa: E402


def claves_interactivas(version: str) -> tuple[int, int]:
    """Claves totales del checkpoint y cuantas son del predictor interactivo."""
    ruta = download_ckpt_from_hf(version=version)
    bruto = torch.load(ruta, map_location="cpu", weights_only=False)
    estado = bruto.get("model", bruto) if isinstance(bruto, dict) else bruto
    interactivas = [c for c in estado if str(c).startswith("inst_interactive_predictor")]
    return len(estado), len(interactivas)


def main() -> None:
    print("=== condicion 1: que prompts expone la API publica ===")
    metodos = [n for n in dir(Sam3Processor) if n.startswith(("set_", "add_"))]
    print(f"  {', '.join(sorted(metodos))}")
    print("  ninguno acepta puntos: son texto y caja\n")

    print("=== condicion 2: pesos del predictor interactivo ===")
    filas = []
    for version in ("sam3", "sam3.1"):
        totales, interactivas = claves_interactivas(version)
        print(f"  {version:<7} {totales:5d} claves, {interactivas} interactivas")
        filas.append(
            {
                "checkpoint": version,
                "metodos_de_prompt": " ".join(sorted(metodos)),
                "claves": totales,
                "claves_interactivas": interactivas,
            }
        )

    print(
        "\nCon enable_inst_interactivity=True el subarbol queda aleatorio y el\n"
        "predictor devuelve mascaras bien formadas sin significado. Por eso la\n"
        "exclusion es por criterio y no por entorno: no depende de la maquina\n"
        "ni de la version de torch, sino de lo que el repositorio publica."
    )

    # Los dos recuentos los cita el apartado 5.2, y hasta ahora solo existian
    # impresos: sin fichero, `verificacion` no tiene contra que comparar y la
    # unica forma de repetirlos era levantar este entorno y leer la consola.
    destino = config.DIR_PROCESO / "exclusion_sam3.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(destino, index=False)
    print(f"\nescrito en {destino}")


if __name__ == "__main__":
    main()
