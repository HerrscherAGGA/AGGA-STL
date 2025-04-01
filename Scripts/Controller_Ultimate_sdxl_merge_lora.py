import torch
from safetensors.torch import load_file, save_file
import argparse

def expand_tensor(tensor, target_shape):
    """Expande un tensor más pequeño a una forma más grande rellenándolo con ceros."""
    expanded_tensor = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
    slices = tuple(slice(0, min(s, es)) for s, es in zip(tensor.shape, expanded_tensor.shape))
    expanded_tensor[slices] = tensor
    return expanded_tensor

def merge_loras(lora_paths, weights, method="weighted", normalize=False):
    """
    Fusiona múltiples LoRAs con distintos métodos de interpolación.
    
    - `method="average"` → Promedio simple
    - `method="weighted"` → Suma ponderada con los pesos dados
    - `method="max"` → Mantiene el valor máximo en cada peso
    - `normalize=True` → Ajusta los valores para evitar que sean demasiado grandes
    """
    
    # Cargar todos los LoRAs en memoria
    loras = [load_file(path) for path in lora_paths]
    
    # Verificar que todos los LoRAs tienen las mismas claves
    all_keys = set(loras[0].keys())
    for lora in loras[1:]:
        all_keys.update(lora.keys())
    
    merged_lora = {}

    for key in all_keys:
        # Obtener los tensores de cada LoRA para la misma clave
        tensors = [lora[key] if key in lora else torch.zeros_like(loras[0][key]) for lora in loras]
        
        # Ajustar dimensiones si es necesario
        max_shape = tuple(max(t.shape[d] for t in tensors) for d in range(len(tensors[0].shape)))
        tensors = [expand_tensor(t, max_shape) for t in tensors]

        # Aplicar el método de fusión
        if method == "average":
            merged_lora[key] = sum(tensors) / len(tensors)
        elif method == "weighted":
            merged_lora[key] = sum(w * t for w, t in zip(weights, tensors))
        elif method == "max":
            merged_lora[key] = torch.max(torch.stack(tensors), dim=0)[0]
        else:
            raise ValueError(f"Método de fusión '{method}' no soportado.")

        # Normalizar si es necesario
        if normalize:
            merged_lora[key] /= torch.norm(merged_lora[key]) + 1e-8

    return merged_lora

def main():
    parser = argparse.ArgumentParser(description="Fusionar múltiples LoRAs con distintos métodos de interpolación.")
    parser.add_argument("--models", nargs="+", required=True, help="Rutas de los LoRAs a fusionar.")
    parser.add_argument("--ratios", nargs="+", type=float, required=True, help="Pesos de fusión para cada LoRA.")
    parser.add_argument("--save_to", type=str, required=True, help="Ruta donde se guardará el LoRA fusionado.")
    parser.add_argument("--method", type=str, choices=["average", "weighted", "max"], default="weighted", help="Método de fusión.")
    parser.add_argument("--normalize", action="store_true", help="Normaliza los valores después de la fusión.")

    args = parser.parse_args()

    # Verificar que los pesos coincidan con la cantidad de LoRAs
    if len(args.models) != len(args.ratios):
        raise ValueError("El número de modelos y pesos debe coincidir.")

    # Fusionar LoRAs
    merged_lora = merge_loras(args.models, args.ratios, method=args.method, normalize=args.normalize)

    # Guardar el LoRA fusionado
    save_file(merged_lora, args.save_to)
    print(f"✅ Fusión completada. LoRA guardado en: {args.save_to}")

if __name__ == "__main__":
    main()
