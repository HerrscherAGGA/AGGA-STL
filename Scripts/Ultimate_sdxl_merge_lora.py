import torch
import safetensors
import argparse
from safetensors.torch import load_file, save_file

def expand_tensor(tensor, target_shape):
    """Expande un tensor con ceros para que coincida con una nueva dimensión."""
    new_tensor = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
    slices = tuple(slice(0, min(s, ts)) for s, ts in zip(tensor.shape, target_shape))
    new_tensor[slices] = tensor
    return new_tensor

def merge_loras(model_paths, save_path, ratios, reference_model=None):
    """Fusiona múltiples LoRAs en un solo archivo, ajustando capas de distintas dimensiones."""
    merged_state_dict = {}
    
    # Cargar todos los LoRAs
    loras = [load_file(path) for path in model_paths]
    
    # Obtener todas las claves de las capas
    all_keys = set().union(*[l.keys() for l in loras])
    
    for key in all_keys:
        tensors = [l[key] for l in loras if key in l]
        
        if len(tensors) == 1:
            merged_state_dict[key] = tensors[0]
        else:
            max_shape = tuple(max(t.shape[i] for t in tensors) for i in range(len(tensors[0].shape)))
            expanded_tensors = [expand_tensor(t, max_shape) for t in tensors]
            
            merged_tensor = sum(t * r for t, r in zip(expanded_tensors, ratios))
            merged_state_dict[key] = merged_tensor

    # Guardar el archivo fusionado
    save_file(merged_state_dict, save_path)
    print(f"LoRA fusionado guardado en: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, help="Lista de rutas de LoRAs a fusionar.")
    parser.add_argument("--ratios", nargs="+", type=float, required=True, help="Pesos de cada LoRA en la fusión.")
    parser.add_argument("--save_to", required=True, help="Ruta donde se guardará el LoRA fusionado.")
    args = parser.parse_args()

    merge_loras(args.models, args.save_to, args.ratios)
