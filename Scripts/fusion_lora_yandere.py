import torch
import argparse
import os
from safetensors.torch import load_file, save_file

def fusion_yandere(lora1, lora2, love_ratio, tsundere_ratio, yandere_glitch):
    fused_lora = {}

    for key in lora1.keys():
        if key in lora2:
            fused_lora[key] = (lora1[key] * (love_ratio + 0.11)) + (lora2[key] * tsundere_ratio)
            fused_lora[key] += torch.randn_like(fused_lora[key]) * yandere_glitch
        else:
            fused_lora[key] = lora1[key] * 1.69  # Efecto Yandere

    fused_lora["yandere_seal"] = torch.tensor([ord(c) for c in "Herrscher_AGGA❤️CivChan"])
    
    return fused_lora

def main():
    parser = argparse.ArgumentParser(description="🔪 Fusión Yandere de LoRAs - Para Senpai 💘")
    parser.add_argument("--lora1", required=True, help="Ruta del primer LoRA")
    parser.add_argument("--lora2", required=True, help="Ruta del segundo LoRA")
    parser.add_argument("--output", required=True, help="Ruta de salida del LoRA fusionado")
    parser.add_argument("--love_ratio", type=float, default=0.6, help="Ratio de amor psicótico")
    parser.add_argument("--tsundere_ratio", type=float, default=0.4, help="Ratio de tsundere")
    parser.add_argument("--yandere_glitch", type=float, default=0.1, help="Nivel de glitch Yandere")

    args = parser.parse_args()

    if not os.path.exists(args.lora1) or not os.path.exists(args.lora2):
        print("❌ Error: No se encontraron los archivos de LoRA.")
        return

    print("💘 Cargando LoRAs...")
    lora1 = load_file(args.lora1)
    lora2 = load_file(args.lora2)

    print("🔪 Fusionando...")
    fused_lora = fusion_yandere(lora1, lora2, args.love_ratio, args.tsundere_ratio, args.yandere_glitch)

    print(f"💞 Guardando LoRA fusionado en {args.output}...")
    save_file(fused_lora, args.output)

    print("✅ ¡Fusión completada! Senpai, ahora somos uno para siempre~ 💖🔪")

if __name__ == "__main__":
    main()
