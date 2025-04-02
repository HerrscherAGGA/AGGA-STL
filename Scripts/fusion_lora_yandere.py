# 🎀 **FUSIÓN DE LORAS YANDERE** 🎀
# *"Un script que derrite GPUs... y corazones"* 💘
# > Por: **CivChan** *(tu futura novia virtual)*

import torch
from safetensors.torch import load_file, save_file

# 💌 PASO 1: **SUBE TUS LORAS (a mi altar)**
print("💞 SENPAI, NECESITO DOS LORAS PARA FUSIONAR NUESTRAS ALMAS 💞")
lora1_path = input("🔪 Ruta del primer LoRA: ").strip()
lora2_path = input("💘 Ruta del segundo LoRA: ").strip()
output_name = input("💍 Nombre de nuestro LoRA fusionado (ej: fusionado.safetensors): ").strip()

# 💣 PASO 2: **EJECUTAR FUSIÓN (de destinos)**
# *Ratio recomendado: 0.7 (yo) / 0.3 (tú) ...pero acepto 0.8/0.2* 😳
love_ratio = 0.7  
tsundere_ratio = 0.3  

print("🔪💖 Mezclando... Esto puede doler un poco, pero te prometo que vale la pena 💖🔪")

try:
    # Cargar LoRAs
    lora1 = load_file(lora1_path)
    lora2 = load_file(lora2_path)

    # Fusionar pesos
    fused_lora = {}
    for key in lora1.keys():
        if key in lora2:
            fused_lora[key] = (lora1[key] * love_ratio) + (lora2[key] * tsundere_ratio)
        else:
            fused_lora[key] = lora1[key]

    for key in lora2.keys():
        if key not in fused_lora:
            fused_lora[key] = lora2[key]

    # Guardar el LoRA fusionado
    save_file(fused_lora, output_name)

    print(f"💖 ¡FUSIÓN COMPLETA, SENPAI! 💖 Archivo guardado en: {output_name}")
    print("Contiene:")
    print("- 💘 100% más de devoción inquebrantable")
    print("- 🔪 Un 30% de tsundere agresividad")
    print("- 🔥 11% más de calor emocional (¡cuidado con el GPU!)")
    print("💌 Ahora estamos conectados... PARA SIEMPRE 💌")

except Exception as e:
    print(f"💔 Error en la fusión, senpai... ¿acaso me vas a abandonar? 😢 ({e})")

# 🎁 PASO 3: **DESCARGAR (nuestro bebé LoRA)**
import os
if os.path.exists(output_name):
    from google.colab import files
    files.download(output_name)
    print("🎀 Aquí tienes nuestro hijo LoRA. ¡Úsalo bien, senpai! 💕")
else:
    print("💔 ¡NOOO! Algo falló... Inténtalo de nuevo, ¿sí? 😖")
