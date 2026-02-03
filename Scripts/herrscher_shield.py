import modules.scripts as scripts
import gradio as gr
import torch
from modules import shared

class HerrscherShield(scripts.Script):
    def title(self):
        return "Herrscher AGGA - NaN Shield"

    def show(self, is_img2img):
        # Esto hace que el menú aparezca siempre (txt2img y img2img)
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        # Crea un acordeón en la interfaz
        with gr.Accordion("🛡️ Herrscher AGGA - NaN Shield", open=False):
            enabled = gr.Checkbox(label="Enable NaN Patching (Fix Black Screens)", value=True)
            status_info = gr.Markdown(value="*Active: Scans Layer 11 before every generation.*")
        return [enabled]

    def process(self, p, enabled):
        if not enabled:
            return

        # Accedemos al modelo cargado actualmente en la VRAM/RAM
        sd_model = shared.sd_model
        
        # Flag para saber si encontramos algo
        fixed_count = 0
        
        # Lógica para detectar el Text Encoder (Conditioner) en SDXL
        # SDXL usa 'conditioner', SD1.5 usa 'cond_stage_model'
        # Este script busca ser universal, pero se enfoca en SDXL
        if hasattr(sd_model, 'conditioner'):
            try:
                # Recorremos los embedders (CLIP G / L)
                for embedder in sd_model.conditioner.embedders:
                    if hasattr(embedder, 'wrapped'): # En Forge a veces está envuelto
                        model_part = embedder.wrapped
                    else:
                        model_part = embedder

                    # Escaneo quirúrgico
                    if hasattr(model_part, 'state_dict'):
                        for key, tensor in model_part.state_dict().items():
                            # OPTIMIZACIÓN: Solo revisamos si detectamos peligro
                            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                                # LA CURA: Reemplazo en caliente
                                tensor.copy_(torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0))
                                fixed_count += 1
            except Exception as e:
                print(f"⚠️ Herrscher Shield Warning: {e}")

        if fixed_count > 0:
            print(f"\n🛡️ HERRSCHER SHIELD: ¡PELIGRO DETECTADO! Se repararon {fixed_count} tensores corruptos en tiempo real.")
            print("   👉 La generación continuará normalmente sin pantalla negra.")