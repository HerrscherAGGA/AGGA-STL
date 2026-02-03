import modules.scripts as scripts
import gradio as gr
import torch
from modules import shared

class HerrscherShield(scripts.Script):
    last_fixed_checkpoint = None

    def title(self):
        return "Herrscher AGGA - NaN Shield"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("🛡️ Herrscher AGGA - NaN Shield", open=False):
            enabled = gr.Checkbox(label="Active Layer 11 Repair", value=True)
            status = gr.Markdown(value="*Status: Ready to scan.*")
        return [enabled]

    def process(self, p, enabled):
        if not enabled: return

        # Detección inteligente de modelo
        sd_model = shared.sd_model
        current_checkpoint = getattr(shared.opts, 'sd_model_checkpoint', "Unknown")

        # Smart Cache: Si ya arreglamos este modelo, no hacemos nada.
        if self.last_fixed_checkpoint == current_checkpoint:
            return

        print(f"🛡️ SHIELD: Analizando integridad de {current_checkpoint}...")
        fixed_count = 0

        # Cirugía (Solo lectura/escritura en tensores, sin overhead de memoria)
        with torch.no_grad():
            if hasattr(sd_model, 'conditioner'):
                try:
                    for embedder in sd_model.conditioner.embedders:
                        wrapper = getattr(embedder, 'wrapped', embedder)
                        if hasattr(wrapper, 'state_dict'):
                            for tensor in wrapper.state_dict().values():
                                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                                    tensor.copy_(torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0))
                                    fixed_count += 1
                except Exception as e:
                    print(f"⚠️ Shield Error: {e}")

        if fixed_count > 0:
            print(f"✅ SHIELD: {fixed_count} tensores reparados. Pantalla negra evitada.")
        else:
            print(f"✨ SHIELD: Modelo sano.")

        self.last_fixed_checkpoint = current_checkpoint
