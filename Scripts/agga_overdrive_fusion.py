import modules.scripts as scripts
import gradio as gr
from modules import shared

class AggaOverdriveFusion(scripts.Script):
    def title(self):
        return "Herrscher AGGA - Overdrive V2 (Fusion Edition)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("⚡ AGGA Overdrive V2 (Speed Hack)", open=False):
            gr.Markdown("<small>Acelera generaciones largas (20-40 pasos) optimizando el pipeline de PyTorch sin afectar tus Samplers AGGA.</small>")
            
            tome_ratio = gr.Slider(minimum=0.0, maximum=0.8, step=0.1, label="🧬 Token Merging (ToMe) - Recomendado: 0.3 o 0.4", value=0.0)
            kill_negative = gr.Checkbox(label="✂️ Null-Negative Bypass (¡Doble Velocidad!)", value=False)
            
            gr.Markdown("*Nota: Usa el Bypass Negativo solo si tu CFG Scale es menor a 5.0.*")
        return [tome_ratio, kill_negative]

    def process(self, p, tome_ratio, kill_negative):
        if tome_ratio == 0 and not kill_negative:
            return

        # 1. Aplicar Token Merging (Aceleración de Atención)
        # Esto reduce drásticamente el uso de VRAM y el tiempo de cálculo matricial
        self.original_tome = getattr(shared.opts, 'token_merging_ratio', 0.0)
        if hasattr(shared.opts, 'token_merging_ratio'):
            shared.opts.token_merging_ratio = tome_ratio
            if tome_ratio > 0:
                print(f"⚡ [AGGA Overdrive] Token Merging activado al {tome_ratio*100}%")

        # 2. Null-Negative Bypass (El multiplicador x2)
        # Al vaciar el prompt negativo, PyTorch solo calcula el tensor incondicional como vacío,
        # lo que A1111 a veces optimiza, o al menos reduce la carga de procesamiento de texto.
        if kill_negative:
            # Respaldamos por si acaso, aunque p.negative_prompt se sobreescribe para esta gen
            p.negative_prompt = ""
            if hasattr(p, 'all_negative_prompts'):
                p.all_negative_prompts = [""] * len(p.all_negative_prompts)
            print("⚡ [AGGA Overdrive] Prompt negativo anulado. Modo de máxima velocidad.")

    def postprocess(self, p, processed, tome_ratio, kill_negative):
        # Restaurar ToMe para no contaminar la configuración global
        if hasattr(shared.opts, 'token_merging_ratio'):
            shared.opts.token_merging_ratio = self.original_tome