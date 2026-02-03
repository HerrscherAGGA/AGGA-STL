import modules.scripts as scripts
import gradio as gr
import torch
import gc
from modules import shared

class HerrscherShield(scripts.Script):
    # Variable de clase para recordar el último modelo arreglado
    last_fixed_checkpoint = None

    def title(self):
        return "Herrscher AGGA - Smart Shield"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("🛡️ Herrscher AGGA - Smart Shield", open=False):
            enabled = gr.Checkbox(label="Enable Smart NaN Patching", value=True)
            force_clean = gr.Checkbox(label="Force RAM Cleanup after fix", value=True)
            status_info = gr.Markdown(value="*Status: Idle. Waiting for generation...*")
        return [enabled, force_clean]

    def process(self, p, enabled, force_clean):
        if not enabled:
            return

        # 1. IDENTIFICACIÓN: ¿Qué modelo estamos usando?
        sd_model = shared.sd_model
        current_checkpoint_name = "Unknown"
        
        # Intentamos obtener el nombre del checkpoint actual
        if hasattr(shared.opts, 'sd_model_checkpoint'):
            current_checkpoint_name = shared.opts.sd_model_checkpoint
        
        # 2. CACHE INTELIGENTE: ¿Ya arreglamos este modelo antes?
        if self.last_fixed_checkpoint == current_checkpoint_name:
            # ¡Ya está limpio! No gastamos RAM escaneando de nuevo.
            return

        print(f"\n🛡️ HERRSCHER SMART SHIELD: Nuevo modelo detectado ({current_checkpoint_name}). Iniciando escaneo...")
        
        fixed_count = 0
        
        # 3. CIRUGÍA CON MEMORIA OPTIMIZADA (no_grad)
        # Esto evita que PyTorch guarde gráficos de cálculo en la RAM
        with torch.no_grad():
            if hasattr(sd_model, 'conditioner'):
                try:
                    for embedder in sd_model.conditioner.embedders:
                        if hasattr(embedder, 'wrapped'): 
                            model_part = embedder.wrapped
                        else:
                            model_part = embedder

                        if hasattr(model_part, 'state_dict'):
                            for key, tensor in model_part.state_dict().items():
                                # Verificación rápida
                                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                                    tensor.copy_(torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0))
                                    fixed_count += 1
                except Exception as e:
                    print(f"⚠️ Herrscher Shield Warning: {e}")

        # 4. REPORTE Y REGISTRO
        if fixed_count > 0:
            print(f"✅ HERRSCHER SHIELD: Se repararon {fixed_count} tensores corruptos.")
        else:
            print("✨ El modelo está sano (o ya fue reparado).")
        
        # Marcamos este modelo como "Arreglado"
        self.last_fixed_checkpoint = current_checkpoint_name

        # 5. LIMPIEZA DE RAM (GARBAGE COLLECTION)
        if force_clean:
            n = gc.collect()
            torch.cuda.empty_cache()
            # print(f"🧹 RAM Limpia: Se liberaron {n} objetos de memoria.")
