import modules.scripts as scripts
import gradio as gr
import torch
import gc
import ctypes # <--- Necesario para hablar con Linux
from modules import shared

# Cargar la librería del sistema para gestión de memoria
try:
    libc = ctypes.CDLL("libc.so.6")
except:
    libc = None

class AggaOptimizer(scripts.Script):
    def title(self):
        return "Herrscher AGGA - System RAM Nuclear Flush"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("🚀 AGGA Optimizer V4 (System Flush)", open=False):
            enabled = gr.Checkbox(label="Enable Aggressive Cleanup", value=True)
            force_sys_flush = gr.Checkbox(label="Force Linux RAM Release (malloc_trim)", value=True)
            debug_info = gr.Markdown(value="*Libera RAM de Python al SO directamente.*")
        return [enabled, force_sys_flush]

    def clean_memory(self, sys_flush=False):
        # 1. Limpieza estándar de Python y CUDA
        gc.collect()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except:
            pass
            
        # 2. LIMPIEZA NUCLEAR (Solo Linux/Colab)
        if sys_flush and libc:
            try:
                # Esto obliga al sistema a reclamar la memoria vacía de Python
                libc.malloc_trim(0)
            except Exception:
                pass

    # ANTES DE GENERAR
    def process(self, p, enabled, force_sys_flush):
        if not enabled: return
        
        # Limpieza preventiva
        self.clean_memory(sys_flush=force_sys_flush)
        
        # Forzar configuración A1111
        if shared.opts.sd_checkpoint_cache > 0:
            shared.opts.sd_checkpoint_cache = 0

    # DESPUÉS DE GENERAR
    def postprocess(self, p, processed, enabled, force_sys_flush):
        if not enabled: return
        
        print("   🧹 [AGGA OPTIMIZER] Ejecutando purga del sistema...")
        self.clean_memory(sys_flush=force_sys_flush)
        
        # Reporte visual para ti
        try:
            mem_vram = torch.cuda.memory_allocated() / 1024**3
            print(f"   📉 VRAM: {mem_vram:.2f} GB | RAM Sistema: Optimizada.")
        except:
            pass