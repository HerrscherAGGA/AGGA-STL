import modules.scripts as scripts
import gradio as gr
import torch
import gc
import ctypes
from modules import shared

try:
    libc = ctypes.CDLL("libc.so.6")
except Exception:
    libc = None

class AggaOptimizer(scripts.Script):
    def title(self):
        return "Herrscher AGGA - System RAM Nuclear Flush"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("🚀 AGGA Optimizer V4.1", open=False):
            enabled = gr.Checkbox(label="Enable Aggressive Cleanup", value=True)
            force_sys_flush = gr.Checkbox(label="Force Linux RAM Release (malloc_trim)", value=True)
            deep_vram = gr.Checkbox(label="Deep VRAM Purge (Slower)", value=True)
            gr.Markdown(f"**Kernel Linux:** {'✅ Detectado' if libc else '❌ No detectado'}")
        return [enabled, force_sys_flush, deep_vram]

    def _flush_memory(self, sys_flush, deep_vram):
        # 1. GC en modo agresivo (todas las generaciones)
        gc.collect()
        
        # 2. VRAM Purge
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if deep_vram:
                torch.cuda.ipc_collect()

        # 3. El 'Martillo' de Linux
        if sys_flush and libc:
            try:
                # Forzamos al asignador de memoria de C a soltar los bloques vacíos
                libc.malloc_trim(0)
            except:
                pass

    def process(self, p, enabled, force_sys_flush, deep_vram):
        if not enabled: return
        
        # Limpieza ligera preventiva para no ralentizar el inicio
        gc.collect()
        
        # Aseguramos que A1111 no intente guardar modelos en RAM
        if shared.opts.sd_checkpoint_cache > 0:
            shared.opts.sd_checkpoint_cache = 0
        if hasattr(shared.opts, 'sd_vae_checkpoint_cache') and shared.opts.sd_vae_checkpoint_cache > 0:
            shared.opts.sd_vae_checkpoint_cache = 0

    def postprocess(self, p, processed, enabled, force_sys_flush, deep_vram):
        if not enabled: return
        
        self._flush_memory(force_sys_flush, deep_vram)
        
        # Feedback real en consola
        if torch.cuda.is_available():
            # Mostramos la memoria reservada actual, que es la que realmente ocupa espacio
            vram_final = torch.cuda.memory_reserved() / 1024**3
            print(f"🧹 [AGGA] Flush Nuclear completado. VRAM actual: {vram_final:.2f} GB")