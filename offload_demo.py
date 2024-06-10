import mindspore as ms
from mindone.diffusers import StableDiffusionXLPipeline

offload_config = {"offload_param": "cpu",
                  "auto_offload": False,
                  "offload_cpu_size": "512GB",
                  "offload_disk_size": "1024GB",
                  "offload_path": "./offload/",
                  "host_mem_block_size":"1GB",
                  "enable_aio": True,
                  "enable_pinned_mem": True}
ms.set_context(mode=ms.GRAPH_MODE, memory_offload='ON', max_device_memory='30GB')
ms.set_offload_context(offload_config=offload_config)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    mindspore_dtype=ms.float16,
    use_safetensors=True
)

prompt = "An astronaut riding a green horse"

images = pipe(prompt=prompt)[0][0]