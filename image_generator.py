import os
import torch
from diffusers import StableDiffusionPipeline

# Create output folder
os.makedirs("generated_images", exist_ok=True)

# Load Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32
)

# Use CPU
pipe = pipe.to("cpu")

# Read prompts from file
with open("prompts.txt", "r", encoding="utf-8") as file:
    prompts = [line.strip() for line in file if line.strip()]

print(f"Found {len(prompts)} prompts.\n")

# Generate images
for i, prompt in enumerate(prompts, start=1):
    print(f"Generating image {i}/{len(prompts)}...")
    
    image = pipe(prompt).images[0]

    filename = f"generated_images/image_{i}.png"
    image.save(filename)

    print(f"Saved: {filename}")

print("\nAll images generated successfully!")