from triposg.pipelines.pipeline_triposg import TripoSGPipeline

pipe = TripoSGPipeline.from_pretrained("VAST-AI/TripoSG").to("cuda")

out = pipe(
    image="assets/example_data/hjswed.png",
    prompt="مجسم دلة نجدية مزخرفة بنقش السدو",
    output_path="output.glb"
)

print("✅ Arabic prompt generation done! Check output.glb")
