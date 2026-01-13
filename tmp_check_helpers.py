# Check the function signatures for conditioning functions
import subprocess
result = subprocess.run(
    ["grep", "-n", "-A", "30", "def image_conditionings_by_adding_guiding_latent", 
     "/opt/ltx2/packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py"],
    capture_output=True, text=True
)
print("=== image_conditionings_by_adding_guiding_latent ===")
print(result.stdout or result.stderr)

result2 = subprocess.run(
    ["grep", "-n", "-A", "30", "def image_conditionings_by_replacing_latent",
     "/opt/ltx2/packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py"],
    capture_output=True, text=True
)
print("\n=== image_conditionings_by_replacing_latent ===")
print(result2.stdout or result2.stderr)
