# Lego Machine Repo

## Contents
- DemoWebsite: Simple combined website/inference web server for our presentation demo.
- SyntheticData: Python script for Blender that produces annotated, synthetic images of Legos.
- UIServer: Flask server that runs on a Raspberry Pi which captures images and forwards them to the inference machine upon request.
- website: Original website code which has since been integrated into the UIServer directory.
- start_lego_proj.sh: Bash script to be placed on a Raspberry Pi which creates an SSH tunnel to the inference server and starts both the UIServer and remotely starts the InferenceServer.
- start_lego_proj_inf.sh: Bash script to be placed on the inference machine which starts the InferenceServer. Called by start_lego_proj.sh.