# Input and output frames

1. We are aiming to work in a realtime system, so the FPS should be at leat 30 frames per second.
2. For the input frames at each time step, we extract 4 of 30 as input, and to select the most informative frames/patches.
    - The extraction should respect the spatial layout of camera pose.
3. For the target frames at each time step, the model should predict well on both current frames and previous frames. So we define a sliding window of size 24, with 16 current and 8 previous frames.
4. The persistent GS is represented in a voxel format, with a resolution of 128x128x128.