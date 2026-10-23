---
title: "OpenReframe"
tagline: "Turn a single photo into a depth-aware virtual-camera view"
description: "A local-first computer vision app that estimates scene depth from one photograph, simulates a small camera translation, and uses AI inpainting to fill newly exposed regions."
category: "Computer Vision / Generative AI"
status: "shipped"
featured: true
technologies:
  - Depth Anything V2 (ViT-L)
  - Relative Depth Estimation
  - Depth-Aware Image Reprojection
  - Stable Diffusion XL Inpainting
  - Playwright
---

## Overview

OpenReframe turns a single photograph into a new, depth-aware view. It estimates the relative depth of the scene, uses that depth to simulate a small virtual-camera movement, and previews the resulting image—including regions that become visible only after the viewpoint shifts.

The project explores how monocular depth estimation and generative inpainting can work together to create a convincing reframed view from a single input image.

## Key Features

- **Single-image depth estimation:** Uses Depth Anything V2 ViT-L to estimate relative scene depth from the input photograph.
- **Depth preparation:** Converts the estimated depth into unitless pseudo-depth and smooths it while preserving important boundaries.
- **Virtual-camera movement:** Simulates a small translation-only camera shift rather than attempting a full 3D scene reconstruction.
- **Image reprojection:** Uses the depth map to project image content into the shifted view.
- **Disocclusion detection:** Identifies newly exposed areas that were hidden in the original viewpoint and previews those regions in magenta.
- **AI-assisted filling:** Uses Stable Diffusion XL inpainting to generate plausible content for the newly exposed areas, without ControlNet.
- **Interactive workflow:** Supports image upload, geometry preview, and smart planning as part of the application flow.

## My Contribution

I developed the end-to-end workflow connecting monocular depth estimation, depth-map preparation, virtual-camera reprojection, disocclusion detection, and generative inpainting. I also worked on the interactive preview workflow and validated the upload, geometry-preview, and smart-planning flow using a Playwright end-to-end test.

## Results

The tested workflow successfully completed image upload, geometry preview, and smart planning. The system produces a depth-aware reframed preview and identifies the newly exposed regions for inpainting.

The output is an approximate view synthesis, not a metrically accurate 3D reconstruction. Since the input contains only one photograph, hidden scene content must be inferred; inpainted regions are plausible generated fills rather than guaranteed recovery of the original scene.

## Technical Approach

1. Estimate relative depth from the input image with Depth Anything V2 ViT-L.
2. Normalize the depth into a unitless pseudo-depth representation and smooth it while preserving edges.
3. Apply a small translation-only virtual-camera shift and reproject the image using the depth map.
4. Detect disoccluded pixels that become exposed after reprojection.
5. Display the newly exposed regions in magenta and use Stable Diffusion XL inpainting to fill them.

## Limitations

- The estimated depth is relative and unitless; it should not be interpreted as real-world distance.
- The current camera simulation is limited to small translations.
- Newly exposed areas require generative inference, so the result may differ from the true hidden scene.

