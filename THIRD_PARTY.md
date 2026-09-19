# Third-party notices

The CUDA extensions are registered as pinned Git submodules under
`submodules/`; GLM is a nested submodule of the rasterizer. Their source is
fetched from upstream with its original source headers and license files.

- 2DGS / 3DGS: original source headers and `LICENSE.md` are retained. See the
  [2DGS repository](https://github.com/hbb1/2d-gaussian-splatting) and
  [3DGS repository](https://github.com/graphdeco-inria/gaussian-splatting) for
  upstream documentation, acknowledgements and citations.
- diff-surfel-rasterization: `e0ed0207b3e0669960cfad70852200a4a5847f61`, upstream
  https://github.com/hbb1/diff-surfel-rasterization; preserve its own license.
- simple-knn: `f155ec04131cb579f53443a06879d37115f4612f`, upstream
  https://gitlab.inria.fr/bkerbl/simple-knn; preserve its own license.
- GLM: `5c46b9c07008ae65cb81ab79cd677ecc1934b903`, upstream
  https://github.com/g-truc/glm; preserve its own license.
- Depth Pro: https://github.com/apple/ml-depth-pro at
  `9efe5c1def37a26c5367a71df664b18e1306c708`; source and pretrained checkpoints
  are downloaded separately and remain subject to Apple's respective terms.
- The inherited LPIPS implementation, Open3D/meshing utilities, Multinerf camera
  utilities and original dataset benchmark scripts retain their attribution.
- The camera adapter includes attribution to a LongSplat-style implementation
  in its source comments.

Third-party code, datasets and pretrained checkpoints are subject to their
respective licenses and terms. No dataset or pretrained checkpoint is included.
