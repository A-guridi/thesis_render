file(REMOVE_RECURSE
  "../../lib/ptx/optixRenderer_generated_path_trace_camera.cu.ptx"
  "../../lib/ptx/optixRenderer_generated_triangle_mesh.cu.ptx"
  "../../lib/ptx/optixRenderer_generated_areaLight.cu.ptx"
  "../../lib/ptx/optixRenderer_generated_envmap.cu.ptx"
  "../../lib/ptx/optixRenderer_generated_albedo.cu.ptx"
  "../../lib/ptx/optixRenderer_generated_depth.cu.ptx"
  "../../lib/ptx/optixRenderer_generated_diffuse.cu.ptx"
  "../../lib/ptx/optixRenderer_generated_mask.cu.ptx"
  "../../lib/ptx/optixRenderer_generated_metallic.cu.ptx"
  "../../lib/ptx/optixRenderer_generated_microfacet.cu.ptx"
  "../../lib/ptx/optixRenderer_generated_normal.cu.ptx"
  "../../lib/ptx/optixRenderer_generated_phong.cu.ptx"
  "../../lib/ptx/optixRenderer_generated_roughness.cu.ptx"
  "../../lib/ptx/optixRenderer_generated_dielectric.cu.ptx"
  "../../lib/ptx/optixRenderer_generated_conductor.cu.ptx"
  "CMakeFiles/optixRenderer.dir/optixRenderer.cpp.o"
  "CMakeFiles/optixRenderer.dir/creator/createAreaLight.cpp.o"
  "CMakeFiles/optixRenderer.dir/creator/createCamera.cpp.o"
  "CMakeFiles/optixRenderer.dir/creator/createContext.cpp.o"
  "CMakeFiles/optixRenderer.dir/creator/createEnvmap.cpp.o"
  "CMakeFiles/optixRenderer.dir/creator/createGeometry.cpp.o"
  "CMakeFiles/optixRenderer.dir/creator/createMaterial.cpp.o"
  "CMakeFiles/optixRenderer.dir/creator/createPointFlashLight.cpp.o"
  "CMakeFiles/optixRenderer.dir/creator/createTextureSampler.cpp.o"
  "CMakeFiles/optixRenderer.dir/inout/readXML.cpp.o"
  "CMakeFiles/optixRenderer.dir/inout/loadBsdfFromXML.cpp.o"
  "CMakeFiles/optixRenderer.dir/inout/loadSensorFromXML.cpp.o"
  "CMakeFiles/optixRenderer.dir/inout/loadLightFromXML.cpp.o"
  "CMakeFiles/optixRenderer.dir/inout/loadShapeFromXML.cpp.o"
  "CMakeFiles/optixRenderer.dir/inout/relativePath.cpp.o"
  "CMakeFiles/optixRenderer.dir/inout/rgbe.cpp.o"
  "CMakeFiles/optixRenderer.dir/material/albedo.cpp.o"
  "CMakeFiles/optixRenderer.dir/material/depth.cpp.o"
  "CMakeFiles/optixRenderer.dir/material/diffuse.cpp.o"
  "CMakeFiles/optixRenderer.dir/material/mask.cpp.o"
  "CMakeFiles/optixRenderer.dir/material/metallic.cpp.o"
  "CMakeFiles/optixRenderer.dir/material/microfacet.cpp.o"
  "CMakeFiles/optixRenderer.dir/material/normal.cpp.o"
  "CMakeFiles/optixRenderer.dir/material/phong.cpp.o"
  "CMakeFiles/optixRenderer.dir/material/roughness.cpp.o"
  "CMakeFiles/optixRenderer.dir/material/dielectric.cpp.o"
  "CMakeFiles/optixRenderer.dir/material/conductor.cpp.o"
  "CMakeFiles/optixRenderer.dir/sampler/sampler.cpp.o"
  "CMakeFiles/optixRenderer.dir/utils/ptxPath.cpp.o"
  "CMakeFiles/optixRenderer.dir/postprocessing/filter.cpp.o"
  "../../bin/optixRenderer.pdb"
  "../../bin/optixRenderer"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/optixRenderer.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
