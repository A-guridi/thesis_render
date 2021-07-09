# thesis_render

## polarization renderer
This repo is based on the optiX renderer of the paper "Through the looking glass", to which we added polarization of light.

For that, under the src/materials/ folder, we can see in the microfacet materials that light is operated in the form of Stokes Vectors instead of the classical RGB light.
For simplicity, this code was later moved together to a new repo with the Mitsuba2 renderer, which provides a more complete implementation of polarization and
already some features. Please refer to that repo for a more deeper look at the rendering pipeline.

This renderer can be called by simply using cmake to build it and then calling the exec "optixRenderer" with the optional parameters as described in "Through the looking glass". 
