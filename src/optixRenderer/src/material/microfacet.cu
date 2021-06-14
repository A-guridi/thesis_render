/* 
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "helpers.h"
#include "structs/prd.h"
#include "light/envmap.h"
#include "random.h"
#include "commonStructs.h"
#include "lightStructs.h"
#include "light/areaLight.h"
#include <vector>

using namespace optix;


rtDeclareVariable( float3, shading_normal, attribute shading_normal, ); 
rtDeclareVariable( float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, tangent_direction, attribute tangent_direction, );
rtDeclareVariable(float3, bitangent_direction, attribute bitangent_direction, );
rtDeclareVariable(int, max_depth, , );

rtDeclareVariable( float3, texcoord, attribute texcoord, );
rtDeclareVariable( float, t_hit, rtIntersectionDistance, );

rtDeclareVariable(optix::Ray, ray,   rtCurrentRay, );
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow, rtPayload, );
rtDeclareVariable(float, scene_epsilon, , );

rtDeclareVariable( float, uvScale, , ); 

// Materials
rtDeclareVariable( float3, albedo, , );
rtTextureSampler<float4, 2> albedoMap;
rtDeclareVariable( int, isAlbedoTexture, , );
rtDeclareVariable( float, rough, , );
rtTextureSampler<float4, 2> roughMap;
rtDeclareVariable( int, isRoughTexture, , );
rtTextureSampler<float4, 2> normalMap;
rtDeclareVariable(int, isNormalTexture, , );
rtDeclareVariable(float, F0, , );
rtDeclareVariable( float, metallic, , );
rtDeclareVariable( int, isMetallicTexture, ,  );
rtTextureSampler<float4, 2> metallicMap;
rtDeclareVariable( int, isSpecularTexture, , );
rtDeclareVariable(float3, specular, , );
rtTextureSampler<float4, 2> specularMap;

// Area Light Buffer
rtDeclareVariable(int, isAreaLight, , );

// Environmental Lighting 
rtDeclareVariable(int, isEnvmap, , );
rtDeclareVariable(float, infiniteFar, , );

// Point lighting 
rtDeclareVariable(int, isPointLight, , );
rtDeclareVariable(int, pointLightNum, , );
rtBuffer<Point> pointLights;

// added polarization
#define M_PI     3.14159265358979323846
#define M_PI2    6.28318530717958647692
#define M_INV_PI 0.3183098861837906715

rtDeclareVariable(float, filterangle, , );
rtDeclareVariable(float, intIOR, , );
rtDeclareVariable(float, extIOR, , );

rtDeclareVariable(float3, cameraU, , );  // camera up vector to rotate the light

#define filterCos2A    cos( 2*filterangle *M_PI/180);     // we keep the name as in the original renderer thesis
#define filterSin2A    sin( 2*filterangle *M_PI/180);
#define filterEnabled   true;

//future note: dont use define for float3, does not work good
//#define nonMetalIoRn make_float3(intIOR, intIOR, intIOR); // the Index of Refraction, usually around 1.3-1.5 for glass materials
//#define metalIoRn make_float3(0.0); // non-metal
//#define metalIoRk make_float3(0.0); // the complex part of the index is 0 for non-metallic materials
/**
 * Added Light Data structures
 */
rtDeclareVariable( rtObject, top_object, , );

rtDeclareVariable(
        rtCallableProgramX<void(unsigned int&, float3&, float3&, float&)>,
        sampleEnvironmapLight, , );
rtDeclareVariable(
        rtCallableProgramX<void(unsigned int&, float3&, float3&, float3&, float&)>,
        sampleAreaLight, , );

//**************************************************************************************************
// Data structures and initializer functions
//**************************************************************************************************
//already declared in prd.h

RT_CALLABLE_PROGRAM StokesLight initStokes()
{
    StokesLight sl;
    sl.svR = make_float4(0.0, 0.0, 0.0, 0.0);
    sl.svG = make_float4(0.0, 0.0, 0.0, 0.0);
    sl.svB = make_float4(0.0, 0.0, 0.0, 0.0);
    sl.referenceX = make_float3(1.0, 0.0, 0.0);

    return sl;
}
//TODO: will it detect rgb from float3 radiance ?
RT_CALLABLE_PROGRAM StokesLight unPolarizedLight(float3 color)
{
    StokesLight sl = initStokes();
    sl.svR.x = color.x;
    sl.svG.x = color.y;
    sl.svB.x = color.z;
    return sl;
}

RT_CALLABLE_PROGRAM float3 stokesToColor(StokesLight sl) {
    float3 r=make_float3(saturate(sl.svR.x), saturate(sl.svG.x), saturate(sl.svB.x));
    return r;
}
//* operator for a light vector and a 4x4 matrix
RT_CALLABLE_PROGRAM float4 vec_mat_multi(float4 vec, float4x4 mat) {
    float4 col0=make_float4(mat.r0.x, mat.r1.x, mat.r2.x, mat.r3.x);
    float4 col1=make_float4(mat.r0.y, mat.r1.y, mat.r2.y, mat.r3.y);
    float4 col2=make_float4(mat.r0.z, mat.r1.z, mat.r2.z, mat.r3.z);
    float4 col3=make_float4(mat.r0.w, mat.r1.w, mat.r2.w, mat.r3.w);
    return make_float4(dot(vec, col0), dot(vec, col1), dot(vec, col2), dot(vec,col3) );
}
//**************************************************************************************************
// Operator macros for StokesLight and MuellerData
//**************************************************************************************************

// operator += for StokesLight and unpolarized float3 color
// unpolarized light can be added directly, no need to first create a Stokes vector for it
#define SL_ADD_EQ_UNPOL(sl, c) sl.svR.x += c.x; \
                               sl.svG.x += c.y; \
                               sl.svB.x += c.z;

// operator += for two already aligned StokesLight parameters
#define SL_ADD_EQ_POL(sl_a, sl_b) sl_a.svR += sl_b.svR; \
                                  sl_a.svG += sl_b.svG; \
                                  sl_a.svB += sl_b.svB;


// operator *= for StokesLight and MuellerData
#define SL_MUL_EQ_MD(sl, md) sl.svR = vec_mat_multi(sl.svR, md.mmR); \
                             sl.svG = vec_mat_multi(sl.svG, md.mmG); \
                             sl.svB = vec_mat_multi(sl.svB, md.mmB);


// operator *= for MuellerData and a scalar
#define MD_MUL_EQ_SCALAR(md, s) md.mmR *= s; md.mmG *= s; md.mmB *= s;

//**************************************************************************************************
// Ray payload
//**************************************************************************************************


RT_CALLABLE_PROGRAM void initPayload()
{
    // adapted function to use the prd_radiance instead of creating a new ray payload
    prd_radiance.lightData=initStokes();
    // pay.lightData = initStokes();    // old one, left for reference
}
//**************************************************************************************************
// StokesLight functions
//**************************************************************************************************

// Get the normalized reference x vector from normalized y and z vectors
RT_CALLABLE_PROGRAM float3 computeX(float3 y, float3 z)
{
    return normalize(cross(y, z));
}

/** Rotate the reference frame of a Stokes vector
  c2p: cos(2phi)
  s2p: sin(2phi)
  inout param substitued by reference &
*/
RT_CALLABLE_PROGRAM void rotateStokes(float4 &S, float c2p, float s2p)
{
    float old_y = S.y;
    float old_z = S.z;

    S.y =  float(c2p*old_y + s2p*old_z);
    S.z = float(-s2p*old_y + c2p*old_z);
}

/** Rotate reference frame
  The light's direction vector dir is needed to rotate the reference X around
*/
RT_CALLABLE_PROGRAM void rotateReferenceFrame(StokesLight &light, float3 newX, float3 dir)
{
    float dotX = dot(light.referenceX, newX);   //dot product
    float detX = dot(dir, cross(light.referenceX, newX));
    float phi = atan2(detX, dotX);

    float c2p = cos(2*phi);
    float s2p = sin(2*phi);

    rotateStokes(light.svR, c2p, s2p);
    rotateStokes(light.svG, c2p, s2p);
    rotateStokes(light.svB, c2p, s2p);
    light.referenceX = newX;
}

/** a+=b operator for StokesLight
  Rotates b's reference frame before addition if needed.
*/
RT_CALLABLE_PROGRAM void slAddEquals(StokesLight &a, StokesLight b, float3 dir)
{
    // Make sure b's reference frame matches a's before adding them
    rotateReferenceFrame(b, a.referenceX, dir);
    SL_ADD_EQ_POL(a, b);
}

/** Applies the polarizing filter to a Stokes vector s
*/
RT_CALLABLE_PROGRAM void polarizeStokes(float4 &s)
{
    const float a = filterCos2A;
    const float b = filterSin2A;
    float3 oldXYZ = make_float3(s.x, s.y, s.z);

    s.x = float(dot(oldXYZ, make_float3(1.0,   a,   b)));
    s.y = float(dot(oldXYZ, make_float3(  a, a*a, a*b)));
    s.z = float(dot(oldXYZ, make_float3(  b, a*b, b*b)));
    s.w = float(0.0);
}

/** Applies a horizontal polarizing filter that's been rotated clockwise by angle A
  Note: also doubles the intensity to compensate for the filter on average blocking half of the
        incoming light.
*/
RT_CALLABLE_PROGRAM void applyPolarizingFilter(StokesLight &l)
{
    polarizeStokes(l.svR);
    polarizeStokes(l.svG);
    polarizeStokes(l.svB);
}
//************************************************************************************************
// Geometry Group (from optiX)
//************************************************************************************************

//************************************************************************************************
// Mueller Matrices
//************************************************************************************************
/** Mueller Matrix for Fresnel reflections
    n, k : Real and complex parts of the Index of Refraction
    theta: angle between h and v
*/
RT_CALLABLE_PROGRAM float4x4 F_MuellerMatrix(float n, float k, float sinTheta, float cosTheta, float tanTheta)
{
    float n2 = n*n;
    float k2 = k*k;
    float st2 = sinTheta*sinTheta;

    float left  = sqrt((n2 - k2 - st2)*(n2 - k2 - st2) + 4*n2*k2);
    float right = n2 - k2 - st2;

    float a2 = 0.5*(left + right);
    float b2 = 0.5*(left - right);

    float a = sqrt(a2);
    float b = sqrt(max(b2,0.0));
    float ct2 = cosTheta*cosTheta;

    // orthogonal
    float ortA = a2 + b2 + ct2;
    float ortB = 2.0*a*cosTheta;

    // parallel
    float parA = a2 + b2 + st2*tanTheta*tanTheta;
    float parB = 2.0*a*sinTheta*tanTheta;

    // Fresnel parameters
    float F_ort = (ortA - ortB)/(ortA + ortB);
    float F_par = ((parA - parB)/(parA + parB))*F_ort;
    float D_ort = atan((2*b*cosTheta)/(ct2 - a2 - b2));
    float D_par = atan((2*cosTheta*((n2 - k2)*b - 2*n*k*a))/((n2 + k2)*(n2 + k2)*ct2 - a2 - b2));

    float phaseDiff = D_ort - D_par;

    // Matrix components
    float A = 0.5*(F_ort + F_par);
    float B = 0.5*(F_ort - F_par);
    float C = cos(phaseDiff)*sqrt(F_ort*F_par);
    float S = sin(phaseDiff)*sqrt(F_ort*F_par);

    /*return float4x4(  A,   B, 0.0, 0.0,
                      B,   A, 0.0, 0.0,
                      0.0, 0.0,   C,   S,
                      0.0, 0.0,  -S,   C);*/
    float4x4 result;
    result.r0=make_float4(A, B, 0.0, 0.0);
    result.r1=make_float4(B, A, 0.0, 0.0);
    result.r2=make_float4(0.0, 0.0, C, S);
    result.r3=make_float4(0.0, 0.0, -S, C);

    return result;
}
/** Polarization sensitive Fresnel term (F)
*/
RT_CALLABLE_PROGRAM MuellerData F_Polarizing(float metalness, float sinTheta, float cosTheta, float tanTheta)
{
    // Index of Refraction is not available in material textures so it is set from the constant buffer
    //IoR_n = nonMetalIoRn + metalness*(metalIoRn-nonMetalIoRn);
    float3 metalIoRn=make_float3(0.0);
    float3 nonMetalIoRn=make_float3(intIOR);
    float3 metalIoRk=make_float3(0.0);
    float3 IoR_n = nonMetalIoRn + metalness*(metalIoRn-nonMetalIoRn);
    float3 IoR_k = metalIoRk * metalness; // k is zero for non-metals

    MuellerData mdF;
    mdF.mmR = F_MuellerMatrix(IoR_n.x, IoR_k.x, sinTheta, cosTheta, tanTheta);
    mdF.mmG = F_MuellerMatrix(IoR_n.y, IoR_k.y, sinTheta, cosTheta, tanTheta);
    mdF.mmB = F_MuellerMatrix(IoR_n.z, IoR_k.z, sinTheta, cosTheta, tanTheta);

    return mdF;
}
RT_CALLABLE_PROGRAM float D_GGX(float rough, const float3& N, const float3& H)
{
    float a2 = rough*rough*rough*rough;
    float NdotH = fmaxf(dot(N,H), 0);
    float d  = ((NdotH*a2 - NdotH)*NdotH + 1.0);
    return a2/(M_PI*d*d);
}

/** Smith-GGX Visibility Function (V)
    V = G/(4*NdotL*NdotV)
*/
RT_CALLABLE_PROGRAM float V_SmithGGX(const float3& N, const float3& L, const float3& V, float roughness)
{
    float a2 = roughness*roughness*roughness*roughness;
    float NdotL = fmaxf(dot(N,L), 0);
    float NdotV = fmaxf(dot(N,V), 0);
    float ggxv = NdotL*sqrt((-NdotV*a2 + NdotV)*NdotV + a2);
    float ggxl = NdotV*sqrt((-NdotL*a2 + NdotL)*NdotL + a2);
    return 0.5/(ggxv + ggxl);
}

// Computing the pdfSolidAngle of BRDF giving a direction
// Lambertian Difuse, same as original in OptiX
RT_CALLABLE_PROGRAM float LambertianPdf(const float3& L, const float3& N)
{
    float NoL = fmaxf(dot(N, L), 0);
    float pdf = NoL / M_PI;
    return fmaxf(pdf, 1e-14f);
}
// Old SpecularSpecular Relfection
/**
RT_CALLABLE_PROGRAM float SpecularPdf(const float3& L, const float3& V, const float3& N, float R)
{
    float a2 = R * R * R * R;
    float3 H = normalize( (L+V) / 2.0 );
    float NoH = fmaxf(dot(N, H), 0);
    float VoH = fmaxf(dot(V, H), 0);
    float pdf = (a2 * NoH) / fmaxf( (4 * M_PI * (1 + (a2-1) * NoH)
            *(1 + (a2-1) * NoH) * VoH ), 1e-14f);
    return fmaxf(pdf, 1e-14f);
}
 **/
// New Specular refrection
/** Cook-Torrance Specular Term
 * Ks= D*V*F
 * the F term is the polarized one with the Mueller matrices
 * the roughness=arccos(dot(N,H))
 * N: surface normal
 * H: half-angle vector
 * L: light vector
*/

// returns the cookTorrance specular reflection ( not the mirror one)
RT_CALLABLE_PROGRAM MuellerData CookTorrance_Pol(float roughness, float metalness, const float3& N, const float3& L, const float3& V)
{
    // do we need to divide by 2? or take it out for a bigger light intensity
    float3 H = normalize((V + L)/2.0f);

    float  D = D_GGX(roughness, N, H);
    float  V_Smith = V_SmithGGX(N, L, V, roughness);

    float sinTheta = length(cross(L, H));
    float cosTheta = fmaxf(dot(L,H), 0); // used since (LdotH == VdotH)
    float tanTheta = sinTheta/cosTheta;
    float NdotL = fmaxf(dot(N,L), 0);

    MuellerData F = F_Polarizing(metalness, sinTheta, cosTheta, tanTheta);
    MD_MUL_EQ_SCALAR(F, (D*V_Smith*NdotL));

    return F;
}
/** Calculates the reflection Stokes components using the Cook-Torrance specular term
*/
/**
StokesLight getReflectionData(const float3& N, const float3& L, const float3& V, const float3& originW,
                              int hitDepth, float roughness, float metalness)
{
    // N is used instead of H since (N == H) for perfect reflections.
    float NdotV = saturate(dot(N, V));

    // Early exit if we're out of rays or if the surface is not facing the ray
    if (hitDepth >= max_depth || NdotV <= 0.0 ) {
        return unPolarizedLight(REFL_MISS_COLOR);
    }

    Payload rPayload = initPayload();

    //RayDesc rRay;
    //rRay.Origin = originW;
    //rRay.Direction = reflect(-V, sd.N);   // ==L
    //rRay.TMin = TMIN;
    //rRay.TMax = TMAX;

    float NdotL = fmaxf(dot(N, L));
    float sinTheta = length(cross(L, N));
    float cosTheta = NdotL;
    float tanTheta = sinTheta/cosTheta;
    // mirror reflection
    float3 H=N;
    float D = D_GGX(roughness, N, H); // NdotH=1.0 since N == H *(only for pure mirror reflection)
    float V = V_SmithGGX(N, L, V, roughness);

    MuellerData reflectionBrdf = F_Polarizing(metalness, sinTheta, cosTheta, tanTheta);
    //saturate to prevent blown out colors
    MD_MUL_EQ_SCALAR(reflectionBrdf, saturate(D*V*NdotL));

    // Send a reflection ray into the scene
    TraceRay(gRtScene, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, hitProgramCount, 0, rRay, rPayload);

    // Align the incoming light's reference frame
    float3 incomingRefX = computeX(sd.N, rRay.Direction);
    rotateReferenceFrame(rPayload.lightData, incomingRefX, rRay.Direction);

    // Multiply with the reflection BRDF Mueller matrices
    SL_MUL_EQ_MD(rPayload.lightData, reflectionBrdf);

    return rPayload.lightData;
}
**/
// this function calculates the mirror term of the light, which is algo polarized and in the form of a Mueller matrix
// the mirror term is the same as the CookTorrance specular term, but now H=N
RT_CALLABLE_PROGRAM MuellerData mirrorTerm (const float3& L, const float3& V, const float3& N, float roughness, float metalness)
{

    // calculation of the mirror term
    float3 H=N;
    float D = D_GGX(roughness, N, H); // NdotH=1.0 since (N == H)
    float V_Smith = V_SmithGGX(N, L, V, roughness);
    float NdotL = fmaxf(dot(N, L), 1e-14);
    float sinTheta = length(cross(L, N));
    float cosTheta = NdotL;
    float tanTheta = sinTheta/cosTheta;

    MuellerData reflectionBrdf = F_Polarizing(metalness, sinTheta, cosTheta, tanTheta);
    MD_MUL_EQ_SCALAR(reflectionBrdf, saturate(D*V_Smith*NdotL));
    return reflectionBrdf;

}


// old pdf function
/**
RT_CALLABLE_PROGRAM float pdf(const float3& L, const float3& V, const float3& N, float R)
{
    float pdfLambertian = LambertianPdf(L, N);
    float pdfSpecular = SpecularPdf(L, V, N, R);
    return pdfLambertian * 0.5 + pdfSpecular * 0.5;
}
**/

//(d) pdfLambertian: Lambertian Difuse relfection term ( no polarization)
//(s) specMueller: cookTorrance specular reflection term (polarized in the Fresnel 'F' term)
//(m) reflectionBrdf: mirror reflection term (also polarized in F), but now H==N

// this function creates the pdf term of the prd_radiance, which is then in areaLight.cu and envmap.cu used to
// calculate the intensity of light
//this pdf is also known as the Bidirectional Reflectance Distribution Function (BRDF)
RT_CALLABLE_PROGRAM float pdf(const float3& L, const float3& V, const float3& N, float roughness, float metalness,
                              const float3& radiance)
{
    // the difuse and specular terms are added in the evaluate function for each ray
    float pdfLambertian = LambertianPdf(L, N);
    MuellerData specMueller=CookTorrance_Pol(roughness, metalness, N, L, V);
    StokesLight specularStokes = unPolarizedLight(radiance);
    SL_MUL_EQ_MD(specularStokes, specMueller);

    // in this function we take those terms saved and add the mirror reflection to it
    MuellerData mirrorMueller=mirrorTerm(L, V, N, roughness, metalness);
    StokesLight mirrorStokes = unPolarizedLight(radiance);
    SL_MUL_EQ_MD(mirrorStokes, mirrorMueller);
    //SL_MUL_EQ_MD(prd_radiance.lightData, mirrorMueller);
    //SL_MUL_EQ_MD(prd_radiance.lightData, specMueller);
    SL_ADD_EQ_POL(specularStokes, mirrorStokes);

    float3 new_intensity = make_float3( specularStokes.svR.x, specularStokes.svG.x, specularStokes.svB.x );
    //we return the modulo of the intensity vector of the light as the BRDF
    //new_intensity*=pdfLambertian;
    return length(new_intensity);
}

// this function gets a ray data and returns the intensity of the light calculated

// new evaluate function with polarized specular term CookTorrance and unpolarized term Lambert-Difuse
RT_CALLABLE_PROGRAM float3 evaluate(const float3& albedoValue, const float3& specularValue, const float3& N, const float rough, const float3& fresnel,
                                    const float3& V, const float3& L, const float3& radiance, const float metalness)
{
    float alpha = rough * rough;
    float k = (alpha + 2 * rough + 1) / 8.0;
    float alpha2 = alpha * alpha;

    float3 H = normalize((L + V) / 2.0f );
    float NoL = fmaxf(dot(N, L), 0);
    float NoV = fmaxf(dot(N, V), 0);
    float NoH = fmaxf(dot(N, H), 0);
    float VoH = fmaxf(dot(V, H), 0);

    /* Diffuse component */
    // Diffuse is unpolarized so calculations with a float3 is sufficient
    float diffuseComp = LambertianPdf(L, N);
    float3 difusseLight = albedoValue*diffuseComp;

    /* Specular component */
    MuellerData specularMueller = CookTorrance_Pol(rough, metalness, N, L, V);

    // All light sources are unpolarized so no reference frame rotation needed before multiplication
    StokesLight specularStokes = unPolarizedLight(specularValue);
    SL_MUL_EQ_MD(specularStokes, specularMueller);

    // The output reference frame's Y vector lies in the specular reflection's plane of
    // incidence so the microfacet normal H is used to calculate the X vector
    specularStokes.referenceX = computeX(H, V);

    // slAddEquals will rotate reference frame if needed
    slAddEquals( prd_radiance.lightData, specularStokes, -ray.direction);
    SL_ADD_EQ_UNPOL( prd_radiance.lightData, difusseLight);
    float3 intensity = make_float3( prd_radiance.lightData.svR.x, prd_radiance.lightData.svG.x, prd_radiance.lightData.svB.x );

    //the intensity is a float3 vector with the values of RGB and the radiance is the intensity of each channel (how bright)
    //return intensity*radiance;
    //float3 specularTerm = specularValue / (2*M_PI) * (2 + 2) * pow(VoH, fmaxf(0, 1e-14) );
    return intensity * radiance;
    //return intensity;
}
// this functions samples the new ray and calculates the spatial information of it
RT_CALLABLE_PROGRAM void sample(unsigned& seed, const float3& albedoValue, const float3& N, const float rough,
                                const float metalness ,const float3& fresnel, const float3& V, const float3& ffnormal,
                                optix::Onb onb, float3& attenuation, float3& direction, float& pdfSolid)
{
    const float z1 = rnd( seed );
    const float z2 = rnd( seed );
    const float z = rnd( seed );
    
    float alpha = rough * rough;
    float k = (alpha + 2 * rough + 1) / 8.0;
    float alpha2 = alpha * alpha;
    
    float3 L;
    if(z < 0.5){
        cosine_sample_hemisphere(z1, z2, L);
        onb.inverse_transform(L);
        direction = L;
        attenuation =  2 * attenuation * albedoValue;
    }
    else{
        // Compute the half angle 
        float phi = 2 * M_PI * z1;
        float cosTheta = sqrt( (1 - z2) / (1 + (alpha2 - 1) * z2) );
        float sinTheta = sqrt( 1 - cosTheta * cosTheta);

        float3 H = make_float3(
                sinTheta * cos(phi),
                sinTheta * sin(phi),
                cosTheta );
        onb.inverse_transform(H);
        L = 2 * dot(V, H) * H - V;
        direction = L;

        float NoV = fmaxf(dot(N, V), 0.0);
        float NoL = dot(N, L);
        float NoH = fmaxf(dot(N, H), 0.0);
        float VoH = fmaxf(dot(V, H), 0.0);

        if( dot(ffnormal, L) >= 0.05 ){
            float G1 = NoV / (NoV * (1-k) + k);
            float G2 = NoL / (NoL * (1-k) + k);
            float FMi = (-5.55473 * VoH - 6.98316) * VoH;
            float3 F = fresnel + (1 - fresnel) * pow(2.0f, FMi);
            float3 reflec = F * G1 * G2 * VoH / fmaxf(NoH * NoV, 1e-14);

            attenuation = 2 * attenuation * reflec;
        }
        else{
            attenuation = make_float3(0.0f);
        }
    }
    // initialize the ray stokes information
    initPayload();
    //calculate the pdf term
    float3 init_radiance=make_float3(1.0);
    pdfSolid = pdf(L, V, N, rough, metalness, init_radiance);
    // get the cameras reference and rotate the light according to it


RT_PROGRAM void closest_hit_radiance()
{
    const float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
    const float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
    float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );
 
    float3 albedoValue;
    if(isAlbedoTexture == 0){
        albedoValue = albedo;
    }
    else{
        albedoValue = make_float3(tex2D(albedoMap, texcoord.x * uvScale, texcoord.y * uvScale) );
        albedoValue.x = pow(albedoValue.x, 2.2);
        albedoValue.y = pow(albedoValue.y, 2.2);
        albedoValue.z = pow(albedoValue.z, 2.2);
    }

    float roughValue = (isRoughTexture == 0) ? rough :
        tex2D(roughMap, texcoord.x * uvScale, texcoord.y * uvScale).x;

    float metallicValue = (isMetallicTexture == 0) ? metallic :
        tex2D(metallicMap, texcoord.x * uvScale, texcoord.y * uvScale).x;
    //R0 or specular color, used later for Schlick's approximation for the Fresnel term, not needed since
    // a polarized Fresnel term is used
    float3 fresnel = F0 * (1 - metallicValue) + metallicValue * albedoValue; //albedoValue= C_base (base color)
    albedoValue = (1 - metallicValue) * albedoValue;

    //added specular light from phong.cu
    float3 specularValue;
    if(isSpecularTexture == 0){
        specularValue = specular;
    }
    else{
        specularValue = make_float3(tex2D(specularMap, texcoord.x * uvScale, texcoord.y * uvScale) );
        specularValue.x = pow(specularValue.x, 2.2);
        specularValue.y = pow(specularValue.y, 2.2);
        specularValue.z = pow(specularValue.z, 2.2);
    }


    float3 colorSum = fmaxf(albedoValue + specularValue, make_float3(1e-14f) );
    float colorMax= colorSum.x;
    if(colorMax < colorSum.y) colorMax = colorSum.y;
    if(colorMax < colorSum.z) colorMax = colorSum.z;
    colorMax = fmaxf(colorMax, 1e-14);

    if(colorMax > 1){
        specularValue = specularValue / colorMax;
        albedoValue = albedoValue / colorMax;
    }
    
    float3 V = normalize(-ray.direction );    
    if(dot(ffnormal, V) < 0)
        ffnormal = -ffnormal;
    
    float3 N;
    if( isNormalTexture == 0){
        N = ffnormal;
    }
    else{
        N = make_float3(tex2D(normalMap, texcoord.x * uvScale, texcoord.y * uvScale) );
        N = normalize(2 * N - 1);
        N = N.x * tangent_direction 
            + N.y * bitangent_direction 
            + N.z * ffnormal;
    }
    N = normalize(N );
    optix::Onb onb(N );
 
    float3 hitPoint = ray.origin + t_hit * ray.direction;
    prd_radiance.origin = hitPoint;

    // Connect to the area Light
    {
        if(isAreaLight == 1){
            float3 position, radiance, normal;
            float pdfAreaLight;
            sampleAreaLight(prd_radiance.seed, radiance, position, normal, pdfAreaLight);
   
            float Dist = length(position - hitPoint);
            float3 L = normalize(position - hitPoint);

            if(fmaxf(dot(ffnormal, L), 0.0f) * fmaxf(dot(ffnormal, V), 0.0f) > 0.0025 ){
                float cosPhi = dot(L, normal);
                cosPhi = (cosPhi < 0) ? -cosPhi : cosPhi;

                Ray shadowRay = make_Ray(hitPoint, L, 1, scene_epsilon, Dist - scene_epsilon);
                PerRayData_shadow prd_shadow; 
                prd_shadow.inShadow = false;
                rtTrace(top_object, shadowRay, prd_shadow);
                if(prd_shadow.inShadow == false)
                {
                    float3 intensity = evaluate(albedoValue, specularValue, N, roughValue, fresnel, V, L, radiance, metallicValue) * cosPhi / Dist / Dist;
                    
                    if(prd_radiance.depth == (max_depth - 1) ){
                    }
                    else{
                        float pdfAreaLight2 = pdfAreaLight * pdfAreaLight;
                        float pdfSolidBRDF = pdf(L, V, N, roughValue, metallicValue, radiance);
                        float pdfAreaBRDF = pdfSolidBRDF * cosPhi / Dist / Dist;
                        float pdfAreaBRDF2 = pdfAreaBRDF * pdfAreaBRDF;

                        prd_radiance.radiance += intensity * pdfAreaLight / 
                            fmaxf(pdfAreaBRDF2 + pdfAreaLight2, 1e-14) * prd_radiance.attenuation;            
                    }
                    float3 cameraX  = computeX( cameraU, -ray.direction);
                    rotateReferenceFrame(prd_radiance.lightData, cameraX, -ray.direction);

                    /* Apply polarizing filter */
                    applyPolarizingFilter(prd_radiance.lightData);
                }
                }
            }
        }   
    }

    
    // Connect to point light 
    {
        if(isPointLight == 1){
            // Connect to every point light 
            for(int i = 0; i < pointLightNum; i++){
                float3 position = pointLights[i].position;
                float3 radiance = pointLights[i].intensity;
                float3 L = normalize(position - hitPoint);
                float Dist = length(position - hitPoint);

                if( fmaxf(dot(ffnormal, L), 0.0f) * fmaxf(dot(ffnormal, V), 0.0f) > 0.0025){
                    Ray shadowRay = make_Ray(hitPoint + 0.1 * L * scene_epsilon, L, 1, scene_epsilon, Dist - scene_epsilon);
                    PerRayData_shadow prd_shadow; 
                    prd_shadow.inShadow = false;
                    rtTrace(top_object, shadowRay, prd_shadow);
                    if(prd_shadow.inShadow == false && prd_radiance.depth != (max_depth - 1) ){
                        float3 intensity = evaluate(albedoValue, specularValue, N, roughValue, fresnel, V, L, radiance, metallicValue) / Dist/ Dist;
                        prd_radiance.radiance += intensity * prd_radiance.attenuation;
                    }
                    float3 cameraX  = computeX( cameraU, -ray.direction);
                    rotateReferenceFrame(prd_radiance.lightData, cameraX, -ray.direction);

                    /* Apply polarizing filter */
                    applyPolarizingFilter(prd_radiance.lightData);
                }
                }
            }
        }
    }

    // Connect to the environmental map 
    { 
        if(isEnvmap == 1){
            float3 L, radiance;
            float pdfSolidEnv;
            sampleEnvironmapLight(prd_radiance.seed, radiance, L, pdfSolidEnv);

            if( fmaxf(dot(L, ffnormal), 0.0f) * fmaxf(dot(V, ffnormal ), 0.0f) > 0.0025){
                Ray shadowRay = make_Ray(hitPoint + 0.1 * scene_epsilon * L, L, 1, scene_epsilon, infiniteFar);
                PerRayData_shadow prd_shadow;
                prd_shadow.inShadow = false;
                rtTrace(top_object, shadowRay, prd_shadow);
                if(prd_shadow.inShadow == false)
                {
                    float3 intensity = evaluate(albedoValue, specularValue, N, roughValue, fresnel, V, L, radiance, metallicValue);
                    if(prd_radiance.depth == (max_depth - 1) ){
                    }
                    else{
                        float pdfSolidBRDF = pdf(L, V, N, roughValue, metallicValue, radiance);
                        float pdfSolidBRDF2 = pdfSolidBRDF * pdfSolidBRDF;
                        float pdfSolidEnv2 = pdfSolidEnv * pdfSolidEnv;
                        prd_radiance.radiance += intensity * pdfSolidEnv / 
                            fmaxf(pdfSolidEnv2 + pdfSolidBRDF2, 1e-14) * prd_radiance.attenuation; 
                    }
                    float3 cameraX  = computeX( cameraU, -ray.direction);
                    rotateReferenceFrame(prd_radiance.lightData, cameraX, -ray.direction);
                    /* Apply polarizing filter */
                    applyPolarizingFilter(prd_radiance.lightData);
                    }
                }
            }
        }
    }


    // Sammple the new ray 
    sample(prd_radiance.seed, 
        albedoValue, N, fmaxf(roughValue, 0.02), metallicValue, fresnel, V,
        ffnormal, onb, 
        prd_radiance.attenuation, prd_radiance.direction, prd_radiance.pdf );


}

// any_hit_shadow program for every material include the lighting should be the same
RT_PROGRAM void any_hit_shadow()
{
    prd_shadow.inShadow = true;
    rtTerminateRay();
}

