
#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "helpers.h"
#include "lightStructs.h"
#include "light/areaLight.h"
#include "structs/prd.h"
#include "random.h"
#include "commonStructs.h"

using namespace optix;

rtDeclareVariable( float3, shading_normal, attribute shading_normal, ); 
rtDeclareVariable( float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable( float, t_hit, rtIntersectionDistance, ); // with this the hitPoint can be calculated
rtDeclareVariable(int, max_depth, , );

rtDeclareVariable(optix::Ray, ray,   rtCurrentRay, );
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow, rtPayload, );

rtDeclareVariable(float3, radiance, , );
rtDeclareVariable(float, areaSum, , );

rtDeclareVariable(int, areaTriangleNum, , );
rtBuffer<areaLight> areaLights;
rtBuffer<float> areaLightCDF;
rtBuffer<float> areaLightPDF;

// HERE STARTS THE CUSTOM IMPLEMENTATION OF THE SHADER
// the implementation is based on the ray tracing algorithm made in Falcor
#define TMIN 0.001
#define TMAX 10000.0

#define M_PI     3.14159265358979323846
#define M_PI2    6.28318530717958647692
#define M_INV_PI 0.3183098861837906715

#define COLOR_BLACK  float3(0.0, 0.0, 0.0)
#define COLOR_RED    float3(1.0, 0.0, 0.0)
#define COLOR_GREEN  float3(0.0, 1.0, 0.0)
#define COLOR_BLUE   float3(0.0, 0.0, 1.0)
#define COLOR_CYAN   float3(0.0, 1.0, 1.0)
#define COLOR_YELLOW float3(1.0, 1.0, 0.0)

#define REFL_MISS_COLOR COLOR_BLACK

//**************************************************************************************************
// We dont need constant buffers as they are the rtVaariables
//**************************************************************************************************

//**************************************************************************************************
// Data structures and initializer functions
//**************************************************************************************************

struct MuellerData
{
    // red, green, and blue Mueller matrices
    float4x4 mmR;
    float4x4 mmG;
    float4x4 mmB;
};

struct StokesLight
{
    // red, green, and blue Stokes vectors
    float4 svR;
    float4 svG;
    float4 svB;

    // local coordinate system's x-axis unit vector
    float3 referenceX;
};

StokesLight initStokes()
{
    StokesLight sl;
    sl.svR = float4(0.0, 0.0, 0.0, 0.0);
    sl.svG = float4(0.0, 0.0, 0.0, 0.0);
    sl.svB = float4(0.0, 0.0, 0.0, 0.0);
    sl.referenceX = float3(1.0, 0.0, 0.0);

    return sl;
}

StokesLight unPolarizedLight(float3 color)
{
    StokesLight sl = initStokes();
    sl.svR.x = color.r;
    sl.svG.x = color.g;
    sl.svB.x = color.b;
    return sl;
}

float3 stokesToColor(StokesLight sl) {
    return saturate(float3(sl.svR.x, sl.svG.x, sl.svB.x));
}
//**************************************************************************************************
// Operator macros for StokesLight and MuellerData
//**************************************************************************************************

// operator += for StokesLight and unpolarized float3 color
// unpolarized light can be added directly, no need to first create a Stokes vector for it
#define SL_ADD_EQ_UNPOL(sl, c) sl.svR.x += c.r; \
                               sl.svG.x += c.g; \
                               sl.svB.x += c.b;

// operator += for two already aligned StokesLight parameters
#define SL_ADD_EQ_POL(sl_a, sl_b) sl_a.svR += sl_b.svR; \
                                  sl_a.svG += sl_b.svG; \
                                  sl_a.svB += sl_b.svB;

// operator *= for StokesLight and MuellerData
#define SL_MUL_EQ_MD(sl, md) sl.svR = mul(sl.svR, md.mmR); \
                             sl.svG = mul(sl.svG, md.mmG); \
                             sl.svB = mul(sl.svB, md.mmB);


// operator *= for MuellerData and a scalar
#define MD_MUL_EQ_SCALAR(md, s) md.mmR *= s; md.mmG *= s; md.mmB *= s;

//**************************************************************************************************
// Ray payload
//**************************************************************************************************


void initPayload()
{
    // adapted function to use the prd_radiance instead of creating a new ray payload
    prd_radiance.lightData=initStokes();
}
//**************************************************************************************************
// StokesLight functions
//**************************************************************************************************

// Get the normalized reference x vector from normalized y and z vectors
float3 computeX(float3 y, float3 z)
{
    return normalize(cross(y, z));
}

/** Rotate the reference frame of a Stokes vector
  c2p: cos(2phi)
  s2p: sin(2phi)
  inout param substitued by reference &
*/
void rotateStokes(float4 &S, float c2p, float s2p)
{
    float old_y = S.y;
    float old_z = S.z;

    S.y =  c2p*old_y + s2p*old_z;
    S.z = -s2p*old_y + c2p*old_z;
}

/** Rotate reference frame
  The light's direction vector dir is needed to rotate the reference X around
*/
void rotateReferenceFrame(StokesLight &light, float3 newX, float3 dir)
{
    float dotX = dot(light.referenceX, newX);
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
void slAddEquals(StokesLight &a, StokesLight b, float3 dir)
{
    // Make sure b's reference frame matches a's before adding them
    rotateReferenceFrame(b, a.referenceX, dir);
    SL_ADD_EQ_POL(a, b);
}

/** Applies the polarizing filter to a Stokes vector s
*/
void polarizeStokes(float4 &s)
{
    uniform float a = filterCos2A;
    uniform float b = filterSin2A;
    float3 oldXYZ = s.xyz;

    s.x = dot(oldXYZ, float3(1.0,   a,   b));
    s.y = dot(oldXYZ, float3(  a, a*a, a*b));
    s.z = dot(oldXYZ, float3(  b, a*b, b*b));
    s.w = 0.0;
}

/** Applies a horizontal polarizing filter that's been rotated clockwise by angle A
  Note: also doubles the intensity to compensate for the filter on average blocking half of the
        incoming light.
*/
void applyPolarizingFilter(StokesLight &l)
{
    polarizeStokes(l.svR);
    polarizeStokes(l.svG);
    polarizeStokes(l.svB);
}
/**
 * RENDER PARTS
 * */
RT_CALLABLE_PROGRAM void sampleAreaLight(unsigned int& seed, float3& radiance, float3& position, float3& normal, float& pdfAreaLight){
    float randf = rnd(seed);

    int left = 0, right = areaTriangleNum;
    int middle = int( (left + right) / 2);
    while(left < right){
        if(areaLightCDF[middle] <= randf)
            left = middle + 1;
        else if(areaLightCDF[middle] > randf)
            right = middle;
        middle = int( (left + right) / 2);
    }
    areaLight L = areaLights[left];
    
    float3 v1 = L.vertices[0];
    float3 v2 = L.vertices[1];
    float3 v3 = L.vertices[2];

    normal = cross(v2 - v1, v3 - v1);
    float area = 0.5 * length(normal);
    normal = normalize(normal);

    float ep1 = rnd(seed);
    float ep2 = rnd(seed);
    
    float u = 1 - sqrt(ep1);
    float v = ep2 * sqrt(ep1);

    position = v1 + (v2 - v1) * u + (v3 - v1) * v;

    radiance = L.radiance;
    pdfAreaLight = areaLightPDF[left] /  fmaxf(area, 1e-14);
}

RT_PROGRAM void closest_hit_radiance()
{
    const float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
    const float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
    const float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

    if(prd_radiance.depth == 0){
        // Directly hit the light
        prd_radiance.radiance = radiance;
    }
    else{
        if(prd_radiance.pdf < 0){
            prd_radiance.radiance += radiance * prd_radiance.attenuation;
        }
        else{
            // Use MIS to compute the radiance
            if(prd_radiance.depth == (max_depth - 1) ){
                prd_radiance.radiance += radiance * prd_radiance.attenuation;
            }
            else{
                float3 hitPoint = ray.origin + t_hit * ray.direction;
                float Dist = length(hitPoint - prd_radiance.origin);
                float3 L = normalize(hitPoint - prd_radiance.origin);
                float cosPhi = dot(L, ffnormal);
                if (cosPhi < 0) cosPhi = -cosPhi;
                if (cosPhi < 1e-14) cosPhi = 0;
        
                float pdfAreaBRDF = prd_radiance.pdf * cosPhi / Dist / Dist;
                float pdfAreaLight = length(radiance) / areaSum;

                float pdfAreaBRDF2 = pdfAreaBRDF * pdfAreaBRDF;
                float pdfAreaLight2 = pdfAreaLight * pdfAreaLight;
       
                prd_radiance.radiance += radiance * pdfAreaBRDF2 / fmaxf(pdfAreaBRDF2 + pdfAreaLight2, 1e-14) * prd_radiance.attenuation;
            }
        }
    }
    prd_radiance.done = true;
}


RT_PROGRAM void any_hit_shadow()
{
    prd_shadow.inShadow = true;
    rtTerminateRay();
}
