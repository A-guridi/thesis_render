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

#pragma once

#include <optixu/optixu_vector_types.h>
// added structs of polarized light
//struct representing a 4x4 matrix for the Mueller Data
struct float4x4{
    float4 r0;
    float4 r1;
    float4 r2;
    float4 r3;
};
struct MuellerData
{
    // they are 4x4 matrices
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

struct PerRayData_radiance
{
  int depth;
  unsigned int seed;

  // shading state
  bool done;
  float3 attenuation;
  float3 radiance;
  float3 origin;
  float3 direction;

 float pdf;

 StokesLight lightData;

};

struct PerRayData_shadow
{
    bool inShadow;
};

//operators for 4x4 matrices

inline __host__ __device__ float4x4 operator+(float4x4 a, float4x4 b)
{
    float4x4 res;
    res.r0=a.r0+b.r0;
    res.r1=a.r1+b.r1;
    res.r2=a.r2+b.r2;
    res.r3=a.r3+b.r3;
    return res;
}
//operator + for 4x4 and consant
inline __host__ __device__ float4x4 operator+(float4x4 a, float b)
{
    float4x4 res;
    res.r0=a.r0+b;
    res.r1=a.r1+b;
    res.r2=a.r2+b;
    res.r3=a.r3+b;
    return res;
}
//operator * for 4x4 and constant
inline __host__ __device__ float4x4 operator*(float4x4 a, float b)
{
    float4x4 res;
    res.r0=a.r0*b;
    res.r1=a.r1*b;
    res.r2=a.r2*b;
    res.r3=a.r3*b;
    return res;
}
inline __host__ __device__ void operator*=(float4x4 &a, float b)
{
    a.r0*=b;
    a.r1*=b;
    a.r2*=b;
    a.r3*=b;
}
