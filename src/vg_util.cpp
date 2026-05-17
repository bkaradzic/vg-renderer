#include "vg_util.h"
#include <vg/vg.h>
#include <bx/bx.h>

#include <bx/simd_t.h>

BX_PRAGMA_DIAGNOSTIC_IGNORED_GCC("-Wimplicit-fallthrough=0")

namespace vgutil
{
bool invertMatrix3(const float* __restrict t, float* __restrict inv)
{
	// nvgTransformInverse
	double invdet, det = (double)t[0] * t[3] - (double)t[2] * t[1];
	if (det > -1e-6 && det < 1e-6) {
		inv[0] = inv[2] = 1.0f;
		inv[1] = inv[3] = inv[4] = inv[5] = 0.0f;
		return false;
	}

	invdet = 1.0 / det;
	inv[0] = (float)(t[3] * invdet);
	inv[2] = (float)(-t[2] * invdet);
	inv[4] = (float)(((double)t[2] * t[5] - (double)t[3] * t[4]) * invdet);
	inv[1] = (float)(-t[1] * invdet);
	inv[3] = (float)(t[0] * invdet);
	inv[5] = (float)(((double)t[1] * t[4] - (double)t[0] * t[5]) * invdet);

	return true;
}

#if VG_CONFIG_ENABLE_SIMD
void memset32(void* __restrict dst, uint32_t n, const void* __restrict src)
{
	const bx::simd128_t s128 = bx::simd128_splat(*(const float*)src);
	float* d = (float*)dst;

	uint32_t iter = n >> 4;
	while (iter-- > 0) {
		bx::simd128_stu(d +  0, s128);
		bx::simd128_stu(d +  4, s128);
		bx::simd128_stu(d +  8, s128);
		bx::simd128_stu(d + 12, s128);
		d += 16;
	}

	uint32_t rem = n & 15;
	if (rem >= 8) {
		bx::simd128_stu(d + 0, s128);
		bx::simd128_stu(d + 4, s128);
		d += 8;
		rem -= 8;
	}

	if (rem >= 4) {
		bx::simd128_stu(d, s128);
		d += 4;
		rem -= 4;
	}

	switch (rem) {
	case 3: *d++ = *(const float*)src;
	case 2: *d++ = *(const float*)src;
	case 1: *d   = *(const float*)src;
	}
}

void memset64(void* __restrict dst, uint32_t n64, const void* __restrict src)
{
	const float* s = (const float*)src;
	const bx::simd128_t s128 = bx::simd128_ld(s[0], s[1], s[0], s[1]);
	float* d = (float*)dst;

	uint32_t iter = n64 >> 3; // 8 64-bit values per iteration (== 16 floats / iter)
	while (iter-- > 0) {
		bx::simd128_stu(d +  0, s128);
		bx::simd128_stu(d +  4, s128);
		bx::simd128_stu(d +  8, s128);
		bx::simd128_stu(d + 12, s128);
		d += 16;
	}

	uint32_t rem = n64 & 7;
	if (rem >= 4) {
		bx::simd128_stu(d + 0, s128);
		bx::simd128_stu(d + 4, s128);
		d += 8;
		rem -= 4;
	}

	if (rem >= 2) {
		bx::simd128_stu(d, s128);
		d += 4;
		rem -= 2;
	}

	if (rem) {
		d[0] = s[0];
		d[1] = s[1];
	}
}

void memset128(void* __restrict dst, uint32_t n128, const void* __restrict src)
{
	const bx::simd128_t s128 = bx::simd128_ldu(src);
	float* d = (float*)dst;

	uint32_t iter = n128 >> 2; // 4 128-bit values per iteration (== 16 floats / iter)
	while (iter-- > 0) {
		bx::simd128_stu(d +  0, s128);
		bx::simd128_stu(d +  4, s128);
		bx::simd128_stu(d +  8, s128);
		bx::simd128_stu(d + 12, s128);
		d += 16;
	}

	uint32_t rem = n128 & 3;
	if (rem >= 2) {
		bx::simd128_stu(d + 0, s128);
		bx::simd128_stu(d + 4, s128);
		d += 8;
		rem -= 2;
	}

	if (rem) {
		bx::simd128_stu(d, s128);
	}
}

void batchTransformPositions(const float* __restrict src, uint32_t n, float* __restrict dst, const float* __restrict mtx)
{
	const bx::simd128_t mtx0 = bx::simd128_splat(mtx[0]);
	const bx::simd128_t mtx1 = bx::simd128_splat(mtx[1]);
	const bx::simd128_t mtx2 = bx::simd128_splat(mtx[2]);
	const bx::simd128_t mtx3 = bx::simd128_splat(mtx[3]);
	const bx::simd128_t mtx4 = bx::simd128_splat(mtx[4]);
	const bx::simd128_t mtx5 = bx::simd128_splat(mtx[5]);

	const uint32_t iter = n >> 3;
	for (uint32_t i = 0; i < iter; ++i) {
		// x' = m[0] * x + m[2] * y + m[4];
		// y' = m[1] * x + m[3] * y + m[5];
		const bx::simd128_t xy01 = bx::simd128_ldu(src +  0); // { x0, y0, x1, y1 }
		const bx::simd128_t xy23 = bx::simd128_ldu(src +  4); // { x2, y2, x3, y3 }
		const bx::simd128_t xy45 = bx::simd128_ldu(src +  8); // { x4, y4, x5, y5 }
		const bx::simd128_t xy67 = bx::simd128_ldu(src + 12); // { x6, y6, x7, y7 }

		const bx::simd128_t x0123 = bx::simd128_x32_shuf_xzAC(xy01, xy23); // { x0, x1, x2, x3 }
		const bx::simd128_t y0123 = bx::simd128_x32_shuf_ywBD(xy01, xy23); // { y0, y1, y2, y3 }
		const bx::simd128_t x4567 = bx::simd128_x32_shuf_xzAC(xy45, xy67); // { x4, x5, x6, x7 }
		const bx::simd128_t y4567 = bx::simd128_x32_shuf_ywBD(xy45, xy67); // { y4, y5, y6, y7 }

		const bx::simd128_t x0123_m0    = bx::simd128_f32_mul(x0123, mtx0);
		const bx::simd128_t x0123_m1    = bx::simd128_f32_mul(x0123, mtx1);
		const bx::simd128_t y0123_m2    = bx::simd128_f32_mul(y0123, mtx2);
		const bx::simd128_t y0123_m3    = bx::simd128_f32_mul(y0123, mtx3);
		const bx::simd128_t x4567_m0    = bx::simd128_f32_mul(x4567, mtx0);
		const bx::simd128_t x4567_m1    = bx::simd128_f32_mul(x4567, mtx1);
		const bx::simd128_t y4567_m2    = bx::simd128_f32_mul(y4567, mtx2);
		const bx::simd128_t y4567_m3    = bx::simd128_f32_mul(y4567, mtx3);

		const bx::simd128_t x0123_m0_m4 = bx::simd128_f32_add(x0123_m0, mtx4);
		const bx::simd128_t x0123_m1_m5 = bx::simd128_f32_add(x0123_m1, mtx5);
		const bx::simd128_t x4567_m0_m4 = bx::simd128_f32_add(x4567_m0, mtx4);
		const bx::simd128_t x4567_m1_m5 = bx::simd128_f32_add(x4567_m1, mtx5);

		const bx::simd128_t resx0123    = bx::simd128_f32_add(x0123_m0_m4, y0123_m2); // { xi * m[0] + yi * m[2] + m[4] }
		const bx::simd128_t resy0123    = bx::simd128_f32_add(x0123_m1_m5, y0123_m3); // { xi * m[1] + yi * m[3] + m[5] }
		const bx::simd128_t resx4567    = bx::simd128_f32_add(x4567_m0_m4, y4567_m2);
		const bx::simd128_t resy4567    = bx::simd128_f32_add(x4567_m1_m5, y4567_m3);

		const bx::simd128_t rxy01_lo    = bx::simd128_x32_shuf_xyAB(resx0123, resy0123); // { rx0, rx1, ry0, ry1 }
		const bx::simd128_t rxy23_hi    = bx::simd128_x32_shuf_zwCD(resx0123, resy0123); // { rx2, rx3, ry2, ry3 }
		const bx::simd128_t rxy45_lo    = bx::simd128_x32_shuf_xyAB(resx4567, resy4567); // { rx4, rx5, ry4, ry5 }
		const bx::simd128_t rxy67_hi    = bx::simd128_x32_shuf_zwCD(resx4567, resy4567); // { rx6, rx7, ry6, ry7 }

		const bx::simd128_t resxy01     = bx::simd128_x32_swiz_xzyw(rxy01_lo); // { rx0, ry0, rx1, ry1 }
		const bx::simd128_t resxy23     = bx::simd128_x32_swiz_xzyw(rxy23_hi); // { rx2, ry2, rx3, ry3 }
		const bx::simd128_t resxy45     = bx::simd128_x32_swiz_xzyw(rxy45_lo); // { rx4, ry4, rx5, ry5 }
		const bx::simd128_t resxy67     = bx::simd128_x32_swiz_xzyw(rxy67_hi); // { rx6, ry6, rx7, ry7 }

		bx::simd128_stu(dst +  0, resxy01);
		bx::simd128_stu(dst +  4, resxy23);
		bx::simd128_stu(dst +  8, resxy45);
		bx::simd128_stu(dst + 12, resxy67);

		src += 16;
		dst += 16;
	}

	uint32_t rem = n & 7;
	if (rem >= 4) {
		const bx::simd128_t xy01 = bx::simd128_ldu(src + 0);
		const bx::simd128_t xy23 = bx::simd128_ldu(src + 4);

		const bx::simd128_t x0123 = bx::simd128_x32_shuf_xzAC(xy01, xy23);
		const bx::simd128_t y0123 = bx::simd128_x32_shuf_ywBD(xy01, xy23);

		const bx::simd128_t x0123_m0 = bx::simd128_f32_mul(x0123, mtx0);
		const bx::simd128_t x0123_m1 = bx::simd128_f32_mul(x0123, mtx1);
		const bx::simd128_t y0123_m2 = bx::simd128_f32_mul(y0123, mtx2);
		const bx::simd128_t y0123_m3 = bx::simd128_f32_mul(y0123, mtx3);

		const bx::simd128_t x0123_m0_m4 = bx::simd128_f32_add(x0123_m0, mtx4);
		const bx::simd128_t x0123_m1_m5 = bx::simd128_f32_add(x0123_m1, mtx5);

		const bx::simd128_t resx0123 = bx::simd128_f32_add(x0123_m0_m4, y0123_m2);
		const bx::simd128_t resy0123 = bx::simd128_f32_add(x0123_m1_m5, y0123_m3);

		const bx::simd128_t rxy01_lo = bx::simd128_x32_shuf_xyAB(resx0123, resy0123);
		const bx::simd128_t rxy23_hi = bx::simd128_x32_shuf_zwCD(resx0123, resy0123);

		const bx::simd128_t resxy01  = bx::simd128_x32_swiz_xzyw(rxy01_lo);
		const bx::simd128_t resxy23  = bx::simd128_x32_swiz_xzyw(rxy23_hi);

		bx::simd128_stu(dst + 0, resxy01);
		bx::simd128_stu(dst + 4, resxy23);

		src += 8;
		dst += 8;

		rem -= 4;
	}

	switch (rem) {
	case 3:
		dst[0] = mtx[0] * src[0] + mtx[2] * src[1] + mtx[4];
		dst[1] = mtx[1] * src[0] + mtx[3] * src[1] + mtx[5];
		src += 2;
		dst += 2;
	case 2:
		dst[0] = mtx[0] * src[0] + mtx[2] * src[1] + mtx[4];
		dst[1] = mtx[1] * src[0] + mtx[3] * src[1] + mtx[5];
		src += 2;
		dst += 2;
	case 1:
		dst[0] = mtx[0] * src[0] + mtx[2] * src[1] + mtx[4];
		dst[1] = mtx[1] * src[0] + mtx[3] * src[1] + mtx[5];
	}
}
#else
void memset32(void* __restrict dst, uint32_t n, const void* __restrict src)
{
	const uint32_t s = *(const uint32_t*)src;
	uint32_t* d = (uint32_t*)dst;
	while (n-- > 0) {
		*d++ = s;
	}
}

void memset64(void* __restrict dst, uint32_t n64, const void* __restrict src)
{
	const uint32_t s0 = *((const uint32_t*)src + 0);
	const uint32_t s1 = *((const uint32_t*)src + 1);
	uint32_t* d = (uint32_t*)dst;
	while (n64-- > 0) {
		d[0] = s0;
		d[1] = s1;
		d += 2;
	}
}

void memset128(void* __restrict dst, uint32_t n128, const void* __restrict src)
{
	const uint32_t s0 = *((const uint32_t*)src + 0);
	const uint32_t s1 = *((const uint32_t*)src + 1);
	const uint32_t s2 = *((const uint32_t*)src + 2);
	const uint32_t s3 = *((const uint32_t*)src + 3);
	uint32_t* d = (uint32_t*)dst;
	while (n128-- > 0) {
		d[0] = s0;
		d[1] = s1;
		d[2] = s2;
		d[3] = s3;
		d += 4;
	}
}

void batchTransformPositions(const float* __restrict v, uint32_t n, float* __restrict p, const float* __restrict mtx)
{
	for (uint32_t i = 0; i < n; ++i) {
		const uint32_t id = i << 1;
		transformPos2D(v[id], v[id + 1], mtx, &p[id]);
	}
}
#endif

void genQuadIndices_unaligned(uint16_t* dst, uint32_t n, uint16_t firstVertexID)
{
#if VG_CONFIG_ENABLE_SIMD
	BX_ALIGN_DECL(16, static const uint16_t delta[]) = {
		0, 1, 2, 0, 2, 3,
		4, 5, 6, 4, 6, 7,
		8, 9, 10, 8, 10, 11,
		12, 13, 14, 12, 14, 15
	};

	const bx::simd128_t xmm_delta0 = bx::simd128_ld(&delta[ 0]);
	const bx::simd128_t xmm_delta1 = bx::simd128_ld(&delta[ 8]);
	const bx::simd128_t xmm_delta2 = bx::simd128_ld(&delta[16]);

	const uint32_t numIter = n >> 2; // 4 quads per iteration
	for (uint32_t i = 0; i < numIter; ++i) {
		const bx::simd128_t id  = bx::simd128_splat((int16_t)firstVertexID);

		const bx::simd128_t id0 = bx::simd128_i16_add(id, xmm_delta0);
		const bx::simd128_t id1 = bx::simd128_i16_add(id, xmm_delta1);
		const bx::simd128_t id2 = bx::simd128_i16_add(id, xmm_delta2);
		bx::simd128_stu(dst +  0, id0);
		bx::simd128_stu(dst +  8, id1);
		bx::simd128_stu(dst + 16, id2);

		dst += 24;
		firstVertexID += 16;
	}

	uint32_t rem = n & 3;
	switch (rem) {
	case 3:
		dst[0] = firstVertexID; dst[1] = firstVertexID + 1; dst[2] = firstVertexID + 2;
		dst[3] = firstVertexID; dst[4] = firstVertexID + 2; dst[5] = firstVertexID + 3;
		dst += 6;
		firstVertexID += 4;
	case 2:
		dst[0] = firstVertexID; dst[1] = firstVertexID + 1; dst[2] = firstVertexID + 2;
		dst[3] = firstVertexID; dst[4] = firstVertexID + 2; dst[5] = firstVertexID + 3;
		dst += 6;
		firstVertexID += 4;
	case 1:
		dst[0] = firstVertexID; dst[1] = firstVertexID + 1; dst[2] = firstVertexID + 2;
		dst[3] = firstVertexID; dst[4] = firstVertexID + 2; dst[5] = firstVertexID + 3;
		dst += 6;
		firstVertexID += 4;
	}
#else
	while (n-- > 0) {
		dst[0] = firstVertexID; dst[1] = firstVertexID + 1; dst[2] = firstVertexID + 2;
		dst[3] = firstVertexID; dst[4] = firstVertexID + 2; dst[5] = firstVertexID + 3;
		dst += 6;
		firstVertexID += 4;
	}
#endif
}

void batchTransformTextQuads(const float* __restrict quads, uint32_t n, const float* __restrict mtx, float* __restrict transformedVertices)
{
#if VG_CONFIG_ENABLE_SIMD
	const bx::simd128_t mtx0 = bx::simd128_splat(mtx[0]);
	const bx::simd128_t mtx1 = bx::simd128_splat(mtx[1]);
	const bx::simd128_t mtx2 = bx::simd128_splat(mtx[2]);
	const bx::simd128_t mtx3 = bx::simd128_splat(mtx[3]);
	const bx::simd128_t mtx4 = bx::simd128_splat(mtx[4]);
	const bx::simd128_t mtx5 = bx::simd128_splat(mtx[5]);

	const uint32_t iter = n >> 1; // 2 quads per iteration
	for (uint32_t i = 0; i < iter; ++i) {
		const bx::simd128_t q0 = bx::simd128_ld(quads);     // (x0, y0, x1, y1)
		const bx::simd128_t q1 = bx::simd128_ld(quads + 8); // (x2, y2, x3, y3)

		const bx::simd128_t q0_xzxz = bx::simd128_x32_swiz_xzxz(q0);
		const bx::simd128_t q1_xzxz = bx::simd128_x32_swiz_xzxz(q1);
		const bx::simd128_t q0_ywyw = bx::simd128_x32_swiz_ywyw(q0);
		const bx::simd128_t q1_ywyw = bx::simd128_x32_swiz_ywyw(q1);
		const bx::simd128_t x0123  = bx::simd128_x32_shuf_xyAB(q0_xzxz, q1_xzxz); // (x0, x1, x2, x3)
		const bx::simd128_t y0123  = bx::simd128_x32_shuf_xyAB(q0_ywyw, q1_ywyw); // (y0, y1, y2, y3)
		const bx::simd128_t x0123_m0 = bx::simd128_f32_mul(x0123, mtx0); // (x0, x1, x2, x3) * mtx[0]
		const bx::simd128_t x0123_m1 = bx::simd128_f32_mul(x0123, mtx1); // (x0, x1, x2, x3) * mtx[1]
		const bx::simd128_t y0123_m2 = bx::simd128_f32_mul(y0123, mtx2); // (y0, y1, y2, y3) * mtx[2]
		const bx::simd128_t y0123_m3 = bx::simd128_f32_mul(y0123, mtx3); // (y0, y1, y2, y3) * mtx[3]

		// v0.x = x0_m0 + y0_m2 + m4
		// v1.x = x1_m0 + y0_m2 + m4
		// v2.x = x1_m0 + y1_m2 + m4
		// v3.x = x0_m0 + y1_m2 + m4
		// v0.y = x0_m1 + y0_m3 + m5
		// v1.y = x1_m1 + y0_m3 + m5
		// v2.y = x1_m1 + y1_m3 + m5
		// v3.y = x0_m1 + y1_m3 + m5
		const bx::simd128_t x0110_m0 = bx::simd128_x32_swiz_xyyx(x0123_m0);
		const bx::simd128_t x0110_m1 = bx::simd128_x32_swiz_xyyx(x0123_m1);
		const bx::simd128_t y0011_m2 = bx::simd128_x32_swiz_xxyy(y0123_m2);
		const bx::simd128_t y0011_m3 = bx::simd128_x32_swiz_xxyy(y0123_m3);

		const bx::simd128_t y0011_m2_m4 = bx::simd128_f32_add(y0011_m2, mtx4);
		const bx::simd128_t y0011_m3_m5 = bx::simd128_f32_add(y0011_m3, mtx5);
		const bx::simd128_t v0123_x    = bx::simd128_f32_add(x0110_m0, y0011_m2_m4);
		const bx::simd128_t v0123_y    = bx::simd128_f32_add(x0110_m1, y0011_m3_m5);

		const bx::simd128_t v01_lo = bx::simd128_x32_shuf_xyAB(v0123_x, v0123_y);
		const bx::simd128_t v23_hi = bx::simd128_x32_shuf_zwCD(v0123_x, v0123_y);
		const bx::simd128_t v01    = bx::simd128_x32_swiz_xzyw(v01_lo);
		const bx::simd128_t v23    = bx::simd128_x32_swiz_xzyw(v23_hi);

		bx::simd128_st(transformedVertices, v01);
		bx::simd128_st(transformedVertices + 4, v23);

		// v4.x = x2_m0 + y2_m2 + m4
		// v5.x = x3_m0 + y2_m2 + m4
		// v6.x = x3_m0 + y3_m2 + m4
		// v7.x = x2_m0 + y3_m2 + m4
		// v4.y = x2_m1 + y2_m3 + m5
		// v5.y = x3_m1 + y2_m3 + m5
		// v6.y = x3_m1 + y3_m3 + m5
		// v7.y = x2_m1 + y3_m3 + m5
		const bx::simd128_t x2332_m0 = bx::simd128_x32_swiz_zwwz(x0123_m0);
		const bx::simd128_t x2332_m1 = bx::simd128_x32_swiz_zwwz(x0123_m1);
		const bx::simd128_t y2233_m2 = bx::simd128_x32_swiz_zzww(y0123_m2);
		const bx::simd128_t y2233_m3 = bx::simd128_x32_swiz_zzww(y0123_m3);

		const bx::simd128_t y2233_m2_m4 = bx::simd128_f32_add(y2233_m2, mtx4);
		const bx::simd128_t y2233_m3_m5 = bx::simd128_f32_add(y2233_m3, mtx5);
		const bx::simd128_t v4567_x    = bx::simd128_f32_add(x2332_m0, y2233_m2_m4);
		const bx::simd128_t v4567_y    = bx::simd128_f32_add(x2332_m1, y2233_m3_m5);

		const bx::simd128_t v45_lo = bx::simd128_x32_shuf_xyAB(v4567_x, v4567_y);
		const bx::simd128_t v67_hi = bx::simd128_x32_shuf_zwCD(v4567_x, v4567_y);
		const bx::simd128_t v45    = bx::simd128_x32_swiz_xzyw(v45_lo);
		const bx::simd128_t v67    = bx::simd128_x32_swiz_xzyw(v67_hi);

		bx::simd128_st(transformedVertices + 8, v45);
		bx::simd128_st(transformedVertices + 12, v67);

		quads += 16;
		transformedVertices += 16;
	}

	const uint32_t rem = n & 1;
	if (rem) {
		const bx::simd128_t q0 = bx::simd128_ld(quads);

		const bx::simd128_t x0101 = bx::simd128_x32_swiz_xzxz(q0); // (x0, x1, x0, x1)
		const bx::simd128_t y0101 = bx::simd128_x32_swiz_ywyw(q0); // (y0, y1, y0, y1)
		const bx::simd128_t x0101_m0 = bx::simd128_f32_mul(x0101, mtx0); // (x0, x1, x0, x1) * mtx[0]
		const bx::simd128_t x0101_m1 = bx::simd128_f32_mul(x0101, mtx1); // (x0, x1, x0, x1) * mtx[1]
		const bx::simd128_t y0101_m2 = bx::simd128_f32_mul(y0101, mtx2); // (y0, y1, y0, y1) * mtx[2]
		const bx::simd128_t y0101_m3 = bx::simd128_f32_mul(y0101, mtx3); // (y0, y1, y0, y1) * mtx[3]

		// v0.x = x0_m0 + y0_m2 + m4
		// v1.x = x1_m0 + y0_m2 + m4
		// v2.x = x1_m0 + y1_m2 + m4
		// v3.x = x0_m0 + y1_m2 + m4
		// v0.y = x0_m1 + y0_m3 + m5
		// v1.y = x1_m1 + y0_m3 + m5
		// v2.y = x1_m1 + y1_m3 + m5
		// v3.y = x0_m1 + y1_m3 + m5
		const bx::simd128_t x0110_m0 = bx::simd128_x32_swiz_xyyx(x0101_m0);
		const bx::simd128_t x0110_m1 = bx::simd128_x32_swiz_xyyx(x0101_m1);
		const bx::simd128_t y0011_m2 = bx::simd128_x32_swiz_xxyy(y0101_m2);
		const bx::simd128_t y0011_m3 = bx::simd128_x32_swiz_xxyy(y0101_m3);

		const bx::simd128_t y0011_m2_m4 = bx::simd128_f32_add(y0011_m2, mtx4);
		const bx::simd128_t y0011_m3_m5 = bx::simd128_f32_add(y0011_m3, mtx5);
		const bx::simd128_t v0123_x    = bx::simd128_f32_add(x0110_m0, y0011_m2_m4);
		const bx::simd128_t v0123_y    = bx::simd128_f32_add(x0110_m1, y0011_m3_m5);

		const bx::simd128_t v01_lo = bx::simd128_x32_shuf_xyAB(v0123_x, v0123_y);
		const bx::simd128_t v23_hi = bx::simd128_x32_shuf_zwCD(v0123_x, v0123_y);
		const bx::simd128_t v01    = bx::simd128_x32_swiz_xzyw(v01_lo);
		const bx::simd128_t v23    = bx::simd128_x32_swiz_xzyw(v23_hi);

		bx::simd128_st(transformedVertices, v01);
		bx::simd128_st(transformedVertices + 4, v23);
	}
#else
	for (uint32_t i = 0; i < n; ++i) {
		const float* q = &quads[i * 8];
		const uint32_t s = i << 3;
		transformPos2D(q[0], q[1], mtx, &transformedVertices[s + 0]);
		transformPos2D(q[2], q[1], mtx, &transformedVertices[s + 2]);
		transformPos2D(q[2], q[3], mtx, &transformedVertices[s + 4]);
		transformPos2D(q[0], q[3], mtx, &transformedVertices[s + 6]);
	}
#endif
}

void batchTransformDrawIndices(const uint16_t* __restrict src, uint32_t n, uint16_t* __restrict dst, uint16_t delta)
{
	if (delta == 0) {
		bx::memCopy(dst, src, sizeof(uint16_t) * n);
		return;
	}

#if VG_CONFIG_ENABLE_SIMD
	const bx::simd128_t xmm_delta = bx::simd128_splat((int16_t)delta);

	const uint32_t iter32 = n >> 5;
	for (uint32_t i = 0; i < iter32; ++i) {
		const bx::simd128_t s0 = bx::simd128_ldu(src);
		const bx::simd128_t s1 = bx::simd128_ldu(src +  8);
		const bx::simd128_t s2 = bx::simd128_ldu(src + 16);
		const bx::simd128_t s3 = bx::simd128_ldu(src + 24);

		const bx::simd128_t d0 = bx::simd128_i16_add(s0, xmm_delta);
		const bx::simd128_t d1 = bx::simd128_i16_add(s1, xmm_delta);
		const bx::simd128_t d2 = bx::simd128_i16_add(s2, xmm_delta);
		const bx::simd128_t d3 = bx::simd128_i16_add(s3, xmm_delta);

		// NOTE: Proper alignment of dst buffer isn't guaranteed because it's part of the global IndexBuffer.
		bx::simd128_stu(dst,       d0);
		bx::simd128_stu(dst +  8,  d1);
		bx::simd128_stu(dst + 16,  d2);
		bx::simd128_stu(dst + 24,  d3);

		src += 32;
		dst += 32;
	}

	uint32_t rem = n & 31;
	if (rem >= 16) {
		const bx::simd128_t s0 = bx::simd128_ldu(src);
		const bx::simd128_t s1 = bx::simd128_ldu(src + 8);

		const bx::simd128_t d0 = bx::simd128_i16_add(s0, xmm_delta);
		const bx::simd128_t d1 = bx::simd128_i16_add(s1, xmm_delta);

		bx::simd128_stu(dst,     d0);
		bx::simd128_stu(dst + 8, d1);

		src += 16;
		dst += 16;
		rem -= 16;
	}

	if (rem >= 8) {
		const bx::simd128_t s0 = bx::simd128_ldu(src);
		const bx::simd128_t d0 = bx::simd128_i16_add(s0, xmm_delta);
		bx::simd128_stu(dst, d0);

		src += 8;
		dst += 8;
		rem -= 8;
	}

	switch (rem) {
	case 7: *dst++ = *src++ + delta;
	case 6: *dst++ = *src++ + delta;
	case 5: *dst++ = *src++ + delta;
	case 4: *dst++ = *src++ + delta;
	case 3: *dst++ = *src++ + delta;
	case 2: *dst++ = *src++ + delta;
	case 1: *dst   = *src   + delta;
	}
#else
	for (uint32_t i = 0; i < n; ++i) {
		*dst++ = *src + delta;
		src++;
	}
#endif
}

void convertA8_to_RGBA8(uint32_t* rgba, const uint8_t* a8, uint32_t w, uint32_t h, uint32_t rgbColor)
{
	const uint32_t rgb0 = rgbColor & 0x00FFFFFF;

	int numPixels = w * h;
	for (int i = 0; i < numPixels; ++i) {
		*rgba++ = rgb0 | (((uint32_t)* a8) << 24);
		++a8;
	}
}

PoolAllocator::PoolAllocator(uint32_t itemSize, uint32_t numItemsPerChunk, bx::AllocatorI* parentAllocator)
	: m_ParentAllocator(parentAllocator)
	, m_FirstChunk(nullptr)
	, m_FirstFreeSlotPtr(nullptr)
	, m_ItemSize(itemSize)
	, m_NumItemsPerChunk(numItemsPerChunk)
	, m_Flags(0)
{
}

PoolAllocator::~PoolAllocator()
{
	PoolAllocator::Chunk* chunk = m_FirstChunk;
	while (chunk->m_Next) {
		PoolAllocator::Chunk* nextChunk = chunk->m_Next;
		bx::free(m_ParentAllocator, chunk);
		chunk = nextChunk;
	}
}

void* PoolAllocator::realloc(void* _ptr, size_t _size, size_t _align, const char* _filePath, uint32_t _line)
{
	BX_UNUSED(_align, _filePath, _line);
	VG_CHECK(align <= 8, "Pool allocators do not support alignment.", 0);

	if (_ptr != nullptr) {
		// Realloc or free?
		if (_size != 0) {
			VG_CHECK(false, "Pool allocators do not support reallocations.", 0);
			return nullptr;
		}

		// Free item
		FreeListItem* freeSlot = (FreeListItem*)_ptr;
		freeSlot->m_Next = m_FirstFreeSlotPtr;
		m_FirstFreeSlotPtr = freeSlot;

		return nullptr;
	}

	// Alloc
	if ((uint32_t)_size != m_ItemSize) {
		VG_CHECK(false, "Pool allocators cannot allocate arbitrary amounts of memory.", 0);
		return nullptr;
	}

	FreeListItem* freeSlot = m_FirstFreeSlotPtr;
	if (freeSlot == nullptr) {
#if 0
		if ((m_Flags & POOL_ALLOCATOR_FLAGS_ALLOW_RESIZE) == 0) {
			return nullptr;
		}
#endif

		const size_t totalMem = 0
			+ sizeof(PoolAllocator::Chunk)
			+ (size_t)m_ItemSize * (size_t)m_NumItemsPerChunk
			;

		uint8_t* newBuffer = (uint8_t*)bx::alloc(m_ParentAllocator, totalMem);
		if (!newBuffer) {
			// Couldn't allocate new chunk. Allocation failed.
			return nullptr;
		}

		uint8_t* newBufferPtr = newBuffer;
		PoolAllocator::Chunk* poolAllocatorChunk = (PoolAllocator::Chunk*)newBufferPtr;
		newBufferPtr += sizeof(PoolAllocator::Chunk);

		poolAllocatorChunk->m_Buffer = newBufferPtr;
		poolAllocatorChunk->m_Next = m_FirstChunk;
		m_FirstChunk = poolAllocatorChunk;

		// Initialize free list
		const uint32_t itemSize = m_ItemSize;
		const uint32_t numItemsPerChunk = m_NumItemsPerChunk;
		for (uint64_t i = 0; i < numItemsPerChunk - 1; ++i) {
			FreeListItem* fli = (FreeListItem*)(newBufferPtr + i * itemSize);
			fli->m_Next = (FreeListItem*)(newBufferPtr + (i + 1) * itemSize);
		}

		// Last item
		{
			FreeListItem* fli = (FreeListItem*)(newBufferPtr + (numItemsPerChunk - 1) * itemSize);
			fli->m_Next = nullptr;
		}

		m_FirstFreeSlotPtr = (FreeListItem*)poolAllocatorChunk->m_Buffer;

		freeSlot = m_FirstFreeSlotPtr;
	}

	m_FirstFreeSlotPtr = freeSlot->m_Next;

	return freeSlot;
}
}
