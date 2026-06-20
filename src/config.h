#ifndef VG_CONFIG_H
#define VG_CONFIG_H

// Internal build-time configuration and debug helpers. This header is private to
// the vg-renderer implementation (src/) and is NOT part of the public API. The
// public uv_t toggle (VG_CONFIG_UV_INT16) lives in <vg/vg.h> because it affects a
// public type.

#ifndef VG_CONFIG_DEBUG
#	define VG_CONFIG_DEBUG 0
#endif

#ifndef VG_CONFIG_ENABLE_SHAPE_CACHING
#	define VG_CONFIG_ENABLE_SHAPE_CACHING 1
#endif

#ifndef VG_CONFIG_ENABLE_SIMD
#	define VG_CONFIG_ENABLE_SIMD 1
#endif

#ifndef VG_CONFIG_FORCE_AA_OFF
#	define VG_CONFIG_FORCE_AA_OFF 0
#endif

#ifndef VG_CONFIG_LIBTESS2_SCRATCH_BUFFER
#	define VG_CONFIG_LIBTESS2_SCRATCH_BUFFER (4 * 1024 * 1024) // Set to 0 to let libtess2 use malloc/free
#endif

// If set to 1, submitCommandList() calls pustState()/popState() and resetClip() before and after
// executing the commands. Otherwise, the state produced by the command list will affect the global
// state after the execution of the commands.
#ifndef VG_CONFIG_COMMAND_LIST_PRESERVE_STATE
#	define VG_CONFIG_COMMAND_LIST_PRESERVE_STATE 0
#endif

#define VG_EPSILON 1e-5f

#if VG_CONFIG_DEBUG
#include <bx/debug.h>

#define VG_TRACE(_format, ...) \
	do { \
		bx::debugPrintf(BX_FILE_LINE_LITERAL "vg " _format "\n", ##__VA_ARGS__); \
	} while(0)

#define VG_WARN(_condition, _format, ...) \
	do { \
		if (!(_condition) ) { \
			VG_TRACE(BX_FILE_LINE_LITERAL _format, ##__VA_ARGS__); \
		} \
	} while(0)

#define VG_CHECK(_condition, _format, ...) \
	do { \
		if (!(_condition) ) { \
			VG_TRACE(BX_FILE_LINE_LITERAL _format, ##__VA_ARGS__); \
			bx::debugBreak(); \
		} \
	} while(0)
#else
#define VG_TRACE(_format, ...)
#define VG_WARN(_condition, _format, ...)
#define VG_CHECK(_condition, _format, ...)
#endif

#endif // VG_CONFIG_H
