/*
 * Copyright 2017-2026 Jim Drygiannakis. All rights reserved.
 * License: https://github.com/jdryg/vg-renderer/blob/master/LICENSE
 */

// TODO:
// - More than 254 clip regions: Either use another view (extra parameter in createContext)
// or draw a fullscreen quad to reset the stencil buffer to 0.
// - Recycle the memory of cached meshes so resetting a cached mesh is faster.
// - Find a way to move stroker operations into separate functions (i.e. all strokePath
// functions differ only on the createDrawCommand_XXX() call; strokerXXX calls are the same
// and the code is duplicated).
// - Allow strokes and fills with gradients and image patterns to be used as clip masks (might
// be useful if the same command list is used both inside and outside a beginClip()/endClip()
// block)
#include <vg/vg.h>
#include "config.h"
#include "path.h"
#include "stroker.h"
#include "vg_util.h"
#include "font_system.h"
#include <bx/allocator.h>
#include <bx/mutex.h>
#include <bx/handlealloc.h>
#include <bx/string.h>
#include <bgfx/bgfx.h>
#include <bgfx/embedded_shader.h>

// Shaders
#include "shaders/vs_textured.bin.h"
#include "shaders/fs_textured.bin.h"
#include "shaders/vs_color_gradient.bin.h"
#include "shaders/fs_color_gradient.bin.h"
#include "shaders/vs_image_pattern.bin.h"
#include "shaders/fs_image_pattern.bin.h"
#include "shaders/vs_stencil.bin.h"
#include "shaders/fs_stencil.bin.h"

BX_PRAGMA_DIAGNOSTIC_IGNORED_MSVC(4706) // assignment within conditional expression

#define VG_CONFIG_MIN_FONT_SCALE                 0.1f
#define VG_CONFIG_MAX_FONT_SCALE                 4.0f
#define VG_CONFIG_MIN_FONT_ATLAS_SIZE            512
#define VG_CONFIG_COMMAND_LIST_CACHE_STACK_SIZE  32
#define VG_CONFIG_COMMAND_LIST_ALIGNMENT         16

namespace vg
{
static const bgfx::EmbeddedShader s_EmbeddedShaders[] =
{
	BGFX_EMBEDDED_SHADER(vs_textured),
	BGFX_EMBEDDED_SHADER(fs_textured),
	BGFX_EMBEDDED_SHADER(vs_color_gradient),
	BGFX_EMBEDDED_SHADER(fs_color_gradient),
	BGFX_EMBEDDED_SHADER(vs_image_pattern),
	BGFX_EMBEDDED_SHADER(fs_image_pattern),
	BGFX_EMBEDDED_SHADER(vs_stencil),
	BGFX_EMBEDDED_SHADER(fs_stencil),

	BGFX_EMBEDDED_SHADER_END()
};

struct State
{
	float m_TransformMtx[6];
	float m_ScissorRect[4];
	float m_GlobalAlpha;
	float m_FontScale;
	float m_AvgScale;
};

struct ClipState
{
	ClipRule::Enum m_Rule;
	uint32_t m_FirstCmdID;
	uint32_t m_NumCmds;
};

struct HandleFlags
{
	enum Enum : uint16_t
	{
		LocalHandle = 0x0001
	};
};

struct Gradient
{
	float m_Matrix[9];
	float m_Params[4]; // {Extent.x, Extent.y, Radius, Feather}
	float m_InnerColor[4];
	float m_OuterColor[4];
};

struct ImagePattern
{
	float m_Matrix[9];
	ImageHandle m_ImageHandle;
};

struct DrawCommand
{
	struct Type
	{
		// NOTE: Originally there were only 3 types of commands, Textured, ColorGradient and Clip.
		// In order to be able to support int16 UVs *and* repeatable image patterns (which require UVs
		// outside the [0, 1) range), a separate type of command has been added for image patterns.
		// The vertex shader of ImagePattern command calculates UVs the same way the gradient shader
		// calculates the gradient factor.
		// The idea is that when using multiple image patterns, a new draw call will always be created
		// for each image, so there's little harm in changing shader program as well (?!?). In other words,
		// 2 paths with different image patterns wouldn't have been batched together either way.
		enum Enum : uint32_t
		{
			Textured = 0,
			ColorGradient,
			ImagePattern,
			Clip,

			NumTypes
		};
	};

	Type::Enum m_Type;
	ClipState m_ClipState;
	uint32_t m_VertexBufferID;
	uint32_t m_FirstVertexID;
	uint32_t m_FirstIndexID;
	uint32_t m_NumVertices;
	uint32_t m_NumIndices;
	uint16_t m_ScissorRect[4];
	uint16_t m_HandleID; // Type::Textured => ImageHandle, Type::ColorGradient => GradientHandle, Type::ImagePattern => ImagePatternHandle
};

struct GPUVertexBuffer
{
	bgfx::DynamicVertexBufferHandle m_PosBufferHandle;
	bgfx::DynamicVertexBufferHandle m_UVBufferHandle;
	bgfx::DynamicVertexBufferHandle m_ColorBufferHandle;
};

struct GPUIndexBuffer
{
	bgfx::DynamicIndexBufferHandle m_bgfxHandle;
};

struct VertexBuffer
{
	float* m_Pos;
	uv_t* m_UV;
	uint32_t* m_Color;
	uint32_t m_Count;
};

struct IndexBuffer
{
	uint16_t* m_Indices;
	uint32_t m_Count;
	uint32_t m_Capacity;
};

struct Image
{
	uint16_t m_Width;
	uint16_t m_Height;
	uint32_t m_Flags;
	bgfx::TextureHandle m_bgfxHandle;
	bool m_Owned;
};

struct CommandType
{
	enum Enum : uint32_t
	{
		// Path commands
		BeginPath = 0,
		MoveTo,
		LineTo,
		CubicTo,
		QuadraticTo,
		ArcTo,
		Arc,
		Rect,
		RoundedRect,
		RoundedRectVarying,
		Circle,
		Ellipse,
		Polyline,
		ClosePath,
		FirstPathCommand = BeginPath,
		LastPathCommand = ClosePath,

		// Stroker commands
		FillPathColor,
		FillPathGradient,
		FillPathImagePattern,
		StrokePathColor,
		StrokePathGradient,
		StrokePathImagePattern,

		FirstStrokerCommand = FillPathColor,
		LastStrokerCommand = StrokePathImagePattern,

		//
		IndexedTriList,

		// State commands
		BeginClip,
		EndClip,
		ResetClip,
		CreateLinearGradient,
		CreateBoxGradient,
		CreateRadialGradient,
		CreateImagePattern,
		PushState,
		PopState,
		ResetScissor,
		SetScissor,
		IntersectScissor,
		TransformIdentity,
		TransformScale,
		TransformTranslate,
		TransformRotate,
		TransformMult,
		SetViewBox,
		SetGlobalAlpha,

		// Text
		Text,
		TextBox,

		// Command lists
		SubmitCommandList,
	};
};

struct CommandHeader
{
	CommandType::Enum m_Type;
	uint32_t m_Size;
};

struct CachedMesh
{
	float* m_Pos;
	uint32_t* m_Colors;
	uint16_t* m_Indices;
	uint32_t m_NumVertices;
	uint32_t m_NumIndices;
};

struct CachedCommand
{
	uint16_t m_FirstMeshID;
	uint16_t m_NumMeshes;
	float m_InvTransformMtx[6];
};

static constexpr uint32_t kMeshGrowth    = 256;
static constexpr uint32_t kCommandGrowth = 256;

struct CommandListCache
{
	CachedMesh* m_Meshes;
	uint32_t m_NumMeshes;
	uint32_t m_MaxMeshes;
	CachedCommand* m_Commands;
	uint32_t m_NumCommands;
	uint32_t m_MaxCommands;
	float m_AvgScale;
};

struct CommandList
{
	uint8_t* m_CommandBuffer;
	uint32_t m_CommandBufferCapacity;
	uint32_t m_CommandBufferPos;

	char* m_StringBuffer;
	uint32_t m_StringBufferCapacity;
	uint32_t m_StringBufferPos;

	uint32_t m_Flags;
	uint16_t m_NumGradients;
	uint16_t m_NumImagePatterns;

	CommandListCache* m_Cache;
};

struct Context
{
	Init m_Config;
	Stats m_Stats;
	bx::AllocatorI* m_Allocator;
	uint16_t m_ViewID;
	uint16_t m_CanvasWidth;
	uint16_t m_CanvasHeight;
	float m_DevicePixelRatio;
	float m_TesselationTolerance;
	float m_FringeWidth;

	Stroker* m_Stroker;
	Path* m_Path;

	VertexBuffer* m_VertexBuffers;
	GPUVertexBuffer* m_GPUVertexBuffers;
	uint32_t m_NumVertexBuffers;
	uint32_t m_VertexBufferCapacity;
	uint32_t m_FirstVertexBufferID;

	IndexBuffer* m_IndexBuffers;
	GPUIndexBuffer* m_GPUIndexBuffers;
	uint32_t m_NumIndexBuffers;
	uint16_t m_ActiveIndexBufferID;

	bx::AllocatorI* m_PosBufferPool;
	bx::AllocatorI* m_ColorBufferPool;
	bx::AllocatorI* m_UVBufferPool;

#if BX_CONFIG_SUPPORTS_THREADING
	bx::Mutex* m_DataPoolMutex;
#endif // BX_CONFIG_SUPPORTS_THREADING

	Image* m_Images;
	uint32_t m_ImageCapacity;
	bx::HandleAlloc* m_ImageHandleAlloc;

	CommandList* m_CmdLists;
	bx::HandleAlloc* m_CmdListHandleAlloc;
	uint32_t m_SubmitCmdListRecursionDepth;
#if VG_CONFIG_ENABLE_SHAPE_CACHING
	CommandListCache* m_CmdListCacheStack[VG_CONFIG_COMMAND_LIST_CACHE_STACK_SIZE];
	uint32_t m_CmdListCacheStackTop;
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING

	float* m_TransformedVertices;
	uint32_t m_TransformedVertexCapacity;
	bool m_PathTransformed;

	DrawCommand* m_DrawCommands;
	uint32_t m_NumDrawCommands;
	uint32_t m_DrawCommandCapacity;

	State* m_StateStack;
	uint32_t m_StateStackTop;

	ClipState m_ClipState;
	DrawCommand* m_ClipCommands;
	uint32_t m_NumClipCommands;
	uint32_t m_ClipCommandCapacity;
	bool m_RecordClipCommands;
	bool m_ForceNewClipCommand;
	bool m_ForceNewDrawCommand;

	Gradient* m_Gradients;
	uint32_t m_NextGradientID;

	ImagePattern* m_ImagePatterns;
	uint32_t m_NextImagePatternID;

	FontSystem* m_FontSystem;
	float* m_TextVertices;
	uint32_t m_TextVertexCapacity;

	bgfx::VertexLayout m_PosVertexDecl;
	bgfx::VertexLayout m_UVVertexDecl;
	bgfx::VertexLayout m_ColorVertexDecl;
	bgfx::ProgramHandle m_ProgramHandle[DrawCommand::Type::NumTypes];
	bgfx::UniformHandle m_TexUniform;
	bgfx::UniformHandle m_PaintMatUniform;
	bgfx::UniformHandle m_ExtentRadiusFeatherUniform;
	bgfx::UniformHandle m_InnerColorUniform;
	bgfx::UniformHandle m_OuterColorUniform;

	// API implementation (see corresponding free functions).
	void beginPath();
	void moveTo(float _x, float _y);
	void lineTo(float _x, float _y);
	void cubicTo(float _c1x, float _c1y, float _c2x, float _c2y, float _x, float _y);
	void quadraticTo(float _cx, float _cy, float _x, float _y);
	void arc(float _cx, float _cy, float _r, float _a0, float _a1, Winding::Enum _dir);
	void arcTo(float _x1, float _y1, float _x2, float _y2, float _r);
	void rect(float _x, float _y, float _w, float _h);
	void roundedRect(float _x, float _y, float _w, float _h, float _r);
	void roundedRectVarying(float _x, float _y, float _w, float _h, float _rtl, float _rtr, float _rbr, float _rbl);
	void circle(float _cx, float _cy, float _radius);
	void ellipse(float _cx, float _cy, float _rx, float _ry);
	void polyline(const float* _coords, uint32_t _numPoints);
	void closePath();
	void fillPathColor(Color _color, uint32_t _flags);
	void fillPathGradient(GradientHandle _gradientHandle, uint32_t _flags);
	void fillPathImagePattern(ImagePatternHandle _imgPatternHandle, Color _color, uint32_t _flags);
	void strokePathColor(Color _color, float _width, uint32_t _flags);
	void strokePathGradient(GradientHandle _gradientHandle, float _width, uint32_t _flags);
	void strokePathImagePattern(ImagePatternHandle _imgPatternHandle, Color _color, float _width, uint32_t _flags);
	void beginClip(ClipRule::Enum _rule);
	void endClip();
	void resetClip();
	GradientHandle createLinearGradient(float _sx, float _sy, float _ex, float _ey, Color _icol, Color _ocol);
	GradientHandle createBoxGradient(float _x, float _y, float _w, float _h, float _r, float _f, Color _icol, Color _ocol);
	GradientHandle createRadialGradient(float _cx, float _cy, float _inr, float _outr, Color _icol, Color _ocol);
	ImagePatternHandle createImagePattern(float _cx, float _cy, float _w, float _h, float _angle, ImageHandle _image);
	void pushState();
	void popState();
	void resetScissor();
	void setScissor(float _x, float _y, float _w, float _h);
	bool intersectScissor(float _x, float _y, float _w, float _h);
	void transformIdentity();
	void transformScale(float _x, float _y);
	void transformTranslate(float _x, float _y);
	void transformRotate(float _ang_rad);
	void transformMult(const float* _mtx, TransformOrder::Enum _order);
	void setViewBox(float _x, float _y, float _w, float _h);
	void setGlobalAlpha(float _alpha);
	void indexedTriList(const float* _pos, const uv_t* _uv, uint32_t _numVertices, const Color* _colors, uint32_t _numColors, const uint16_t* _indices, uint32_t _numIndices, ImageHandle _img);
	void text(const TextConfig& _cfg, float _x, float _y, const char* _str, const char* _end);
	void textBox(const TextConfig& _cfg, float _x, float _y, float _breakWidth, const char* _str, const char* _end, uint32_t _textboxFlags);
	void submitCommandList(CommandListHandle _handle);
	void begin(uint16_t _viewID, uint16_t _canvasWidth, uint16_t _canvasHeight, float _devicePixelRatio);
	void end();
	void frame();
	const Stats* getStats();
	void getTransform(float* _mtx);
	void getScissor(float* _rect);
	FontHandle createFont(const char* _name, uint8_t* _data, uint32_t _size, uint32_t _flags);
	FontHandle getFontByName(const char* _name);
	bool setFallbackFont(FontHandle _base, FontHandle _fallback);
	float measureText(const TextConfig& _cfg, float _x, float _y, const char* _str, const char* _end, float* _bounds);
	void measureTextBox(const TextConfig& _cfg, float _x, float _y, float _breakWidth, const char* _str, const char* _end, float* _bounds, uint32_t _textBreakFlags);
	float getTextLineHeight(const TextConfig& _cfg);
	int textBreakLines(const TextConfig& _cfg, const char* _str, const char* _end, float _breakRowWidth, TextRow* _rows, int _maxRows, uint32_t _flags);
	int textGlyphPositions(const TextConfig& _cfg, float _x, float _y, const char* _str, const char* _end, GlyphPosition* _positions, int _maxPositions);
	bool getImageSize(ImageHandle _handle, uint16_t* _w, uint16_t* _h);
	ImageHandle createImage(uint16_t _w, uint16_t _h, uint32_t _flags, const uint8_t* _data);
	ImageHandle createImage(uint32_t _flags, const bgfx::TextureHandle& _bgfxTextureHandle);
	bool updateImage(ImageHandle _image, uint16_t _x, uint16_t _y, uint16_t _w, uint16_t _h, const uint8_t* _data);
	bool destroyImage(ImageHandle _img);
	bool isImageValid(ImageHandle _image);
	CommandListHandle createCommandList(uint32_t _flags);
	void destroyCommandList(CommandListHandle _handle);
	void resetCommandList(CommandListHandle _handle);
	void beginPath(CommandListHandle _handle);
	void moveTo(CommandListHandle _handle, float _x, float _y);
	void lineTo(CommandListHandle _handle, float _x, float _y);
	void cubicTo(CommandListHandle _handle, float _c1x, float _c1y, float _c2x, float _c2y, float _x, float _y);
	void quadraticTo(CommandListHandle _handle, float _cx, float _cy, float _x, float _y);
	void arc(CommandListHandle _handle, float _cx, float _cy, float _r, float _a0, float _a1, Winding::Enum _dir);
	void arcTo(CommandListHandle _handle, float _x1, float _y1, float _x2, float _y2, float _r);
	void rect(CommandListHandle _handle, float _x, float _y, float _w, float _h);
	void roundedRect(CommandListHandle _handle, float _x, float _y, float _w, float _h, float _r);
	void roundedRectVarying(CommandListHandle _handle, float _x, float _y, float _w, float _h, float _rtl, float _rtr, float _rbr, float _rbl);
	void circle(CommandListHandle _handle, float _cx, float _cy, float _radius);
	void ellipse(CommandListHandle _handle, float _cx, float _cy, float _rx, float _ry);
	void polyline(CommandListHandle _handle, const float* _coords, uint32_t _numPoints);
	void closePath(CommandListHandle _handle);
	void indexedTriList(CommandListHandle _handle, const float* _pos, const uv_t* _uv, uint32_t _numVertices, const Color* _color, uint32_t _numColors, const uint16_t* _indices, uint32_t _numIndices, ImageHandle _img);
	void fillPath(CommandListHandle _handle, Color _color, uint32_t _flags);
	void fillPath(CommandListHandle _handle, GradientHandle _gradient, uint32_t _flags);
	void fillPath(CommandListHandle _handle, ImagePatternHandle _img, Color _color, uint32_t _flags);
	void strokePath(CommandListHandle _handle, Color _color, float _width, uint32_t _flags);
	void strokePath(CommandListHandle _handle, GradientHandle _gradient, float _width, uint32_t _flags);
	void strokePath(CommandListHandle _handle, ImagePatternHandle _img, Color _color, float _width, uint32_t _flags);
	void beginClip(CommandListHandle _handle, ClipRule::Enum _rule);
	void endClip(CommandListHandle _handle);
	void resetClip(CommandListHandle _handle);
	GradientHandle createLinearGradient(CommandListHandle _handle, float _sx, float _sy, float _ex, float _ey, Color _icol, Color _ocol);
	GradientHandle createBoxGradient(CommandListHandle _handle, float _x, float _y, float _w, float _h, float _r, float _f, Color _icol, Color _ocol);
	GradientHandle createRadialGradient(CommandListHandle _handle, float _cx, float _cy, float _inr, float _outr, Color _icol, Color _ocol);
	ImagePatternHandle createImagePattern(CommandListHandle _handle, float _cx, float _cy, float _w, float _h, float _angle, ImageHandle _image);
	void pushState(CommandListHandle _handle);
	void popState(CommandListHandle _handle);
	void resetScissor(CommandListHandle _handle);
	void setScissor(CommandListHandle _handle, float _x, float _y, float _w, float _h);
	void intersectScissor(CommandListHandle _handle, float _x, float _y, float _w, float _h);
	void transformIdentity(CommandListHandle _handle);
	void transformScale(CommandListHandle _handle, float _x, float _y);
	void transformTranslate(CommandListHandle _handle, float _x, float _y);
	void transformRotate(CommandListHandle _handle, float _ang_rad);
	void transformMult(CommandListHandle _handle, const float* _mtx, TransformOrder::Enum _order);
	void setViewBox(CommandListHandle _handle, float _x, float _y, float _w, float _h);
	void setGlobalAlpha(CommandListHandle _handle, float _alpha);
	void text(CommandListHandle _handle, const TextConfig& _cfg, float _x, float _y, const char* _str, const char* _end);
	void textBox(CommandListHandle _handle, const TextConfig& _cfg, float _x, float _y, float _breakWidth, const char* _str, const char* _end, uint32_t _textboxFlags);
	void submitCommandList(CommandListHandle _parent, CommandListHandle _child);
	bool init(bx::AllocatorI* _allocator, const Init* _cfg);
	void shutdown();
};

static State* getState(Context* _ctx);
static void updateState(State* _state);

static float* allocTransformedVertices(Context* _ctx, uint32_t _numVertices);
static const float* transformPath(Context* _ctx);

static VertexBuffer* allocVertexBuffer(Context* _ctx);
static void releaseVertexBufferPosCallback(void* _ptr, void* _userData);
static void releaseVertexBufferColorCallback(void* _ptr, void* _userData);
static void releaseVertexBufferUVCallback(void* _ptr, void* _userData);

static uint16_t allocIndexBuffer(Context* _ctx);
static void releaseIndexBuffer(Context* _ctx, uint16_t* _data);
static void releaseIndexBufferCallback(void* _ptr, void* _userData);

static DrawCommand* allocDrawCommand(Context* _ctx, uint32_t _numVertices, uint32_t _numIndices, DrawCommand::Type::Enum _type, uint16_t _handle);
static DrawCommand* allocClipCommand(Context* _ctx, uint32_t _numVertices, uint32_t _numIndices);
static void createDrawCommand_VertexColor(Context* _ctx, const float* _vtx, uint32_t _numVertices, const uint32_t* _colors, uint32_t _numColors, const uint16_t* _indices, uint32_t _numIndices);
static void createDrawCommand_ImagePattern(Context* _ctx, ImagePatternHandle _handle, const float* _vtx, uint32_t _numVertices, const uint32_t* _colors, uint32_t _numColors, const uint16_t* _indices, uint32_t _numIndices);
static void createDrawCommand_ColorGradient(Context* _ctx, GradientHandle _handle, const float* _vtx, uint32_t _numVertices, const uint32_t* _colors, uint32_t _numColors, const uint16_t* _indices, uint32_t _numIndices);
static void createDrawCommand_Clip(Context* _ctx, const float* _vtx, uint32_t _numVertices, const uint16_t* _indices, uint32_t _numIndices);

static ImageHandle allocImage(Context* _ctx);
static void resetImage(Image* _img);

static void renderTextQuads(Context* _ctx, const TextQuad* _quads, uint32_t _numQuads, Color _color, ImageHandle _img);

static CommandListHandle allocCommandList(Context* _ctx);
static bool isCommandListHandleValid(Context* _ctx, CommandListHandle _handle);
static uint8_t* clAllocCommand(Context* _ctx, CommandList* _cl, CommandType::Enum _cmdType, uint32_t _dataSize);
static uint32_t clStoreString(Context* _ctx, CommandList* _cl, const char* _str, uint32_t _len);

#if VG_CONFIG_ENABLE_SHAPE_CACHING
static void clCacheRender(Context* _ctx, CommandList* _cl);
static void clCacheReset(Context* _ctx, CommandListCache* _cache);
static CommandListCache* clGetCache(Context* _ctx, CommandList* _cl);
static CommandListCache* allocCommandListCache(Context* _ctx);
static void freeCommandListCache(Context* _ctx, CommandListCache* _cache);
static void pushCommandListCache(Context* _ctx, CommandListCache* _cache);
static void popCommandListCache(Context* _ctx);
static CommandListCache* getCommandListCacheStackTop(Context* _ctx);
static void beginCachedCommand(Context* _ctx);
static void endCachedCommand(Context* _ctx);
static void addCachedCommand(Context* _ctx, const float* _pos, uint32_t _numVertices, const uint32_t* _colors, uint32_t _numColors, const uint16_t* _indices, uint32_t _numIndices);
static void submitCachedMesh(Context* _ctx, Color _col, const CachedMesh* _meshList, uint32_t _numMeshes);
static void submitCachedMesh(Context* _ctx, GradientHandle _gradientHandle, const CachedMesh* _meshList, uint32_t _numMeshes);
static void submitCachedMesh(Context* _ctx, ImagePatternHandle _imgPatter, Color _color, const CachedMesh* _meshList, uint32_t _numMeshes);
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING


#define CMD_WRITE(ptr, type, value) *(type*)(ptr) = (value); ptr += sizeof(type)
#define CMD_READ(ptr, type) *(type*)(ptr); ptr += sizeof(type)

inline uint32_t alignSize(uint32_t _sz, uint32_t _alignment)
{
	VG_CHECK(bx::isPowerOf2<uint32_t>(_alignment), "Invalid alignment value");
	const uint32_t mask = _alignment - 1;
	return (_sz & (~mask)) + ((_sz & mask) != 0 ? _alignment : 0);
}

inline bool isAligned(uint32_t _sz, uint32_t _alignment)
{
	VG_CHECK(bx::isPowerOf2<uint32_t>(_alignment), "Invalid alignment value");
	return (_sz & (_alignment - 1)) == 0;
}

static const uint32_t kAlignedCommandHeaderSize = alignSize(sizeof(CommandHeader), VG_CONFIG_COMMAND_LIST_ALIGNMENT);

inline bool isLocal(uint16_t handleFlags)      { return (handleFlags & HandleFlags::LocalHandle) != 0; }
inline bool isLocal(GradientHandle handle)     { return isLocal(handle.flags); }
inline bool isLocal(ImagePatternHandle handle) { return isLocal(handle.flags); }

//////////////////////////////////////////////////////////////////////////
// Public interface
//
static Context* s_ctx = NULL;

bool init(bx::AllocatorI* _allocator, const Init* _userCfg)
{
	static const Init defaultConfig =
	{
		.maxGradients            = 64,
		.maxImagePatterns        = 64,
		.maxFonts                = 8,
		.maxStateStackSize       = 32,
		.maxImages               = 16,
		.maxCommandLists         = 256,
		.maxVBVertices           = 65536,
		.fontAtlasImageFlags     = ImageFlags::Filter_Bilinear,
		.maxCommandListDepth     = 16,
		.resetViewTransformOnEnd = true,
	};

	const Init* cfg = _userCfg ? _userCfg : &defaultConfig;

	VG_CHECK(cfg->maxVBVertices <= 65536, "Vertex buffers cannot be larger than 64k vertices because indices are always uint16");

	const uint32_t alignment = 16;
	const uint32_t totalMem = 0
		+ alignSize(sizeof(Context), alignment)
		+ alignSize(sizeof(Gradient) * cfg->maxGradients, alignment)
		+ alignSize(sizeof(ImagePattern) * cfg->maxImagePatterns, alignment)
		+ alignSize(sizeof(State) * cfg->maxStateStackSize, alignment)
		+ alignSize(sizeof(CommandList) * cfg->maxCommandLists, alignment);

	uint8_t* mem = (uint8_t*)bx::alignedAlloc(_allocator, totalMem, alignment);
	bx::memSet(mem, 0, totalMem);

	Context* ctx = (Context*)mem;              mem += alignSize(sizeof(Context), alignment);
	ctx->m_Gradients = (Gradient*)mem;         mem += alignSize(sizeof(Gradient) * cfg->maxGradients, alignment);
	ctx->m_ImagePatterns = (ImagePattern*)mem; mem += alignSize(sizeof(ImagePattern) * cfg->maxImagePatterns, alignment);
	ctx->m_StateStack = (State*)mem;           mem += alignSize(sizeof(State) * cfg->maxStateStackSize, alignment);
	ctx->m_CmdLists = (CommandList*)mem;       mem += alignSize(sizeof(CommandList) * cfg->maxCommandLists, alignment);

	s_ctx = ctx;
	return ctx->init(_allocator, cfg);
}

bool Context::init(bx::AllocatorI* _allocator, const Init* _cfg)
{
	bx::memCopy(&m_Config, _cfg, sizeof(Init));
	m_Allocator = _allocator;
	m_ViewID = 0;
	m_DevicePixelRatio = 1.0f;
	m_TesselationTolerance = 0.25f;
	m_FringeWidth = 1.0f;
	m_StateStackTop = 0;
	m_StateStack[0].m_GlobalAlpha = 1.0f;
	resetScissor();
	transformIdentity();

#if VG_CONFIG_ENABLE_SHAPE_CACHING
	m_CmdListCacheStackTop = ~0u;
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING

	m_PosBufferPool = BX_NEW(_allocator, vgutil::PoolAllocator)(sizeof(float) * 2 * m_Config.maxVBVertices, 4, _allocator);
	m_ColorBufferPool = BX_NEW(_allocator, vgutil::PoolAllocator)(sizeof(uint32_t) * m_Config.maxVBVertices, 4, _allocator);
	m_UVBufferPool = BX_NEW(_allocator, vgutil::PoolAllocator)(sizeof(uv_t) * 2 * m_Config.maxVBVertices, 4, _allocator);

#if BX_CONFIG_SUPPORTS_THREADING
	m_DataPoolMutex = BX_NEW(_allocator, bx::Mutex)();
#endif // BX_CONFIG_SUPPORTS_THREADING
	m_Path = createPath(_allocator);
	m_Stroker = createStroker(_allocator);

	m_ImageHandleAlloc = bx::createHandleAlloc(_allocator, _cfg->maxImages);
	m_CmdListHandleAlloc = bx::createHandleAlloc(_allocator, _cfg->maxCommandLists);

	// bgfx setup
	m_PosVertexDecl.begin().add(bgfx::Attrib::Position, 2, bgfx::AttribType::Float).end();
	m_ColorVertexDecl.begin().add(bgfx::Attrib::Color0, 4, bgfx::AttribType::Uint8, true).end();
#if VG_CONFIG_UV_INT16
	m_UVVertexDecl.begin().add(bgfx::Attrib::TexCoord0, 2, bgfx::AttribType::Int16, true).end();
#else
	m_UVVertexDecl.begin().add(bgfx::Attrib::TexCoord0, 2, bgfx::AttribType::Float).end();
#endif // VG_CONFIG_UV_INT16

	// NOTE: A couple of shaders can be shared between programs. Since bgfx
	// cares only whether the program handle changed and not (at least the D3D11 backend
	// doesn't check shader handles), there's little point in complicating this atm.
	bgfx::RendererType::Enum bgfxRendererType = bgfx::getRendererType();
	m_ProgramHandle[DrawCommand::Type::Textured] = bgfx::createProgram(
		bgfx::createEmbeddedShader(s_EmbeddedShaders, bgfxRendererType, "vs_textured"),
		bgfx::createEmbeddedShader(s_EmbeddedShaders, bgfxRendererType, "fs_textured"),
		true);

	m_ProgramHandle[DrawCommand::Type::ColorGradient] = bgfx::createProgram(
		bgfx::createEmbeddedShader(s_EmbeddedShaders, bgfxRendererType, "vs_color_gradient"),
		bgfx::createEmbeddedShader(s_EmbeddedShaders, bgfxRendererType, "fs_color_gradient"),
		true);

	m_ProgramHandle[DrawCommand::Type::ImagePattern] = bgfx::createProgram(
		bgfx::createEmbeddedShader(s_EmbeddedShaders, bgfxRendererType, "vs_image_pattern"),
		bgfx::createEmbeddedShader(s_EmbeddedShaders, bgfxRendererType, "fs_image_pattern"),
		true);

	m_ProgramHandle[DrawCommand::Type::Clip] = bgfx::createProgram(
		bgfx::createEmbeddedShader(s_EmbeddedShaders, bgfxRendererType, "vs_stencil"),
		bgfx::createEmbeddedShader(s_EmbeddedShaders, bgfxRendererType, "fs_stencil"),
		true);

	m_TexUniform = bgfx::createUniform("s_tex", bgfx::UniformType::Sampler, 1);
	m_PaintMatUniform = bgfx::createUniform("u_paintMat", bgfx::UniformType::Mat3, 1);
	m_ExtentRadiusFeatherUniform = bgfx::createUniform("u_extentRadiusFeather", bgfx::UniformType::Vec4, 1);
	m_InnerColorUniform = bgfx::createUniform("u_innerCol", bgfx::UniformType::Vec4, 1);
	m_OuterColorUniform = bgfx::createUniform("u_outerCol", bgfx::UniformType::Vec4, 1);

	// Initialize font system
	const bgfx::Caps* caps = bgfx::getCaps();
	FontSystemConfig fsCfg
	{
		.m_AtlasWidth = VG_CONFIG_MIN_FONT_ATLAS_SIZE,
		.m_AtlasHeight = VG_CONFIG_MIN_FONT_ATLAS_SIZE,
		// NOTE: White rect might get too large but since the atlas limit is the texture size limit
		// it should be that large. Otherwise shapes cached when the atlas was 512x512 will get wrong
		// white pixel UVs when the atlas gets to the texture size limit (should not happen but better
		// be safe).
		.m_WhiteRectWidth = (uint16_t)(caps->limits.maxTextureSize / VG_CONFIG_MIN_FONT_ATLAS_SIZE),
		.m_WhiteRectHeight = (uint16_t)(caps->limits.maxTextureSize / VG_CONFIG_MIN_FONT_ATLAS_SIZE),
		.m_MaxTextureSize = caps->limits.maxTextureSize,
		.m_Flags = FontSystemFlags::Origin_TopLeft,
		.m_FontAtlasImageFlags = _cfg->fontAtlasImageFlags,
	};
	m_FontSystem = fsCreate(this, _allocator, &fsCfg);
	if (!m_FontSystem)
	{
		shutdown();
		return false;
	}

	return true;
}

void shutdown()
{
	s_ctx->shutdown();
}

void Context::shutdown()
{
	bx::AllocatorI* allocator = m_Allocator;

	for (uint32_t ii = 0; ii < DrawCommand::Type::NumTypes; ++ii)
	{
		if (bgfx::isValid(m_ProgramHandle[ii]))
		{
			bgfx::destroy(m_ProgramHandle[ii]);
			m_ProgramHandle[ii] = BGFX_INVALID_HANDLE;
		}
	}

	bgfx::destroy(m_TexUniform);
	bgfx::destroy(m_PaintMatUniform);
	bgfx::destroy(m_ExtentRadiusFeatherUniform);
	bgfx::destroy(m_InnerColorUniform);
	bgfx::destroy(m_OuterColorUniform);

	for (uint32_t ii = 0; ii < m_VertexBufferCapacity; ++ii)
	{
		GPUVertexBuffer* vb = &m_GPUVertexBuffers[ii];
		if (bgfx::isValid(vb->m_PosBufferHandle))
		{
			bgfx::destroy(vb->m_PosBufferHandle);
			vb->m_PosBufferHandle = BGFX_INVALID_HANDLE;
		}
		if (bgfx::isValid(vb->m_UVBufferHandle))
		{
			bgfx::destroy(vb->m_UVBufferHandle);
			vb->m_UVBufferHandle = BGFX_INVALID_HANDLE;
		}
		if (bgfx::isValid(vb->m_ColorBufferHandle))
		{
			bgfx::destroy(vb->m_ColorBufferHandle);
			vb->m_ColorBufferHandle = BGFX_INVALID_HANDLE;
		}
	}
	bx::free(allocator, m_GPUVertexBuffers);
	bx::free(allocator, m_VertexBuffers);
	m_GPUVertexBuffers = NULL;
	m_VertexBuffers = NULL;
	m_VertexBufferCapacity = 0;
	m_NumVertexBuffers = 0;

	for (uint32_t ii = 0; ii < m_NumIndexBuffers; ++ii)
	{
		GPUIndexBuffer* gpuib = &m_GPUIndexBuffers[ii];
		if (bgfx::isValid(gpuib->m_bgfxHandle))
		{
			bgfx::destroy(gpuib->m_bgfxHandle);
			gpuib->m_bgfxHandle = BGFX_INVALID_HANDLE;
		}

		IndexBuffer* ib = &m_IndexBuffers[ii];
		bx::alignedFree(allocator, ib->m_Indices, 16);
		ib->m_Indices = NULL;
		ib->m_Capacity = 0;
		ib->m_Count = 0;
	}
	bx::free(allocator, m_GPUIndexBuffers);
	bx::free(allocator, m_IndexBuffers);
	m_GPUIndexBuffers = NULL;
	m_IndexBuffers = NULL;
	m_ActiveIndexBufferID = UINT16_MAX;

	if (m_UVBufferPool)
	{
		bx::deleteObject(allocator, m_UVBufferPool);
		m_UVBufferPool = NULL;
	}

	if (m_ColorBufferPool)
	{
		bx::deleteObject(allocator, m_ColorBufferPool);
		m_ColorBufferPool = NULL;
	}

	if (m_PosBufferPool)
	{
		bx::deleteObject(allocator, m_PosBufferPool);
		m_PosBufferPool = NULL;
 	}

	bx::free(allocator, m_DrawCommands);
	m_DrawCommands = NULL;

	bx::free(allocator, m_ClipCommands);
	m_ClipCommands = NULL;

	if (m_FontSystem)
	{
		fsDestroy(m_FontSystem, this);
		m_FontSystem = NULL;
	}

	for (uint32_t ii = 0; ii < m_ImageCapacity; ++ii)
	{
		Image* img = &m_Images[ii];
		if (bgfx::isValid(img->m_bgfxHandle))
		{
			bgfx::destroy(img->m_bgfxHandle);
		}
	}
	bx::free(allocator, m_Images);
	m_Images = NULL;

	bx::destroyHandleAlloc(allocator, m_ImageHandleAlloc);
	m_ImageHandleAlloc = NULL;

	bx::destroyHandleAlloc(allocator, m_CmdListHandleAlloc);
	m_CmdListHandleAlloc = NULL;

	destroyPath(m_Path);
	m_Path = NULL;

	destroyStroker(m_Stroker);
	m_Stroker = NULL;

	if (m_TextVertices)
	{
		bx::alignedFree(allocator, m_TextVertices, 16);
		m_TextVertices = NULL;
	}

	if (m_TransformedVertices)
	{
		bx::alignedFree(allocator, m_TransformedVertices, 16);
		m_TransformedVertices = NULL;
	}

#if BX_CONFIG_SUPPORTS_THREADING
	bx::deleteObject(allocator, m_DataPoolMutex);
#endif // BX_CONFIG_SUPPORTS_THREADING

	bx::alignedFree(allocator, this, 16);
	s_ctx = NULL;
}

void begin(uint16_t _viewID, uint16_t _canvasWidth, uint16_t _canvasHeight, float _devicePixelRatio)
{
	s_ctx->begin(_viewID, _canvasWidth, _canvasHeight, _devicePixelRatio);
}

void Context::begin(uint16_t _viewID, uint16_t _canvasWidth, uint16_t _canvasHeight, float _devicePixelRatio)
{
	m_ViewID = _viewID;
	m_CanvasWidth = _canvasWidth;
	m_CanvasHeight = _canvasHeight;
	m_DevicePixelRatio = _devicePixelRatio;
	m_TesselationTolerance = 0.25f / _devicePixelRatio;
	m_FringeWidth = 1.0f / _devicePixelRatio;
	m_SubmitCmdListRecursionDepth = 0;

#if VG_CONFIG_ENABLE_SHAPE_CACHING
	m_CmdListCacheStackTop = ~0u;
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING

	VG_CHECK(m_StateStackTop == 0, "State stack hasn't been properly reset in the previous frame");
	resetScissor();
	transformIdentity();

	m_FirstVertexBufferID = m_NumVertexBuffers;
	allocVertexBuffer(this);

	m_ActiveIndexBufferID = allocIndexBuffer(this);
	VG_CHECK(m_IndexBuffers[m_ActiveIndexBufferID].m_Count == 0, "Not empty index buffer");

	m_NumDrawCommands = 0;
	m_ForceNewDrawCommand = true;

	m_NumClipCommands = 0;
	m_ForceNewClipCommand = true;
	m_ClipState.m_FirstCmdID = ~0u;
	m_ClipState.m_NumCmds = 0;
	m_ClipState.m_Rule = ClipRule::In;

	m_NextGradientID = 0;
	m_NextImagePatternID = 0;
}

void end()
{
	s_ctx->end();
}

void Context::end()
{
	VG_CHECK(m_StateStackTop == 0, "pushState()/popState() mismatch");
	VG_CHECK(!isValid(m_ActiveCommandList), "endCommandList() hasn't been called");

	const uint32_t numDrawCommands = m_NumDrawCommands;
	if (numDrawCommands == 0)
	{
		// Release the vertex buffer allocated in beginFrame()
		VertexBuffer* vb = &m_VertexBuffers[m_FirstVertexBufferID];

		bx::free(m_PosBufferPool, vb->m_Pos);
		bx::free(m_ColorBufferPool, vb->m_Color);
		bx::free(m_UVBufferPool, vb->m_UV);

		return;
	}

	fsFlushFontAtlasImage(m_FontSystem, this);

	// Update bgfx vertex buffers...
	const uint32_t numVertexBuffers = m_NumVertexBuffers;
	for (uint32_t iVB = m_FirstVertexBufferID; iVB < numVertexBuffers; ++iVB)
	{
		VertexBuffer* vb = &m_VertexBuffers[iVB];
		GPUVertexBuffer* gpuvb = &m_GPUVertexBuffers[iVB];

		const uint32_t maxVBVertices = m_Config.maxVBVertices;
		if (!bgfx::isValid(gpuvb->m_PosBufferHandle))
		{
			gpuvb->m_PosBufferHandle = bgfx::createDynamicVertexBuffer(maxVBVertices, m_PosVertexDecl, 0);
		}
		if (!bgfx::isValid(gpuvb->m_UVBufferHandle))
		{
			gpuvb->m_UVBufferHandle = bgfx::createDynamicVertexBuffer(maxVBVertices, m_UVVertexDecl, 0);
		}
		if (!bgfx::isValid(gpuvb->m_ColorBufferHandle))
		{
			gpuvb->m_ColorBufferHandle = bgfx::createDynamicVertexBuffer(maxVBVertices, m_ColorVertexDecl, 0);
		}

		const bgfx::Memory* posMem = bgfx::makeRef(vb->m_Pos, sizeof(float) * 2 * vb->m_Count, releaseVertexBufferPosCallback, this);
		const bgfx::Memory* colorMem = bgfx::makeRef(vb->m_Color, sizeof(uint32_t) * vb->m_Count, releaseVertexBufferColorCallback, this);
		const bgfx::Memory* uvMem = bgfx::makeRef(vb->m_UV, sizeof(uv_t) * 2 * vb->m_Count, releaseVertexBufferUVCallback, this);

		bgfx::update(gpuvb->m_PosBufferHandle, 0, posMem);
		bgfx::update(gpuvb->m_UVBufferHandle, 0, uvMem);
		bgfx::update(gpuvb->m_ColorBufferHandle, 0, colorMem);

		vb->m_Pos = NULL;
		vb->m_UV = NULL;
		vb->m_Color = NULL;
	}

	// Update bgfx index buffer...
	IndexBuffer* ib = &m_IndexBuffers[m_ActiveIndexBufferID];
	GPUIndexBuffer* gpuib = &m_GPUIndexBuffers[m_ActiveIndexBufferID];
	const bgfx::Memory* indexMem = bgfx::makeRef(&ib->m_Indices[0], sizeof(uint16_t) * ib->m_Count, releaseIndexBufferCallback, this);
	if (!bgfx::isValid(gpuib->m_bgfxHandle))
	{
		gpuib->m_bgfxHandle = bgfx::createDynamicIndexBuffer(indexMem, BGFX_BUFFER_ALLOW_RESIZE);
	}
	else
	{
		bgfx::update(gpuib->m_bgfxHandle, 0, indexMem);
	}

	const uint16_t viewID = m_ViewID;
	const uint16_t canvasWidth = m_CanvasWidth;
	const uint16_t canvasHeight = m_CanvasHeight;
	const float devicePixelRatio = m_DevicePixelRatio;

	if (m_Config.resetViewTransformOnEnd)
	{
		float viewMtx[16];
		float projMtx[16];
		bx::mtxIdentity(viewMtx);
		bx::mtxOrtho(projMtx, 0.0f, (float)canvasWidth, (float)canvasHeight, 0.0f, 0.0f, 1.0f, 0.0f, bgfx::getCaps()->homogeneousDepth);
		bgfx::setViewTransform(viewID, viewMtx, projMtx);
	}

	uint16_t prevScissorRect[4] = { 0, 0, canvasWidth, canvasHeight};
	uint16_t prevScissorID = UINT16_MAX;
	uint32_t prevClipCmdID = UINT32_MAX;
	uint32_t stencilState = BGFX_STENCIL_NONE;
	uint8_t nextStencilValue = 1;

	for (uint32_t iCmd = 0; iCmd < numDrawCommands; ++iCmd)
	{
		DrawCommand* cmd = &m_DrawCommands[iCmd];

		const ClipState* cmdClipState = &cmd->m_ClipState;
		if (cmdClipState->m_FirstCmdID != prevClipCmdID)
		{
			prevClipCmdID = cmdClipState->m_FirstCmdID;
			const uint32_t numClipCommands = cmdClipState->m_NumCmds;
			if (numClipCommands)
			{
				for (uint32_t iClip = 0; iClip < numClipCommands; ++iClip)
				{
					VG_CHECK(cmdClipState->m_FirstCmdID + iClip < m_NumClipCommands, "Invalid clip command index");

					DrawCommand* clipCmd = &m_ClipCommands[cmdClipState->m_FirstCmdID + iClip];

					GPUVertexBuffer* gpuvb = &m_GPUVertexBuffers[clipCmd->m_VertexBufferID];
					bgfx::setVertexBuffer(0, gpuvb->m_PosBufferHandle, clipCmd->m_FirstVertexID, clipCmd->m_NumVertices);
					bgfx::setIndexBuffer(gpuib->m_bgfxHandle, clipCmd->m_FirstIndexID, clipCmd->m_NumIndices);

					// Set scissor.
					{
						const uint16_t* cmdScissorRect = &clipCmd->m_ScissorRect[0];
						if (!bx::memCmp(cmdScissorRect, &prevScissorRect[0], sizeof(uint16_t) * 4))
						{
							bgfx::setScissor(prevScissorID);
						}
						else
						{
							prevScissorID = bgfx::setScissor((uint16_t)(cmdScissorRect[0] * devicePixelRatio), (uint16_t)(cmdScissorRect[1] * devicePixelRatio), (uint16_t)(cmdScissorRect[2] * devicePixelRatio), (uint16_t)(cmdScissorRect[3] * devicePixelRatio));
							bx::memCopy(prevScissorRect, cmdScissorRect, sizeof(uint16_t) * 4);
						}
					}

					VG_CHECK(clipCmd->m_Type == DrawCommand::Type::Clip, "Invalid clip command");
					VG_CHECK(clipCmd->m_HandleID == UINT16_MAX, "Invalid clip command image handle");

					bgfx::setState(0);
					bgfx::setStencil(0
						| BGFX_STENCIL_TEST_ALWAYS                // pass always
						| BGFX_STENCIL_FUNC_REF(nextStencilValue) // value = nextStencilValue
						| BGFX_STENCIL_FUNC_RMASK(0xff)
						| BGFX_STENCIL_OP_FAIL_S_REPLACE
						| BGFX_STENCIL_OP_FAIL_Z_REPLACE
						| BGFX_STENCIL_OP_PASS_Z_REPLACE, BGFX_STENCIL_NONE);

					// TODO: Check if it's better to use Type_TexturedVertexColor program here to avoid too many
					// state switches.
					bgfx::submit(viewID, m_ProgramHandle[DrawCommand::Type::Clip]);
				}

				stencilState = 0
					| (cmdClipState->m_Rule == ClipRule::In ? BGFX_STENCIL_TEST_EQUAL : BGFX_STENCIL_TEST_NOTEQUAL)
					| BGFX_STENCIL_FUNC_REF(nextStencilValue)
					| BGFX_STENCIL_FUNC_RMASK(0xff)
					| BGFX_STENCIL_OP_FAIL_S_KEEP
					| BGFX_STENCIL_OP_FAIL_Z_KEEP
					| BGFX_STENCIL_OP_PASS_Z_KEEP;

				++nextStencilValue;
			}
			else
			{
				stencilState = BGFX_STENCIL_NONE;
			}
		}

		GPUVertexBuffer* gpuvb = &m_GPUVertexBuffers[cmd->m_VertexBufferID];
		bgfx::setVertexBuffer(0, gpuvb->m_PosBufferHandle, cmd->m_FirstVertexID, cmd->m_NumVertices);
		bgfx::setVertexBuffer(1, gpuvb->m_ColorBufferHandle, cmd->m_FirstVertexID, cmd->m_NumVertices);
		bgfx::setIndexBuffer(gpuib->m_bgfxHandle, cmd->m_FirstIndexID, cmd->m_NumIndices);

		// Set scissor.
		{
			const uint16_t* cmdScissorRect = &cmd->m_ScissorRect[0];
			if (!bx::memCmp(cmdScissorRect, &prevScissorRect[0], sizeof(uint16_t) * 4))
			{
				bgfx::setScissor(prevScissorID);
			}
			else
			{
				prevScissorID = bgfx::setScissor(
					(uint16_t)(cmdScissorRect[0] * devicePixelRatio),
					(uint16_t)(cmdScissorRect[1] * devicePixelRatio),
					(uint16_t)(cmdScissorRect[2] * devicePixelRatio),
					(uint16_t)(cmdScissorRect[3] * devicePixelRatio));
				bx::memCopy(prevScissorRect, cmdScissorRect, sizeof(uint16_t) * 4);
			}
		}

		if (cmd->m_Type == DrawCommand::Type::Textured)
		{
			VG_CHECK(cmd->m_HandleID != UINT16_MAX, "Invalid image handle");
			Image* tex = &m_Images[cmd->m_HandleID];

			bgfx::setVertexBuffer(2, gpuvb->m_UVBufferHandle, cmd->m_FirstVertexID, cmd->m_NumVertices);
			bgfx::setTexture(0, m_TexUniform, tex->m_bgfxHandle, tex->m_Flags);

			bgfx::setState(0
				| BGFX_STATE_WRITE_A
				| BGFX_STATE_WRITE_RGB
				| BGFX_STATE_BLEND_FUNC_SEPARATE(BGFX_STATE_BLEND_SRC_ALPHA, BGFX_STATE_BLEND_INV_SRC_ALPHA, BGFX_STATE_BLEND_ONE, BGFX_STATE_BLEND_INV_SRC_ALPHA));
			bgfx::setStencil(stencilState);

			bgfx::submit(viewID, m_ProgramHandle[DrawCommand::Type::Textured]);
		}
		else if (cmd->m_Type == DrawCommand::Type::ColorGradient)
		{
			VG_CHECK(cmd->m_HandleID != UINT16_MAX, "Invalid gradient handle");
			Gradient* grad = &m_Gradients[cmd->m_HandleID];

			bgfx::setUniform(m_PaintMatUniform, grad->m_Matrix, 1);
			bgfx::setUniform(m_ExtentRadiusFeatherUniform, grad->m_Params, 1);
			bgfx::setUniform(m_InnerColorUniform, grad->m_InnerColor, 1);
			bgfx::setUniform(m_OuterColorUniform, grad->m_OuterColor, 1);

			bgfx::setState(0
				| BGFX_STATE_WRITE_A
				| BGFX_STATE_WRITE_RGB
				| BGFX_STATE_BLEND_FUNC_SEPARATE(BGFX_STATE_BLEND_SRC_ALPHA, BGFX_STATE_BLEND_INV_SRC_ALPHA, BGFX_STATE_BLEND_ONE, BGFX_STATE_BLEND_INV_SRC_ALPHA));
			bgfx::setStencil(stencilState);

			bgfx::submit(viewID, m_ProgramHandle[DrawCommand::Type::ColorGradient]);
		}
		else if(cmd->m_Type == DrawCommand::Type::ImagePattern)
		{
			VG_CHECK(cmd->m_HandleID != UINT16_MAX, "Invalid image pattern handle");
			ImagePattern* imgPattern = &m_ImagePatterns[cmd->m_HandleID];

			VG_CHECK(isValid(imgPattern->m_ImageHandle), "Invalid image handle in pattern");
			Image* tex = &m_Images[imgPattern->m_ImageHandle.idx];

			bgfx::setTexture(0, m_TexUniform, tex->m_bgfxHandle, tex->m_Flags);
			bgfx::setUniform(m_PaintMatUniform, imgPattern->m_Matrix, 1);

			bgfx::setState(0
				| BGFX_STATE_WRITE_A
				| BGFX_STATE_WRITE_RGB
				| BGFX_STATE_BLEND_FUNC_SEPARATE(BGFX_STATE_BLEND_SRC_ALPHA, BGFX_STATE_BLEND_INV_SRC_ALPHA, BGFX_STATE_BLEND_ONE, BGFX_STATE_BLEND_INV_SRC_ALPHA));
			bgfx::setStencil(stencilState);

			bgfx::submit(viewID, m_ProgramHandle[DrawCommand::Type::ImagePattern]);
		}
		else
		{
			VG_CHECK(false, "Unknown draw command type");
		}
	}
}

void frame()
{
	s_ctx->frame();
}

void Context::frame()
{
	m_NumVertexBuffers = 0;

	fsFrame(m_FontSystem, this);
}

const Stats* getStats()
{
	return s_ctx->getStats();
}

const Stats* Context::getStats()
{
	return &m_Stats;
}

void beginPath()
{
	s_ctx->beginPath();
}

void moveTo(float _x, float _y)
{
	s_ctx->moveTo(_x, _y);
}

void lineTo(float _x, float _y)
{
	s_ctx->lineTo(_x, _y);
}

void cubicTo(float _c1x, float _c1y, float _c2x, float _c2y, float _x, float _y)
{
	s_ctx->cubicTo(_c1x, _c1y, _c2x, _c2y, _x, _y);
}

void quadraticTo(float _cx, float _cy, float _x, float _y)
{
	s_ctx->quadraticTo(_cx, _cy, _x, _y);
}

void arc(float _cx, float _cy, float _r, float _a0, float _a1, Winding::Enum _dir)
{
	s_ctx->arc(_cx, _cy, _r, _a0, _a1, _dir);
}

void arcTo(float _x1, float _y1, float _x2, float _y2, float _r)
{
	s_ctx->arcTo(_x1, _y1, _x2, _y2, _r);
}

void rect(float _x, float _y, float _w, float _h)
{
	s_ctx->rect(_x, _y, _w, _h);
}

void roundedRect(float _x, float _y, float _w, float _h, float _r)
{
	s_ctx->roundedRect(_x, _y, _w, _h, _r);
}

void roundedRectVarying(float _x, float _y, float _w, float _h, float _rtl, float _rtr, float _rbr, float _rbl)
{
	s_ctx->roundedRectVarying(_x, _y, _w, _h, _rtl, _rtr, _rbr, _rbl);
}

void circle(float _cx, float _cy, float _radius)
{
	s_ctx->circle(_cx, _cy, _radius);
}

void ellipse(float _cx, float _cy, float _rx, float _ry)
{
	s_ctx->ellipse(_cx, _cy, _rx, _ry);
}

void polyline(const float* _coords, uint32_t _numPoints)
{
	s_ctx->polyline(_coords, _numPoints);
}

void closePath()
{
	s_ctx->closePath();
}

void fillPath(Color _color, uint32_t _flags)
{
	s_ctx->fillPathColor(_color, _flags);
}

void fillPath(GradientHandle _gradientHandle, uint32_t _flags)
{
	s_ctx->fillPathGradient(_gradientHandle, _flags);
}

void fillPath(ImagePatternHandle _imgPatternHandle, Color _color, uint32_t _flags)
{
	s_ctx->fillPathImagePattern(_imgPatternHandle, _color, _flags);
}

void strokePath(Color _color, float _width, uint32_t _flags)
{
	s_ctx->strokePathColor(_color, _width, _flags);
}

void strokePath(GradientHandle _gradientHandle, float _width, uint32_t _flags)
{
	s_ctx->strokePathGradient(_gradientHandle, _width, _flags);
}

void strokePath(ImagePatternHandle _imgPatternHandle, Color _color, float _width, uint32_t _flags)
{
	s_ctx->strokePathImagePattern(_imgPatternHandle, _color, _width, _flags);
}

void beginClip(ClipRule::Enum _rule)
{
	s_ctx->beginClip(_rule);
}

void endClip()
{
	s_ctx->endClip();
}

void resetClip()
{
	s_ctx->resetClip();
}

GradientHandle createLinearGradient(float _sx, float _sy, float _ex, float _ey, Color _icol, Color _ocol)
{
	return s_ctx->createLinearGradient(_sx, _sy, _ex, _ey, _icol, _ocol);
}

GradientHandle createBoxGradient(float _x, float _y, float _w, float _h, float _r, float _f, Color _icol, Color _ocol)
{
	return s_ctx->createBoxGradient(_x, _y, _w, _h, _r, _f, _icol, _ocol);
}

GradientHandle createRadialGradient(float _cx, float _cy, float _inr, float _outr, Color _icol, Color _ocol)
{
	return s_ctx->createRadialGradient(_cx, _cy, _inr, _outr, _icol, _ocol);
}

ImagePatternHandle createImagePattern(float _cx, float _cy, float _w, float _h, float _angle, ImageHandle _image)
{
	return s_ctx->createImagePattern(_cx, _cy, _w, _h, _angle, _image);
}

void pushState()
{
	s_ctx->pushState();
}

void popState()
{
	s_ctx->popState();
}

void resetScissor()
{
	s_ctx->resetScissor();
}

void setScissor(float _x, float _y, float _w, float _h)
{
	s_ctx->setScissor(_x, _y, _w, _h);
}

bool intersectScissor(float _x, float _y, float _w, float _h)
{
	return s_ctx->intersectScissor(_x, _y, _w, _h);
}

void transformIdentity()
{
	s_ctx->transformIdentity();
}

void transformScale(float _x, float _y)
{
	s_ctx->transformScale(_x, _y);
}

void transformTranslate(float _x, float _y)
{
	s_ctx->transformTranslate(_x, _y);
}

void transformRotate(float _ang_rad)
{
	s_ctx->transformRotate(_ang_rad);
}

void transformMult(const float* _mtx, TransformOrder::Enum _order)
{
	s_ctx->transformMult(_mtx, _order);
}

void setViewBox(float _x, float _y, float _w, float _h)
{
	s_ctx->setViewBox(_x, _y, _w, _h);
}

void indexedTriList(const float* _pos, const uv_t* _uv, uint32_t _numVertices, const Color* _colors, uint32_t _numColors, const uint16_t* _indices, uint32_t _numIndices, ImageHandle _img)
{
	s_ctx->indexedTriList(_pos, _uv, _numVertices, _colors, _numColors, _indices, _numIndices, _img);
}

void text(const TextConfig& _cfg, float _x, float _y, const char* _str, const char* _end)
{
	s_ctx->text(_cfg, _x, _y, _str, _end);
}

void textBox(const TextConfig& _cfg, float _x, float _y, float _breakWidth, const char* _str, const char* _end, uint32_t _textboxFlags)
{
	s_ctx->textBox(_cfg, _x, _y, _breakWidth, _str, _end, _textboxFlags);
}

void submitCommandList(CommandListHandle _handle)
{
	s_ctx->submitCommandList(_handle);
}

void setGlobalAlpha(float _alpha)
{
	s_ctx->setGlobalAlpha(_alpha);
}

void getTransform(float* _mtx)
{
	s_ctx->getTransform(_mtx);
}

void Context::getTransform(float* _mtx)
{
	const State* state = getState(this);
	bx::memCopy(_mtx, state->m_TransformMtx, sizeof(float) * 6);
}

void getScissor(float* _rect)
{
	s_ctx->getScissor(_rect);
}

void Context::getScissor(float* _rect)
{
	const State* state = getState(this);
	bx::memCopy(_rect, state->m_ScissorRect, sizeof(float) * 4);
}

FontHandle createFont(const char* _name, uint8_t* _data, uint32_t _size, uint32_t _flags)
{
	return s_ctx->createFont(_name, _data, _size, _flags);
}

FontHandle Context::createFont(const char* _name, uint8_t* _data, uint32_t _size, uint32_t _flags)
{
	return fsAddFont(m_FontSystem, _name, _data, _size, _flags);
}

FontHandle getFontByName(const char* _name)
{
	return s_ctx->getFontByName(_name);
}

FontHandle Context::getFontByName(const char* _name)
{
	return fsFindFont(m_FontSystem, _name);
}

bool setFallbackFont(FontHandle _base, FontHandle _fallback)
{
	return s_ctx->setFallbackFont(_base, _fallback);
}

bool Context::setFallbackFont(FontHandle _base, FontHandle _fallback)
{
	VG_CHECK(isValid(_base) && isValid(_fallback), "Invalid font handle");
	return fsAddFallbackFont(m_FontSystem, _base, _fallback);
}

float measureText(const TextConfig& _cfg, float _x, float _y, const char* _str, const char* _end, float* _bounds)
{
	return s_ctx->measureText(_cfg, _x, _y, _str, _end, _bounds);
}

float Context::measureText(const TextConfig& _cfg, float _x, float _y, const char* _str, const char* _end, float* _bounds)
{
	const uint32_t len = _end
		? (uint32_t)(_end - _str)
		: bx::strLen(_str)
		;

	TextMesh mesh;
	bx::memSet(&mesh, 0, sizeof(TextMesh));
	if (!fsText(m_FontSystem, NULL, _cfg, _str, len, 0, &mesh))
	{
		if (_bounds)
		{
			bx::memSet(_bounds, 0, sizeof(float) * 4);
		}
		return 0.0f;
	}

	if (_bounds)
	{
		fsLineBounds(m_FontSystem, _cfg, 0.0f, &mesh.m_Bounds[1], &mesh.m_Bounds[3]);

		_bounds[0] = _x + mesh.m_Bounds[0];
		_bounds[1] = _y + mesh.m_Bounds[1];
		_bounds[2] = _x + mesh.m_Bounds[2];
		_bounds[3] = _y + mesh.m_Bounds[3];
	}

	return mesh.m_Width;
}

void measureTextBox(const TextConfig& _cfg, float _x, float _y, float _breakWidth, const char* _str, const char* _end, float* _bounds, uint32_t _textBreakFlags)
{
	s_ctx->measureTextBox(_cfg, _x, _y, _breakWidth, _str, _end, _bounds, _textBreakFlags);
}

void Context::measureTextBox(const TextConfig& _cfg, float _x, float _y, float _breakWidth, const char* _str, const char* _end, float* _bounds, uint32_t _textBreakFlags)
{
	VG_CHECK(_bounds != NULL, "There's no point in calling this functions without a valid bounds pointer.");

	_end = _end
		? _end
		: _str + bx::strLen(_str)
		;

	const TextAlignHor::Enum halign = (TextAlignHor::Enum)((_cfg.alignment & VG_TEXT_ALIGN_HOR_Msk) >> VG_TEXT_ALIGN_HOR_Pos);
	const TextAlignVer::Enum valign = (TextAlignVer::Enum)((_cfg.alignment & VG_TEXT_ALIGN_VER_Msk) >> VG_TEXT_ALIGN_VER_Pos);

	const TextConfig newCfg = makeTextConfig(_cfg.fontHandle, _cfg.fontSize, VG_TEXT_ALIGN(vg::TextAlignHor::Left, valign), _cfg.color, _cfg.blur, _cfg.spacing);

	fsLineBounds(m_FontSystem, newCfg, _y, &_bounds[1], &_bounds[3]);
	const float lineHeight = _bounds[3] - _bounds[1];
	_bounds[3] = _bounds[1];
	_bounds[0] = _x;
	_bounds[2] = _x;

	TextRow rows[4];
	uint32_t numRows = 0;
	while ((numRows = fsTextBreakLines(m_FontSystem, _cfg, _str, _end, _breakWidth, &rows[0], BX_COUNTOF(rows), _textBreakFlags)) != 0)
	{
		for (uint32_t ii = 0; ii < numRows; ++ii)
		{
			float dx = 0.0f;
			if (halign == TextAlignHor::Center)
			{
				dx = (_breakWidth - rows[ii].width) * 0.5f;
			}
			else if (halign == TextAlignHor::Right)
			{
				dx = _breakWidth - rows[ii].width;
			}

			_bounds[0] = bx::min<float>(_bounds[0], _x + dx + rows[ii].minx);
			_bounds[2] = bx::max<float>(_bounds[2], _x + dx + rows[ii].maxx);
		}

		_bounds[3] += lineHeight * numRows;

		_str = rows[numRows - 1].next;
	}
}

float getTextLineHeight(const TextConfig& _cfg)
{
	return s_ctx->getTextLineHeight(_cfg);
}

float Context::getTextLineHeight(const TextConfig& _cfg)
{
	return fsGetLineHeight(m_FontSystem, _cfg);
}

int textBreakLines(const TextConfig& _cfg, const char* _str, const char* _end, float _breakRowWidth, TextRow* _rows, int _maxRows, uint32_t _flags)
{
	return s_ctx->textBreakLines(_cfg, _str, _end, _breakRowWidth, _rows, _maxRows, _flags);
}

int Context::textBreakLines(const TextConfig& _cfg, const char* _str, const char* _end, float _breakRowWidth, TextRow* _rows, int _maxRows, uint32_t _flags)
{
	return (int32_t)fsTextBreakLines(m_FontSystem, _cfg, _str, _end, _breakRowWidth, _rows, _maxRows, _flags);
}

int textGlyphPositions(const TextConfig& _cfg, float _x, float _y, const char* _str, const char* _end, GlyphPosition* _positions, int _maxPositions)
{
	return s_ctx->textGlyphPositions(_cfg, _x, _y, _str, _end, _positions, _maxPositions);
}

int Context::textGlyphPositions(const TextConfig& _cfg, float _x, float _y, const char* _str, const char* _end, GlyphPosition* _positions, int _maxPositions)
{
	BX_UNUSED(_y);
	const uint32_t len = _end
		? (uint32_t)(_end - _str)
		: bx::strLen(_str)
		;

	TextMesh mesh;
	if (!fsText(m_FontSystem, this, _cfg, _str, len, 0, &mesh))
	{
		return 0;
	}

	float curX = _x;
	uint32_t cursor = 0;
	const uint32_t nn = bx::min<uint32_t>(_maxPositions, mesh.m_Size);
	for (uint32_t ii = 0; ii < nn; ++ii)
	{
		_positions[ii].str = &_str[cursor];
		_positions[ii].x = curX;
		_positions[ii].minx = _x + mesh.m_Quads[ii].m_Pos[0];
		_positions[ii].maxx = _x + mesh.m_Quads[ii].m_Pos[2];

		curX += (mesh.m_Quads[ii].m_Pos[2] - mesh.m_Quads[ii].m_Pos[0]);
		cursor += mesh.m_CodepointSize[ii];
	}

	return nn;
}

bool getImageSize(ImageHandle _handle, uint16_t* _w, uint16_t* _h)
{
	return s_ctx->getImageSize(_handle, _w, _h);
}

bool Context::getImageSize(ImageHandle _handle, uint16_t* _w, uint16_t* _h)
{
	if (!isValid(_handle))
	{
		*_w = UINT16_MAX;
		*_h = UINT16_MAX;
		return false;
	}

	Image* img = &m_Images[_handle.idx];
	if (!bgfx::isValid(img->m_bgfxHandle))
	{
		*_w = UINT16_MAX;
		*_h = UINT16_MAX;
		return false;
	}

	*_w = img->m_Width;
	*_h = img->m_Height;

	return true;
}

ImageHandle createImage(uint16_t _w, uint16_t _h, uint32_t _flags, const uint8_t* _data)
{
	return s_ctx->createImage(_w, _h, _flags, _data);
}

ImageHandle Context::createImage(uint16_t _w, uint16_t _h, uint32_t _flags, const uint8_t* _data)
{
	ImageHandle handle = allocImage(this);
	if (!isValid(handle))
	{
		return VG_INVALID_HANDLE;
	}

	Image* tex = &m_Images[handle.idx];
	tex->m_Width = _w;
	tex->m_Height = _h;

	uint32_t bgfxFlags = BGFX_SAMPLER_NONE;

#if BX_PLATFORM_EMSCRIPTEN
	if (!bx::isPowerOf2(_w) || !bx::isPowerOf2(_h))
	{
		_flags = ImageFlags::Filter_NearestUV | ImageFlags::Filter_NearestW;
		bgfxFlags |= BGFX_SAMPLER_U_CLAMP | BGFX_SAMPLER_V_CLAMP | BGFX_SAMPLER_W_CLAMP;
	}
#endif // BX_PLATFORM_EMSCRIPTEN

	if (_flags & ImageFlags::Filter_NearestUV)
	{
		bgfxFlags |= BGFX_SAMPLER_MIN_POINT | BGFX_SAMPLER_MAG_POINT;
	}
	if (_flags & ImageFlags::Filter_NearestW)
	{
		bgfxFlags |= BGFX_SAMPLER_MIP_POINT;
	}
	if (_flags & ImageFlags::Clamp_U)
	{
		bgfxFlags |= BGFX_SAMPLER_U_CLAMP;
	}
	if (_flags & ImageFlags::Clamp_V)
	{
		bgfxFlags |= BGFX_SAMPLER_V_CLAMP;
	}
	tex->m_Flags = bgfxFlags;

	tex->m_bgfxHandle = bgfx::createTexture2D(tex->m_Width, tex->m_Height, false, 1, bgfx::TextureFormat::RGBA8, bgfxFlags);
	tex->m_Owned = true;

	if (bgfx::isValid(tex->m_bgfxHandle) && _data)
	{
		const uint32_t bytesPerPixel = 4;
		const uint32_t pitch = tex->m_Width * bytesPerPixel;
		const bgfx::Memory* mem = bgfx::copy(_data, tex->m_Height * pitch);

		bgfx::updateTexture2D(tex->m_bgfxHandle, 0, 0, 0, 0, tex->m_Width, tex->m_Height, mem);
	}
	//bx::printf("[VB] createImage() : img=%hu, texture=%hu\n", handle.idx, tex->m_bgfxHandle.idx);

	return handle;
}

ImageHandle createImage(uint32_t _flags, const bgfx::TextureHandle& _bgfxTextureHandle)
{
	return s_ctx->createImage(_flags, _bgfxTextureHandle);
}

ImageHandle Context::createImage(uint32_t _flags, const bgfx::TextureHandle& _bgfxTextureHandle)
{
	VG_CHECK(bgfx::isValid(_bgfxTextureHandle), "Invalid bgfx texture handle");

	ImageHandle handle = allocImage(this);
	if (!isValid(handle))
	{
		return VG_INVALID_HANDLE;
	}

	Image* tex = &m_Images[handle.idx];
	tex->m_Width = UINT16_MAX;
	tex->m_Height = UINT16_MAX;

	uint32_t bgfxFlags = BGFX_TEXTURE_NONE;

	if (_flags & ImageFlags::Filter_NearestUV)
	{
		bgfxFlags |= BGFX_SAMPLER_MIN_POINT | BGFX_SAMPLER_MAG_POINT;
	}
	if (_flags & ImageFlags::Filter_NearestW)
	{
		bgfxFlags |= BGFX_SAMPLER_MIP_POINT;
	}
	if (_flags & ImageFlags::Clamp_U)
	{
		bgfxFlags |= BGFX_SAMPLER_U_CLAMP;
	}
	if (_flags & ImageFlags::Clamp_V)
	{
		bgfxFlags |= BGFX_SAMPLER_V_CLAMP;
	}
	tex->m_Flags = bgfxFlags;
	tex->m_Owned = false;

	tex->m_bgfxHandle.idx = _bgfxTextureHandle.idx;
	//bx::printf("[VB] createImage() : img=%hu, texture=%hu\n", handle.idx, tex->m_bgfxHandle.idx);

	return handle;
}

bool updateImage(ImageHandle _image, uint16_t _x, uint16_t _y, uint16_t _w, uint16_t _h, const uint8_t* _data)
{
	return s_ctx->updateImage(_image, _x, _y, _w, _h, _data);
}

bool Context::updateImage(ImageHandle _image, uint16_t _x, uint16_t _y, uint16_t _w, uint16_t _h, const uint8_t* _data)
{
	if (!isValid(_image))
	{
		return false;
	}

	Image* tex = &m_Images[_image.idx];
	VG_CHECK(bgfx::isValid(tex->m_bgfxHandle), "Invalid texture handle");

	const uint32_t bytesPerPixel = 4;
	const uint32_t pitch = tex->m_Width * bytesPerPixel;

	const bgfx::Memory* mem = bgfx::alloc(_w * _h * bytesPerPixel);
	bx::gather(mem->data, _data + _y * pitch + _x * bytesPerPixel, pitch, _w * bytesPerPixel, _h);

	bgfx::updateTexture2D(tex->m_bgfxHandle, 0, 0, _x, _y, _w, _h, mem, UINT16_MAX);

	return true;
}

bool destroyImage(ImageHandle _img)
{
	return s_ctx->destroyImage(_img);
}

bool Context::destroyImage(ImageHandle _img)
{
	if (!isValid(_img))
	{
		return false;
	}

	Image* tex = &m_Images[_img.idx];
	if (tex->m_Owned)
	{
		VG_CHECK(bgfx::isValid(tex->m_bgfxHandle), "Invalid texture handle");
		bgfx::destroy(tex->m_bgfxHandle);
	}
	resetImage(tex);

	m_ImageHandleAlloc->free(_img.idx);

	return true;
}

bool isImageValid(ImageHandle _image)
{
	return s_ctx->isImageValid(_image);
}

bool Context::isImageValid(ImageHandle _image)
{
	if (!isValid(_image))
	{
		return false;
	}

	Image* tex = &m_Images[_image.idx];
	return bgfx::isValid(tex->m_bgfxHandle);
}

CommandListHandle createCommandList(uint32_t _flags)
{
	return s_ctx->createCommandList(_flags);
}

CommandListHandle Context::createCommandList(uint32_t _flags)
{
	VG_CHECK(!isValid(m_ActiveCommandList), "Cannot create command list while inside a beginCommandList()/endCommandList() block");

	CommandListHandle handle = allocCommandList(this);
	if (!isValid(handle))
	{
		return VG_INVALID_HANDLE;
	}

	CommandList* cl = &m_CmdLists[handle.idx];
	cl->m_Flags = _flags;

	return handle;
}

void destroyCommandList(CommandListHandle _handle)
{
	s_ctx->destroyCommandList(_handle);
}

void Context::destroyCommandList(CommandListHandle _handle)
{
	VG_CHECK(!isValid(m_ActiveCommandList), "Cannot destroy command list while inside a beginCommandList()/endCommandList() block");
	VG_CHECK(isValid(_handle), "Invalid command list handle");

	bx::AllocatorI* allocator = m_Allocator;

	CommandList* cl = &m_CmdLists[_handle.idx];

#if VG_CONFIG_ENABLE_SHAPE_CACHING
	if (cl->m_Cache)
	{
		freeCommandListCache(this, cl->m_Cache);
	}
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING

	m_Stats.cmdListMemoryTotal -= cl->m_CommandBufferCapacity;
	m_Stats.cmdListMemoryUsed -= cl->m_CommandBufferPos;

	if (cl->m_CommandBuffer)
	{
		bx::alignedFree(allocator, cl->m_CommandBuffer, VG_CONFIG_COMMAND_LIST_ALIGNMENT);
		cl->m_CommandBuffer = NULL;
	}
	bx::free(allocator, cl->m_StringBuffer);
	bx::memSet(cl, 0, sizeof(CommandList));

	m_CmdListHandleAlloc->free(_handle.idx);
}

void resetCommandList(CommandListHandle _handle)
{
	s_ctx->resetCommandList(_handle);
}

void Context::resetCommandList(CommandListHandle _handle)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

#if VG_CONFIG_ENABLE_SHAPE_CACHING
	if (cl->m_Cache)
	{
		clCacheReset(this, cl->m_Cache);
	}
#else
	BX_UNUSED(this);
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING

	m_Stats.cmdListMemoryUsed -= cl->m_CommandBufferPos;
	cl->m_CommandBufferPos = 0;
	cl->m_StringBufferPos = 0;
	cl->m_NumImagePatterns = 0;
	cl->m_NumGradients = 0;
}

void beginPath(CommandListHandle _handle)
{
	s_ctx->beginPath(_handle);
}

void Context::beginPath(CommandListHandle _handle)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	clAllocCommand(this, cl, CommandType::BeginPath, 0);
}

void moveTo(CommandListHandle _handle, float _x, float _y)
{
	s_ctx->moveTo(_handle, _x, _y);
}

void Context::moveTo(CommandListHandle _handle, float _x, float _y)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	uint8_t* ptr = clAllocCommand(this, cl, CommandType::MoveTo, sizeof(float) * 2);
	CMD_WRITE(ptr, float, _x);
	CMD_WRITE(ptr, float, _y);
}

void lineTo(CommandListHandle _handle, float _x, float _y)
{
	s_ctx->lineTo(_handle, _x, _y);
}

void Context::lineTo(CommandListHandle _handle, float _x, float _y)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	uint8_t* ptr = clAllocCommand(this, cl, CommandType::LineTo, sizeof(float) * 2);
	CMD_WRITE(ptr, float, _x);
	CMD_WRITE(ptr, float, _y);
}

void cubicTo(CommandListHandle _handle, float _c1x, float _c1y, float _c2x, float _c2y, float _x, float _y)
{
	s_ctx->cubicTo(_handle, _c1x, _c1y, _c2x, _c2y, _x, _y);
}

void Context::cubicTo(CommandListHandle _handle, float _c1x, float _c1y, float _c2x, float _c2y, float _x, float _y)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	uint8_t* ptr = clAllocCommand(this, cl, CommandType::CubicTo, sizeof(float) * 6);
	CMD_WRITE(ptr, float, _c1x);
	CMD_WRITE(ptr, float, _c1y);
	CMD_WRITE(ptr, float, _c2x);
	CMD_WRITE(ptr, float, _c2y);
	CMD_WRITE(ptr, float, _x);
	CMD_WRITE(ptr, float, _y);
}

void quadraticTo(CommandListHandle _handle, float _cx, float _cy, float _x, float _y)
{
	s_ctx->quadraticTo(_handle, _cx, _cy, _x, _y);
}

void Context::quadraticTo(CommandListHandle _handle, float _cx, float _cy, float _x, float _y)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	uint8_t* ptr = clAllocCommand(this, cl, CommandType::QuadraticTo, sizeof(float) * 4);
	CMD_WRITE(ptr, float, _cx);
	CMD_WRITE(ptr, float, _cy);
	CMD_WRITE(ptr, float, _x);
	CMD_WRITE(ptr, float, _y);
}

void arc(CommandListHandle _handle, float _cx, float _cy, float _r, float _a0, float _a1, Winding::Enum _dir)
{
	s_ctx->arc(_handle, _cx, _cy, _r, _a0, _a1, _dir);
}

void Context::arc(CommandListHandle _handle, float _cx, float _cy, float _r, float _a0, float _a1, Winding::Enum _dir)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	uint8_t* ptr = clAllocCommand(this, cl, CommandType::Arc, sizeof(float) * 5 + sizeof(Winding::Enum));
	CMD_WRITE(ptr, float, _cx);
	CMD_WRITE(ptr, float, _cy);
	CMD_WRITE(ptr, float, _r);
	CMD_WRITE(ptr, float, _a0);
	CMD_WRITE(ptr, float, _a1);
	CMD_WRITE(ptr, Winding::Enum, _dir);
}

void arcTo(CommandListHandle _handle, float _x1, float _y1, float _x2, float _y2, float _r)
{
	s_ctx->arcTo(_handle, _x1, _y1, _x2, _y2, _r);
}

void Context::arcTo(CommandListHandle _handle, float _x1, float _y1, float _x2, float _y2, float _r)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	uint8_t* ptr = clAllocCommand(this, cl, CommandType::ArcTo, sizeof(float) * 5);
	CMD_WRITE(ptr, float, _x1);
	CMD_WRITE(ptr, float, _y1);
	CMD_WRITE(ptr, float, _x2);
	CMD_WRITE(ptr, float, _y2);
	CMD_WRITE(ptr, float, _r);
}

void rect(CommandListHandle _handle, float _x, float _y, float _w, float _h)
{
	s_ctx->rect(_handle, _x, _y, _w, _h);
}

void Context::rect(CommandListHandle _handle, float _x, float _y, float _w, float _h)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	uint8_t* ptr = clAllocCommand(this, cl, CommandType::Rect, sizeof(float) * 4);
	CMD_WRITE(ptr, float, _x);
	CMD_WRITE(ptr, float, _y);
	CMD_WRITE(ptr, float, _w);
	CMD_WRITE(ptr, float, _h);
}

void roundedRect(CommandListHandle _handle, float _x, float _y, float _w, float _h, float _r)
{
	s_ctx->roundedRect(_handle, _x, _y, _w, _h, _r);
}

void Context::roundedRect(CommandListHandle _handle, float _x, float _y, float _w, float _h, float _r)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	uint8_t* ptr = clAllocCommand(this, cl, CommandType::RoundedRect, sizeof(float) * 5);
	CMD_WRITE(ptr, float, _x);
	CMD_WRITE(ptr, float, _y);
	CMD_WRITE(ptr, float, _w);
	CMD_WRITE(ptr, float, _h);
	CMD_WRITE(ptr, float, _r);
}

void roundedRectVarying(CommandListHandle _handle, float _x, float _y, float _w, float _h, float _rtl, float _rtr, float _rbr, float _rbl)
{
	s_ctx->roundedRectVarying(_handle, _x, _y, _w, _h, _rtl, _rtr, _rbr, _rbl);
}

void Context::roundedRectVarying(CommandListHandle _handle, float _x, float _y, float _w, float _h, float _rtl, float _rtr, float _rbr, float _rbl)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	uint8_t* ptr = clAllocCommand(this, cl, CommandType::RoundedRectVarying, sizeof(float) * 8);
	CMD_WRITE(ptr, float, _x);
	CMD_WRITE(ptr, float, _y);
	CMD_WRITE(ptr, float, _w);
	CMD_WRITE(ptr, float, _h);
	CMD_WRITE(ptr, float, _rtl);
	CMD_WRITE(ptr, float, _rtr);
	CMD_WRITE(ptr, float, _rbr);
	CMD_WRITE(ptr, float, _rbl);
}

void circle(CommandListHandle _handle, float _cx, float _cy, float _radius)
{
	s_ctx->circle(_handle, _cx, _cy, _radius);
}

void Context::circle(CommandListHandle _handle, float _cx, float _cy, float _radius)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	uint8_t* ptr = clAllocCommand(this, cl, CommandType::Circle, sizeof(float) * 3);
	CMD_WRITE(ptr, float, _cx);
	CMD_WRITE(ptr, float, _cy);
	CMD_WRITE(ptr, float, _radius);
}

void ellipse(CommandListHandle _handle, float _cx, float _cy, float _rx, float _ry)
{
	s_ctx->ellipse(_handle, _cx, _cy, _rx, _ry);
}

void Context::ellipse(CommandListHandle _handle, float _cx, float _cy, float _rx, float _ry)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	uint8_t* ptr = clAllocCommand(this, cl, CommandType::Ellipse, sizeof(float) * 4);
	CMD_WRITE(ptr, float, _cx);
	CMD_WRITE(ptr, float, _cy);
	CMD_WRITE(ptr, float, _rx);
	CMD_WRITE(ptr, float, _ry);
}

void polyline(CommandListHandle _handle, const float* _coords, uint32_t _numPoints)
{
	s_ctx->polyline(_handle, _coords, _numPoints);
}

void Context::polyline(CommandListHandle _handle, const float* _coords, uint32_t _numPoints)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	uint8_t* ptr = clAllocCommand(this, cl, CommandType::Polyline, sizeof(uint32_t) + sizeof(float) * 2 * _numPoints);
	CMD_WRITE(ptr, uint32_t, _numPoints);
	bx::memCopy(ptr, _coords, sizeof(float) * 2 * _numPoints);
}

void closePath(CommandListHandle _handle)
{
	s_ctx->closePath(_handle);
}

void Context::closePath(CommandListHandle _handle)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	clAllocCommand(this, cl, CommandType::ClosePath, 0);
}

void indexedTriList(CommandListHandle _handle, const float* _pos, const uv_t* _uv, uint32_t _numVertices, const Color* _color, uint32_t _numColors, const uint16_t* _indices, uint32_t _numIndices, ImageHandle _img)
{
	s_ctx->indexedTriList(_handle, _pos, _uv, _numVertices, _color, _numColors, _indices, _numIndices, _img);
}

void Context::indexedTriList(CommandListHandle _handle, const float* _pos, const uv_t* _uv, uint32_t _numVertices, const Color* _color, uint32_t _numColors, const uint16_t* _indices, uint32_t _numIndices, ImageHandle _img)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	const uint32_t dataSize = 0
		+ sizeof(uint32_t) // num positions
		+ sizeof(float) * 2 * _numVertices // positions
		+ sizeof(uint32_t) // num UVs
		+ (_uv != NULL ? (sizeof(uv_t) * 2 * _numVertices) : 0) // UVs
		+ sizeof(uint32_t) // num colors
		+ sizeof(Color) * _numColors // colors
		+ sizeof(uint32_t) // num indices
		+ sizeof(uint16_t) * _numIndices // indices
		+ sizeof(uint16_t) // image handle
		;

	uint8_t* ptr = clAllocCommand(this, cl, CommandType::IndexedTriList, dataSize);

	// positions
	CMD_WRITE(ptr, uint32_t, _numVertices);
	bx::memCopy(ptr, _pos, sizeof(float) * 2 * _numVertices);
	ptr += sizeof(float) * 2 * _numVertices;

	// UVs
	if (_uv)
	{
		CMD_WRITE(ptr, uint32_t, _numVertices);
		bx::memCopy(ptr, _uv, sizeof(uv_t) * 2 * _numVertices);
		ptr += sizeof(uv_t) * 2 * _numVertices;
	}
	else
	{
		CMD_WRITE(ptr, uint32_t, 0);
	}

	// Colors
	CMD_WRITE(ptr, uint32_t, _numColors);
	bx::memCopy(ptr, _color, sizeof(Color) * _numColors);
	ptr += sizeof(Color) * _numColors;

	// Indices
	CMD_WRITE(ptr, uint32_t, _numIndices);
	bx::memCopy(ptr, _indices, sizeof(uint16_t) * _numIndices);
	ptr += sizeof(uint16_t) * _numIndices;

	// Image
	CMD_WRITE(ptr, uint16_t, _img.idx);
}

void fillPath(CommandListHandle _handle, Color _color, uint32_t _flags)
{
	s_ctx->fillPath(_handle, _color, _flags);
}

void Context::fillPath(CommandListHandle _handle, Color _color, uint32_t _flags)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	uint8_t* ptr = clAllocCommand(this, cl, CommandType::FillPathColor, sizeof(uint32_t) + sizeof(Color));
	CMD_WRITE(ptr, uint32_t, _flags);
	CMD_WRITE(ptr, Color, _color);
}

void fillPath(CommandListHandle _handle, GradientHandle _gradient, uint32_t _flags)
{
	s_ctx->fillPath(_handle, _gradient, _flags);
}

void Context::fillPath(CommandListHandle _handle, GradientHandle _gradient, uint32_t _flags)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	VG_CHECK(isValid(_gradient), "Invalid gradient handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	VG_CHECK(isValid(_gradient), "Invalid gradient handle");

	uint8_t* ptr = clAllocCommand(this, cl, CommandType::FillPathGradient, sizeof(uint32_t) + sizeof(uint16_t) * 2);
	CMD_WRITE(ptr, uint32_t, _flags);
	CMD_WRITE(ptr, uint16_t, _gradient.idx);
	CMD_WRITE(ptr, uint16_t, _gradient.flags);
}

void fillPath(CommandListHandle _handle, ImagePatternHandle _img, Color _color, uint32_t _flags)
{
	s_ctx->fillPath(_handle, _img, _color, _flags);
}

void Context::fillPath(CommandListHandle _handle, ImagePatternHandle _img, Color _color, uint32_t _flags)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	VG_CHECK(isValid(_img), "Invalid image pattern handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	VG_CHECK(isValid(_img), "Invalid image pattern handle");

	uint8_t* ptr = clAllocCommand(this, cl, CommandType::FillPathImagePattern, sizeof(uint32_t) + sizeof(Color) + sizeof(uint16_t) * 2);
	CMD_WRITE(ptr, uint32_t, _flags);
	CMD_WRITE(ptr, Color, _color);
	CMD_WRITE(ptr, uint16_t, _img.idx);
	CMD_WRITE(ptr, uint16_t, _img.flags);
}

void strokePath(CommandListHandle _handle, Color _color, float _width, uint32_t _flags)
{
	s_ctx->strokePath(_handle, _color, _width, _flags);
}

void Context::strokePath(CommandListHandle _handle, Color _color, float _width, uint32_t _flags)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	uint8_t* ptr = clAllocCommand(this, cl, CommandType::StrokePathColor, sizeof(float) + sizeof(uint32_t) + sizeof(Color));
	CMD_WRITE(ptr, float, _width);
	CMD_WRITE(ptr, uint32_t, _flags);
	CMD_WRITE(ptr, Color, _color);
}

void strokePath(CommandListHandle _handle, GradientHandle _gradient, float _width, uint32_t _flags)
{
	s_ctx->strokePath(_handle, _gradient, _width, _flags);
}

void Context::strokePath(CommandListHandle _handle, GradientHandle _gradient, float _width, uint32_t _flags)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	VG_CHECK(isValid(_gradient), "Invalid gradient handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	VG_CHECK(isValid(_gradient), "Invalid gradient handle");

	uint8_t* ptr = clAllocCommand(this, cl, CommandType::StrokePathGradient, sizeof(float) + sizeof(uint32_t) + sizeof(uint16_t) * 2);
	CMD_WRITE(ptr, float, _width);
	CMD_WRITE(ptr, uint32_t, _flags);
	CMD_WRITE(ptr, uint16_t, _gradient.idx);
	CMD_WRITE(ptr, uint16_t, _gradient.flags);
}

void strokePath(CommandListHandle _handle, ImagePatternHandle _img, Color _color, float _width, uint32_t _flags)
{
	s_ctx->strokePath(_handle, _img, _color, _width, _flags);
}

void Context::strokePath(CommandListHandle _handle, ImagePatternHandle _img, Color _color, float _width, uint32_t _flags)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	VG_CHECK(isValid(_img), "Invalid image pattern handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	VG_CHECK(isValid(_img), "Invalid image pattern handle");

	uint8_t* ptr = clAllocCommand(this, cl, CommandType::StrokePathImagePattern, sizeof(float) + sizeof(uint32_t) + sizeof(Color) + sizeof(uint16_t) * 2);
	CMD_WRITE(ptr, float, _width);
	CMD_WRITE(ptr, uint32_t, _flags);
	CMD_WRITE(ptr, Color, _color);
	CMD_WRITE(ptr, uint16_t, _img.idx);
	CMD_WRITE(ptr, uint16_t, _img.flags);
}

void beginClip(CommandListHandle _handle, ClipRule::Enum _rule)
{
	s_ctx->beginClip(_handle, _rule);
}

void Context::beginClip(CommandListHandle _handle, ClipRule::Enum _rule)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	uint8_t* ptr = clAllocCommand(this, cl, CommandType::BeginClip, sizeof(ClipRule::Enum));
	CMD_WRITE(ptr, ClipRule::Enum, _rule);
}

void endClip(CommandListHandle _handle)
{
	s_ctx->endClip(_handle);
}

void Context::endClip(CommandListHandle _handle)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	clAllocCommand(this, cl, CommandType::EndClip, 0);
}

void resetClip(CommandListHandle _handle)
{
	s_ctx->resetClip(_handle);
}

void Context::resetClip(CommandListHandle _handle)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	clAllocCommand(this, cl, CommandType::ResetClip, 0);
}

GradientHandle createLinearGradient(CommandListHandle _handle, float _sx, float _sy, float _ex, float _ey, Color _icol, Color _ocol)
{
	return s_ctx->createLinearGradient(_handle, _sx, _sy, _ex, _ey, _icol, _ocol);
}

GradientHandle Context::createLinearGradient(CommandListHandle _handle, float _sx, float _sy, float _ex, float _ey, Color _icol, Color _ocol)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	uint8_t* ptr = clAllocCommand(this, cl, CommandType::CreateLinearGradient, sizeof(float) * 4 + sizeof(Color) * 2);
	CMD_WRITE(ptr, float, _sx);
	CMD_WRITE(ptr, float, _sy);
	CMD_WRITE(ptr, float, _ex);
	CMD_WRITE(ptr, float, _ey);
	CMD_WRITE(ptr, Color, _icol);
	CMD_WRITE(ptr, Color, _ocol);

	const uint16_t gradientHandle = cl->m_NumGradients;
	cl->m_NumGradients++;
	return { gradientHandle, HandleFlags::LocalHandle };
}

GradientHandle createBoxGradient(CommandListHandle _handle, float _x, float _y, float _w, float _h, float _r, float _f, Color _icol, Color _ocol)
{
	return s_ctx->createBoxGradient(_handle, _x, _y, _w, _h, _r, _f, _icol, _ocol);
}

GradientHandle Context::createBoxGradient(CommandListHandle _handle, float _x, float _y, float _w, float _h, float _r, float _f, Color _icol, Color _ocol)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	uint8_t* ptr = clAllocCommand(this, cl, CommandType::CreateBoxGradient, sizeof(float) * 6 + sizeof(Color) * 2);
	CMD_WRITE(ptr, float, _x);
	CMD_WRITE(ptr, float, _y);
	CMD_WRITE(ptr, float, _w);
	CMD_WRITE(ptr, float, _h);
	CMD_WRITE(ptr, float, _r);
	CMD_WRITE(ptr, float, _f);
	CMD_WRITE(ptr, Color, _icol);
	CMD_WRITE(ptr, Color, _ocol);

	const uint16_t gradientHandle = cl->m_NumGradients;
	cl->m_NumGradients++;
	return { gradientHandle, HandleFlags::LocalHandle };
}

GradientHandle createRadialGradient(CommandListHandle _handle, float _cx, float _cy, float _inr, float _outr, Color _icol, Color _ocol)
{
	return s_ctx->createRadialGradient(_handle, _cx, _cy, _inr, _outr, _icol, _ocol);
}

GradientHandle Context::createRadialGradient(CommandListHandle _handle, float _cx, float _cy, float _inr, float _outr, Color _icol, Color _ocol)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	uint8_t* ptr = clAllocCommand(this, cl, CommandType::CreateRadialGradient, sizeof(float) * 4 + sizeof(Color) * 2);
	CMD_WRITE(ptr, float, _cx);
	CMD_WRITE(ptr, float, _cy);
	CMD_WRITE(ptr, float, _inr);
	CMD_WRITE(ptr, float, _outr);
	CMD_WRITE(ptr, Color, _icol);
	CMD_WRITE(ptr, Color, _ocol);

	const uint16_t gradientHandle = cl->m_NumGradients;
	cl->m_NumGradients++;
	return { gradientHandle, HandleFlags::LocalHandle };
}

ImagePatternHandle createImagePattern(CommandListHandle _handle, float _cx, float _cy, float _w, float _h, float _angle, ImageHandle _image)
{
	return s_ctx->createImagePattern(_handle, _cx, _cy, _w, _h, _angle, _image);
}

ImagePatternHandle Context::createImagePattern(CommandListHandle _handle, float _cx, float _cy, float _w, float _h, float _angle, ImageHandle _image)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	VG_CHECK(isValid(_image), "Invalid image handle");

	CommandList* cl = &m_CmdLists[_handle.idx];

	VG_CHECK(isValid(_image), "Invalid image handle");

	uint8_t* ptr = clAllocCommand(this, cl, CommandType::CreateImagePattern, sizeof(float) * 5 + sizeof(uint16_t));
	CMD_WRITE(ptr, float, _cx);
	CMD_WRITE(ptr, float, _cy);
	CMD_WRITE(ptr, float, _w);
	CMD_WRITE(ptr, float, _h);
	CMD_WRITE(ptr, float, _angle);
	CMD_WRITE(ptr, uint16_t, _image.idx);

	const uint16_t patternHandle = cl->m_NumImagePatterns;
	cl->m_NumImagePatterns++;
	return { patternHandle, HandleFlags::LocalHandle };
}

void pushState(CommandListHandle _handle)
{
	s_ctx->pushState(_handle);
}

void Context::pushState(CommandListHandle _handle)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	clAllocCommand(this, cl, CommandType::PushState, 0);
}

void popState(CommandListHandle _handle)
{
	s_ctx->popState(_handle);
}

void Context::popState(CommandListHandle _handle)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	clAllocCommand(this, cl, CommandType::PopState, 0);
}

void resetScissor(CommandListHandle _handle)
{
	s_ctx->resetScissor(_handle);
}

void Context::resetScissor(CommandListHandle _handle)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	clAllocCommand(this, cl, CommandType::ResetScissor, 0);
}

void setScissor(CommandListHandle _handle, float _x, float _y, float _w, float _h)
{
	s_ctx->setScissor(_handle, _x, _y, _w, _h);
}

void Context::setScissor(CommandListHandle _handle, float _x, float _y, float _w, float _h)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	uint8_t* ptr = clAllocCommand(this, cl, CommandType::SetScissor, sizeof(float) * 4);
	CMD_WRITE(ptr, float, _x);
	CMD_WRITE(ptr, float, _y);
	CMD_WRITE(ptr, float, _w);
	CMD_WRITE(ptr, float, _h);
}

void intersectScissor(CommandListHandle _handle, float _x, float _y, float _w, float _h)
{
	s_ctx->intersectScissor(_handle, _x, _y, _w, _h);
}

void Context::intersectScissor(CommandListHandle _handle, float _x, float _y, float _w, float _h)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	uint8_t* ptr = clAllocCommand(this, cl, CommandType::IntersectScissor, sizeof(float) * 4);
	CMD_WRITE(ptr, float, _x);
	CMD_WRITE(ptr, float, _y);
	CMD_WRITE(ptr, float, _w);
	CMD_WRITE(ptr, float, _h);
}

void transformIdentity(CommandListHandle _handle)
{
	s_ctx->transformIdentity(_handle);
}

void Context::transformIdentity(CommandListHandle _handle)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	clAllocCommand(this, cl, CommandType::TransformIdentity, 0);
}

void transformScale(CommandListHandle _handle, float _x, float _y)
{
	s_ctx->transformScale(_handle, _x, _y);
}

void Context::transformScale(CommandListHandle _handle, float _x, float _y)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	uint8_t* ptr = clAllocCommand(this, cl, CommandType::TransformScale, sizeof(float) * 2);
	CMD_WRITE(ptr, float, _x);
	CMD_WRITE(ptr, float, _y);
}

void transformTranslate(CommandListHandle _handle, float _x, float _y)
{
	s_ctx->transformTranslate(_handle, _x, _y);
}

void Context::transformTranslate(CommandListHandle _handle, float _x, float _y)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	uint8_t* ptr = clAllocCommand(this, cl, CommandType::TransformTranslate, sizeof(float) * 2);
	CMD_WRITE(ptr, float, _x);
	CMD_WRITE(ptr, float, _y);
}

void transformRotate(CommandListHandle _handle, float _ang_rad)
{
	s_ctx->transformRotate(_handle, _ang_rad);
}

void Context::transformRotate(CommandListHandle _handle, float _ang_rad)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	uint8_t* ptr = clAllocCommand(this, cl, CommandType::TransformRotate, sizeof(float));
	CMD_WRITE(ptr, float, _ang_rad);
}

void transformMult(CommandListHandle _handle, const float* _mtx, TransformOrder::Enum _order)
{
	s_ctx->transformMult(_handle, _mtx, _order);
}

void Context::transformMult(CommandListHandle _handle, const float* _mtx, TransformOrder::Enum _order)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	uint8_t* ptr = clAllocCommand(this, cl, CommandType::TransformMult, sizeof(float) * 6 + sizeof(TransformOrder::Enum));
	bx::memCopy(ptr, _mtx, sizeof(float) * 6);
	ptr += sizeof(float) * 6;
	CMD_WRITE(ptr, TransformOrder::Enum, _order);
}

void setViewBox(CommandListHandle _handle, float _x, float _y, float _w, float _h)
{
	s_ctx->setViewBox(_handle, _x, _y, _w, _h);
}

void Context::setViewBox(CommandListHandle _handle, float _x, float _y, float _w, float _h)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	uint8_t* ptr = clAllocCommand(this, cl, CommandType::SetViewBox, sizeof(float) * 4);
	CMD_WRITE(ptr, float, _x);
	CMD_WRITE(ptr, float, _y);
	CMD_WRITE(ptr, float, _w);
	CMD_WRITE(ptr, float, _h);
}

void setGlobalAlpha(CommandListHandle _handle, float _alpha)
{
	s_ctx->setGlobalAlpha(_handle, _alpha);
}

void Context::setGlobalAlpha(CommandListHandle _handle, float _alpha)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	uint8_t* ptr = clAllocCommand(this, cl, CommandType::SetGlobalAlpha, sizeof(float));
	CMD_WRITE(ptr, float, _alpha);
}

void text(CommandListHandle _handle, const TextConfig& _cfg, float _x, float _y, const char* _str, const char* _end)
{
	s_ctx->text(_handle, _cfg, _x, _y, _str, _end);
}

void Context::text(CommandListHandle _handle, const TextConfig& _cfg, float _x, float _y, const char* _str, const char* _end)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	VG_CHECK(isValid(_cfg.fontHandle), "Invalid font handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	const uint32_t len = _end ? (uint32_t)(_end - _str) : (uint32_t)bx::strLen(_str);
	if (len == 0)
	{
		return;
	}

	const uint32_t offset = clStoreString(this, cl, _str, len);

	uint8_t* ptr = clAllocCommand(this, cl, CommandType::Text, sizeof(TextConfig) + sizeof(float) * 2 + sizeof(uint32_t) * 2);
	bx::memCopy(ptr, &_cfg, sizeof(TextConfig));
	ptr += sizeof(TextConfig);
	CMD_WRITE(ptr, float, _x);
	CMD_WRITE(ptr, float, _y);
	CMD_WRITE(ptr, uint32_t, offset);
	CMD_WRITE(ptr, uint32_t, len);
}

void textBox(CommandListHandle _handle, const TextConfig& _cfg, float _x, float _y, float _breakWidth, const char* _str, const char* _end, uint32_t _textboxFlags)
{
	s_ctx->textBox(_handle, _cfg, _x, _y, _breakWidth, _str, _end, _textboxFlags);
}

void Context::textBox(CommandListHandle _handle, const TextConfig& _cfg, float _x, float _y, float _breakWidth, const char* _str, const char* _end, uint32_t _textboxFlags)
{
	VG_CHECK(isValid(_handle), "Invalid command list handle");
	VG_CHECK(isValid(_cfg.fontHandle), "Invalid font handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	const uint32_t len = _end ? (uint32_t)(_end - _str) : (uint32_t)bx::strLen(_str);
	if (len == 0)
	{
		return;
	}

	const uint32_t offset = clStoreString(this, cl, _str, len);

	uint8_t* ptr = clAllocCommand(this, cl, CommandType::TextBox, sizeof(TextConfig) + sizeof(float) * 3 + sizeof(uint32_t) * 3);
	bx::memCopy(ptr, &_cfg, sizeof(TextConfig));
	ptr += sizeof(TextConfig);
	CMD_WRITE(ptr, float, _x);
	CMD_WRITE(ptr, float, _y);
	CMD_WRITE(ptr, float, _breakWidth);
	CMD_WRITE(ptr, uint32_t, offset);
	CMD_WRITE(ptr, uint32_t, len);
	CMD_WRITE(ptr, uint32_t, _textboxFlags);
}

void submitCommandList(CommandListHandle _parent, CommandListHandle _child)
{
	s_ctx->submitCommandList(_parent, _child);
}

void Context::submitCommandList(CommandListHandle _parent, CommandListHandle _child)
{
	VG_CHECK(isValid(_parent), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_parent.idx];

	uint8_t* ptr = clAllocCommand(this, cl, CommandType::SubmitCommandList, sizeof(uint16_t));
	CMD_WRITE(ptr, uint16_t, _child.idx);
}

// Context
void Context::beginPath()
{
	const State* state = getState(this);
	const float avgScale = state->m_AvgScale;
	const float testTol = m_TesselationTolerance;
	const float fringeWidth = m_FringeWidth;
	Path* path = m_Path;
	Stroker* stroker = m_Stroker;

	pathReset(path, avgScale, testTol);
	strokerReset(stroker, avgScale, testTol, fringeWidth);
	m_PathTransformed = false;
}

void Context::moveTo(float _x, float _y)
{
	VG_CHECK(!m_PathTransformed, "Call beginPath() before starting a new path");
	pathMoveTo(m_Path, _x, _y);
}

void Context::lineTo(float _x, float _y)
{
	VG_CHECK(!m_PathTransformed, "Call beginPath() before starting a new path");
	pathLineTo(m_Path, _x, _y);
}

void Context::cubicTo(float _c1x, float _c1y, float _c2x, float _c2y, float _x, float _y)
{
	VG_CHECK(!m_PathTransformed, "Call beginPath() before starting a new path");
	pathCubicTo(m_Path, _c1x, _c1y, _c2x, _c2y, _x, _y);
}

void Context::quadraticTo(float _cx, float _cy, float _x, float _y)
{
	VG_CHECK(!m_PathTransformed, "Call beginPath() before starting a new path");
	pathQuadraticTo(m_Path, _cx, _cy, _x, _y);
}

void Context::arc(float _cx, float _cy, float _r, float _a0, float _a1, Winding::Enum _dir)
{
	VG_CHECK(!m_PathTransformed, "Call beginPath() before starting a new path");
	pathArc(m_Path, _cx, _cy, _r, _a0, _a1, _dir);
}

void Context::arcTo(float _x1, float _y1, float _x2, float _y2, float _r)
{
	VG_CHECK(!m_PathTransformed, "Call beginPath() before starting a new path");
	pathArcTo(m_Path, _x1, _y1, _x2, _y2, _r);
}

void Context::rect(float _x, float _y, float _w, float _h)
{
	VG_CHECK(!m_PathTransformed, "Call beginPath() before starting a new path");
	pathRect(m_Path, _x, _y, _w, _h);
}

void Context::roundedRect(float _x, float _y, float _w, float _h, float _r)
{
	VG_CHECK(!m_PathTransformed, "Call beginPath() before starting a new path");
	pathRoundedRect(m_Path, _x, _y, _w, _h, _r);
}

void Context::roundedRectVarying(float _x, float _y, float _w, float _h, float _rtl, float _rtr, float _rbr, float _rbl)
{
	VG_CHECK(!m_PathTransformed, "Call beginPath() before starting a new path");
	pathRoundedRectVarying(m_Path, _x, _y, _w, _h, _rtl, _rtr, _rbr, _rbl);
}

void Context::circle(float _cx, float _cy, float _radius)
{
	VG_CHECK(!m_PathTransformed, "Call beginPath() before starting a new path");
	pathCircle(m_Path, _cx, _cy, _radius);
}

void Context::ellipse(float _cx, float _cy, float _rx, float _ry)
{
	VG_CHECK(!m_PathTransformed, "Call beginPath() before starting a new path");
	pathEllipse(m_Path, _cx, _cy, _rx, _ry);
}

void Context::polyline(const float* _coords, uint32_t _numPoints)
{
	VG_CHECK(!m_PathTransformed, "Call beginPath() before starting a new path");
	pathPolyline(m_Path, _coords, _numPoints);
}

void Context::closePath()
{
	VG_CHECK(!m_PathTransformed, "Call beginPath() before starting a new path");
	pathClose(m_Path);
}

void Context::fillPathColor(Color _color, uint32_t _flags)
{
	const bool recordClipCommands = m_RecordClipCommands;
#if VG_CONFIG_ENABLE_SHAPE_CACHING
	const bool hasCache = getCommandListCacheStackTop(this) != NULL;
#else
	const bool hasCache = false;
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING

	const State* state = getState(this);
	const float globalAlpha = hasCache ? 1.0f : state->m_GlobalAlpha;
	const Color col = recordClipCommands ? Colors::Black : colorSetAlpha(_color, (uint8_t)(globalAlpha * colorGetAlpha(_color)));
	if (!hasCache && colorGetAlpha(col) == 0)
	{
		return;
	}

	const float* pathVertices = transformPath(this);

#if VG_CONFIG_FORCE_AA_OFF
	const bool aa = false;
#else
	const bool aa = recordClipCommands
		? false
		: (bool)((_flags & VG_FILL_FLAGS_AA_Msk) >> VG_FILL_FLAGS_AA_Pos)
		;
#endif // VG_CONFIG_FORCE_AA_OFF
	const PathType::Enum pathType = (PathType::Enum)((_flags & VG_FILL_FLAGS_PATH_TYPE_Msk) >> VG_FILL_FLAGS_PATH_TYPE_Pos);
	const FillRule::Enum fillRule = (FillRule::Enum)((_flags & VG_FILL_FLAGS_FILL_RULE_Msk) >> VG_FILL_FLAGS_FILL_RULE_Pos);

	const Path* path = m_Path;
	const uint32_t numSubPaths = pathGetNumSubPaths(path);
	const SubPath* subPaths = pathGetSubPaths(path);
	Stroker* stroker = m_Stroker;

#if VG_CONFIG_ENABLE_SHAPE_CACHING
	if (hasCache)
	{
		beginCachedCommand(this);
	}
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING

	if (pathType == PathType::Convex)
	{
		for (uint32_t ii = 0; ii < numSubPaths; ++ii)
		{
			const SubPath* subPath = &subPaths[ii];
			if (subPath->m_NumVertices < 3)
			{
				continue;
			}

			const float* vtx = &pathVertices[subPath->m_FirstVertexID << 1];
			const uint32_t numPathVertices = subPath->m_NumVertices;

			Mesh mesh;
			const uint32_t* colors = &col;
			uint32_t numColors = 1;

			if (aa)
			{
				strokerConvexFillAA(stroker, &mesh, vtx, numPathVertices, col);
				colors = mesh.colorBuffer;
				numColors = mesh.numVertices;
			}
			else
			{
				strokerConvexFill(stroker, &mesh, vtx, numPathVertices);
			}

#if VG_CONFIG_ENABLE_SHAPE_CACHING
			if (hasCache)
			{
				addCachedCommand(this, mesh.posBuffer, mesh.numVertices, colors, numColors, mesh.indexBuffer, mesh.numIndices);
			}
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING

			if (recordClipCommands)
			{
				createDrawCommand_Clip(this, mesh.posBuffer, mesh.numVertices, mesh.indexBuffer, mesh.numIndices);
			}
			else
			{
				createDrawCommand_VertexColor(this, mesh.posBuffer, mesh.numVertices, colors, numColors, mesh.indexBuffer, mesh.numIndices);
			}
		}
	}
	else if (pathType == PathType::Concave)
	{
		strokerConcaveFillBegin(stroker);
		for (uint32_t ii = 0; ii < numSubPaths; ++ii)
		{
			const SubPath* subPath = &subPaths[ii];
			if (subPath->m_NumVertices < 3)
			{
				return;
			}

			const float* vtx = &pathVertices[subPath->m_FirstVertexID << 1];
			const uint32_t numPathVertices = subPath->m_NumVertices;
			strokerConcaveFillAddContour(stroker, vtx, numPathVertices);
		}

		Mesh mesh;
		const uint32_t* colors = &col;
		uint32_t numColors = 1;

		bool decomposed = false;
		if (aa)
		{
			decomposed = strokerConcaveFillEndAA(stroker, &mesh, col, fillRule);
			colors = mesh.colorBuffer;
			numColors = mesh.numVertices;
		}
		else
		{
			decomposed = strokerConcaveFillEnd(stroker, &mesh, fillRule);
		}

		VG_WARN(decomposed, "Failed to triangulate concave polygon");
		if (decomposed)
		{
#if VG_CONFIG_ENABLE_SHAPE_CACHING
			if (hasCache)
			{
				addCachedCommand(this, mesh.posBuffer, mesh.numVertices, colors, numColors, mesh.indexBuffer, mesh.numIndices);
			}
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING

			if (recordClipCommands)
			{
				createDrawCommand_Clip(this, mesh.posBuffer, mesh.numVertices, mesh.indexBuffer, mesh.numIndices);
			}
			else
			{
				createDrawCommand_VertexColor(this, mesh.posBuffer, mesh.numVertices, colors, numColors, mesh.indexBuffer, mesh.numIndices);
			}
		}
	}

#if VG_CONFIG_ENABLE_SHAPE_CACHING
	if (hasCache)
	{
		endCachedCommand(this);
	}
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING
}

void Context::fillPathGradient(GradientHandle _gradientHandle, uint32_t _flags)
{
	VG_CHECK(!m_RecordClipCommands, "Only fillPath(Color) is supported inside BeginClip()/EndClip()");
	VG_CHECK(isValid(_gradientHandle), "Invalid gradient handle");
	VG_CHECK(!isLocal(_gradientHandle), "Invalid gradient handle");

#if VG_CONFIG_ENABLE_SHAPE_CACHING
	const bool hasCache = getCommandListCacheStackTop(this) != NULL;
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING

	const float* pathVertices = transformPath(this);

	const PathType::Enum pathType = (PathType::Enum)((_flags & VG_FILL_FLAGS_PATH_TYPE_Msk) >> VG_FILL_FLAGS_PATH_TYPE_Pos);
	const FillRule::Enum fillRule = (FillRule::Enum)((_flags & VG_FILL_FLAGS_FILL_RULE_Msk) >> VG_FILL_FLAGS_FILL_RULE_Pos);
#if VG_CONFIG_FORCE_AA_OFF
	const bool aa = false;
#else
	const bool aa = (bool)((_flags & VG_FILL_FLAGS_AA_Msk) >> VG_FILL_FLAGS_AA_Pos);
#endif // VG_CONFIG_FORCE_AA_OFF

	Stroker* stroker = m_Stroker;
	const Path* path = m_Path;
	const uint32_t numSubPaths = pathGetNumSubPaths(path);
	const SubPath* subPaths = pathGetSubPaths(path);

#if VG_CONFIG_ENABLE_SHAPE_CACHING
	if (hasCache)
	{
		beginCachedCommand(this);
	}
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING


	const State *state = getState(this);
	const Color black = colorSetAlpha(Colors::Black, (uint8_t)(0xff * state->m_GlobalAlpha));
	Mesh mesh;
	const uint32_t* colors = &black;
	uint32_t numColors = 1;

	if (pathType == PathType::Convex)
	{
		for (uint32_t ii = 0; ii < numSubPaths; ++ii)
		{
			const SubPath* subPath = &subPaths[ii];
			if (subPath->m_NumVertices < 3)
			{
				continue;
			}

			const float* vtx = &pathVertices[subPath->m_FirstVertexID << 1];
			const uint32_t numPathVertices = subPath->m_NumVertices;

			if (aa)
			{
				strokerConvexFillAA(stroker, &mesh, vtx, numPathVertices, Colors::Black);
				colors = mesh.colorBuffer;
				numColors = mesh.numVertices;
			}
			else
			{
				strokerConvexFill(stroker, &mesh, vtx, numPathVertices);
			}

#if VG_CONFIG_ENABLE_SHAPE_CACHING
			if (hasCache)
			{
				addCachedCommand(this, mesh.posBuffer, mesh.numVertices, colors, numColors, mesh.indexBuffer, mesh.numIndices);
			}
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING

			createDrawCommand_ColorGradient(this, _gradientHandle, mesh.posBuffer, mesh.numVertices, colors, numColors, mesh.indexBuffer, mesh.numIndices);
		}
	}
	else if (pathType == PathType::Concave)
	{
		strokerConcaveFillBegin(stroker);
		for (uint32_t ii = 0; ii < numSubPaths; ++ii)
		{
			const SubPath* subPath = &subPaths[ii];
			if (subPath->m_NumVertices < 3)
			{
				return;
			}

			const float* vtx = &pathVertices[subPath->m_FirstVertexID << 1];
			const uint32_t numPathVertices = subPath->m_NumVertices;
			strokerConcaveFillAddContour(stroker, vtx, numPathVertices);
		}

		bool decomposed = false;
		if (aa)
		{
			decomposed = strokerConcaveFillEndAA(stroker, &mesh, black, fillRule);
			colors = mesh.colorBuffer;
			numColors = mesh.numVertices;
		}
		else
		{
			decomposed = strokerConcaveFillEnd(stroker, &mesh, fillRule);
		}

		VG_WARN(decomposed, "Failed to triangulate concave polygon");
		if (decomposed)
		{
#if VG_CONFIG_ENABLE_SHAPE_CACHING
			if (hasCache)
			{
				addCachedCommand(this, mesh.posBuffer, mesh.numVertices, colors, numColors, mesh.indexBuffer, mesh.numIndices);
			}
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING

			createDrawCommand_ColorGradient(this, _gradientHandle, mesh.posBuffer, mesh.numVertices, colors, numColors, mesh.indexBuffer, mesh.numIndices);
		}
	}

#if VG_CONFIG_ENABLE_SHAPE_CACHING
	if (hasCache)
	{
		endCachedCommand(this);
	}
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING
}

void Context::fillPathImagePattern(ImagePatternHandle _imgPatternHandle, Color _color, uint32_t _flags)
{
	VG_CHECK(!m_RecordClipCommands, "Only fillPath(Color) is supported inside BeginClip()/EndClip()");
	VG_CHECK(isValid(_imgPatternHandle), "Invalid image pattern handle");
	VG_CHECK(!isLocal(_imgPatternHandle), "Invalid gradient handle");

#if VG_CONFIG_ENABLE_SHAPE_CACHING
	const bool hasCache = getCommandListCacheStackTop(this) != NULL;
#else
	const bool hasCache = false;
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING

	const State* state = getState(this);
	const float globalAlpha = hasCache ? 1.0f : state->m_GlobalAlpha;
	const Color col = colorSetAlpha(_color, (uint8_t)(globalAlpha * colorGetAlpha(_color)));
	if (!hasCache && colorGetAlpha(col) == 0)
	{
		return;
	}

	const PathType::Enum pathType = (PathType::Enum)((_flags & VG_FILL_FLAGS_PATH_TYPE_Msk) >> VG_FILL_FLAGS_PATH_TYPE_Pos);
	const FillRule::Enum fillRule = (FillRule::Enum)((_flags & VG_FILL_FLAGS_FILL_RULE_Msk) >> VG_FILL_FLAGS_FILL_RULE_Pos);
#if VG_CONFIG_FORCE_AA_OFF
	const bool aa = false;
#else
	const bool aa = (bool)((_flags & VG_FILL_FLAGS_AA_Msk) >> VG_FILL_FLAGS_AA_Pos);
#endif // VG_CONFIG_FORCE_AA_OFF

	const float* pathVertices = transformPath(this);

	Stroker* stroker = m_Stroker;
	const Path* path = m_Path;
	const uint32_t numSubPaths = pathGetNumSubPaths(path);
	const SubPath* subPaths = pathGetSubPaths(path);

#if VG_CONFIG_ENABLE_SHAPE_CACHING
	if (hasCache)
	{
		beginCachedCommand(this);
	}
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING

	if (pathType == PathType::Convex)
	{
		for (uint32_t ii = 0; ii < numSubPaths; ++ii)
		{
			const SubPath* subPath = &subPaths[ii];
			if (subPath->m_NumVertices < 3)
			{
				continue;
			}

			const float* vtx = &pathVertices[subPath->m_FirstVertexID << 1];
			const uint32_t numPathVertices = subPath->m_NumVertices;

			Mesh mesh;
			const uint32_t* colors = &col;
			uint32_t numColors = 1;

			if (aa)
			{
				strokerConvexFillAA(stroker, &mesh, vtx, numPathVertices, col);
				colors = mesh.colorBuffer;
				numColors = mesh.numVertices;
			}
			else
			{
				strokerConvexFill(stroker, &mesh, vtx, numPathVertices);
			}

#if VG_CONFIG_ENABLE_SHAPE_CACHING
			if (hasCache)
			{
				addCachedCommand(this, mesh.posBuffer, mesh.numVertices, colors, numColors, mesh.indexBuffer, mesh.numIndices);
			}
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING

			createDrawCommand_ImagePattern(this, _imgPatternHandle, mesh.posBuffer, mesh.numVertices, colors, numColors, mesh.indexBuffer, mesh.numIndices);
		}
	}
	else if (pathType == PathType::Concave)
	{
		strokerConcaveFillBegin(stroker);
		for (uint32_t ii = 0; ii < numSubPaths; ++ii)
		{
			const SubPath* subPath = &subPaths[ii];
			if (subPath->m_NumVertices < 3)
			{
				return;
			}

			const float* vtx = &pathVertices[subPath->m_FirstVertexID << 1];
			const uint32_t numPathVertices = subPath->m_NumVertices;
			strokerConcaveFillAddContour(stroker, vtx, numPathVertices);
		}

		Mesh mesh;
		const uint32_t* colors = &col;
		uint32_t numColors = 1;

		bool decomposed = false;
		if (aa)
		{
			decomposed = strokerConcaveFillEndAA(stroker, &mesh, col, fillRule);
			colors = mesh.colorBuffer;
			numColors = mesh.numVertices;
		}
		else
		{
			decomposed = strokerConcaveFillEnd(stroker, &mesh, fillRule);
		}

		VG_WARN(decomposed, "Failed to triangulate concave polygon");
		if (decomposed)
		{
#if VG_CONFIG_ENABLE_SHAPE_CACHING
			if (hasCache)
			{
				addCachedCommand(this, mesh.posBuffer, mesh.numVertices, colors, numColors, mesh.indexBuffer, mesh.numIndices);
			}
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING

			createDrawCommand_ImagePattern(this, _imgPatternHandle, mesh.posBuffer, mesh.numVertices, colors, numColors, mesh.indexBuffer, mesh.numIndices);
		}
	}

#if VG_CONFIG_ENABLE_SHAPE_CACHING
	if (hasCache)
	{
		endCachedCommand(this);
	}
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING
}

void Context::strokePathColor(Color _color, float _width, uint32_t _flags)
{
	const bool recordClipCommands = m_RecordClipCommands;

#if VG_CONFIG_ENABLE_SHAPE_CACHING
	const bool hasCache = getCommandListCacheStackTop(this) != NULL;
#else
	const bool hasCache = false;
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING

	const State* state = getState(this);
	const float avgScale = state->m_AvgScale;
	const float globalAlpha = hasCache ? 1.0f : state->m_GlobalAlpha;
	const float fringeWidth = m_FringeWidth;

	const float scaledStrokeWidth = ((_flags & StrokeFlags::FixedWidth) != 0) ? _width : bx::clamp<float>(_width * avgScale, 0.0f, 200.0f);
	const bool isThin = scaledStrokeWidth <= fringeWidth;

	const float alphaScale = !isThin ? globalAlpha : globalAlpha * bx::square(bx::clamp<float>(scaledStrokeWidth, 0.0f, fringeWidth));
	const Color col = recordClipCommands ? Colors::Black : colorSetAlpha(_color, (uint8_t)(alphaScale * colorGetAlpha(_color)));
	if (!hasCache && colorGetAlpha(col) == 0)
	{
		return;
	}

	const LineJoin::Enum lineJoin = (LineJoin::Enum)((_flags & VG_STROKE_FLAGS_LINE_JOIN_Msk) >> VG_STROKE_FLAGS_LINE_JOIN_Pos);
	const LineCap::Enum lineCap = (LineCap::Enum)((_flags & VG_STROKE_FLAGS_LINE_CAP_Msk) >> VG_STROKE_FLAGS_LINE_CAP_Pos);
#if VG_CONFIG_FORCE_AA_OFF
	const bool aa = false;
#else
	const bool aa = recordClipCommands
		? false
		: (bool)((_flags & VG_STROKE_FLAGS_AA_Msk) >> VG_STROKE_FLAGS_AA_Pos)
		;
#endif // VG_CONFIG_FORCE_AA_OFF

	const float strokeWidth = isThin ? fringeWidth : scaledStrokeWidth;

	const float* pathVertices = transformPath(this);

	const Path* path = m_Path;
	const uint32_t numSubPaths = pathGetNumSubPaths(path);
	const SubPath* subPaths = pathGetSubPaths(path);
	Stroker* stroker = m_Stroker;

#if VG_CONFIG_ENABLE_SHAPE_CACHING
	if (hasCache)
	{
		beginCachedCommand(this);
	}
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING

	for (uint32_t iSubPath = 0; iSubPath < numSubPaths; ++iSubPath)
	{
		const SubPath* subPath = &subPaths[iSubPath];
		if (subPath->m_NumVertices < 2)
		{
			continue;
		}

		const float* vtx = &pathVertices[subPath->m_FirstVertexID << 1];
		const uint32_t numPathVertices = subPath->m_NumVertices;
		const bool isClosed = subPath->m_IsClosed;

		Mesh mesh;
		const uint32_t* colors = &col;
		uint32_t numColors = 1;
		if (aa)
		{
			if (isThin)
			{
				strokerPolylineStrokeAAThin(stroker, &mesh, vtx, numPathVertices, isClosed, col, lineCap, lineJoin);
			}
			else
			{
				strokerPolylineStrokeAA(stroker, &mesh, vtx, numPathVertices, isClosed, col, strokeWidth, lineCap, lineJoin);
			}

			colors = mesh.colorBuffer;
			numColors = mesh.numVertices;
		}
		else
		{
			strokerPolylineStroke(stroker, &mesh, vtx, numPathVertices, isClosed, strokeWidth, lineCap, lineJoin);
		}

#if VG_CONFIG_ENABLE_SHAPE_CACHING
		if (hasCache)
		{
			addCachedCommand(this, mesh.posBuffer, mesh.numVertices, colors, numColors, mesh.indexBuffer, mesh.numIndices);
		}
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING

		if (recordClipCommands)
		{
			createDrawCommand_Clip(this, mesh.posBuffer, mesh.numVertices, mesh.indexBuffer, mesh.numIndices);
		}
		else
		{
			createDrawCommand_VertexColor(this, mesh.posBuffer, mesh.numVertices, colors, numColors, mesh.indexBuffer, mesh.numIndices);
		}
	}

#if VG_CONFIG_ENABLE_SHAPE_CACHING
	if (hasCache)
	{
		endCachedCommand(this);
	}
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING
}

void Context::strokePathGradient(GradientHandle _gradientHandle, float _width, uint32_t _flags)
{
	VG_CHECK(!m_RecordClipCommands, "Only strokePath(Color) is supported inside BeginClip()/EndClip()");
	VG_CHECK(isValid(_gradientHandle), "Invalid gradient handle");
	VG_CHECK(!isLocal(_gradientHandle), "Invalid gradient handle");

#if VG_CONFIG_ENABLE_SHAPE_CACHING
	const bool hasCache = getCommandListCacheStackTop(this) != NULL;
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING

	const LineJoin::Enum lineJoin = (LineJoin::Enum)((_flags & VG_STROKE_FLAGS_LINE_JOIN_Msk) >> VG_STROKE_FLAGS_LINE_JOIN_Pos);
	const LineCap::Enum lineCap = (LineCap::Enum)((_flags & VG_STROKE_FLAGS_LINE_CAP_Msk) >> VG_STROKE_FLAGS_LINE_CAP_Pos);
#if VG_CONFIG_FORCE_AA_OFF
	const bool aa = false;
#else
	const bool aa = (bool)((_flags & VG_STROKE_FLAGS_AA_Msk) >> VG_STROKE_FLAGS_AA_Pos);
#endif // VG_CONFIG_FORCE_AA_OFF

	const float* pathVertices = transformPath(this);

	const State* state = getState(this);
	const float avgScale = state->m_AvgScale;
	float strokeWidth = ((_flags & StrokeFlags::FixedWidth) != 0) ? _width : bx::clamp<float>(_width * avgScale, 0.0f, 200.0f);
	bool isThin = false;
	if (strokeWidth <= m_FringeWidth)
	{
		strokeWidth = m_FringeWidth;
		isThin = true;
	}

	Stroker* stroker = m_Stroker;
	const Path* path = m_Path;
	const uint32_t numSubPaths = pathGetNumSubPaths(path);
	const SubPath* subPaths = pathGetSubPaths(path);

#if VG_CONFIG_ENABLE_SHAPE_CACHING
	if (hasCache)
	{
		beginCachedCommand(this);
	}
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING

	for (uint32_t iSubPath = 0; iSubPath < numSubPaths; ++iSubPath)
	{
		const SubPath* subPath = &subPaths[iSubPath];
		if (subPath->m_NumVertices < 2)
		{
			continue;
		}

		const float* vtx = &pathVertices[subPath->m_FirstVertexID << 1];
		const uint32_t numPathVertices = subPath->m_NumVertices;
		const bool isClosed = subPath->m_IsClosed;

		Mesh mesh;
		const uint32_t black = colorSetAlpha(Colors::Black, (uint8_t)(0xff * state->m_GlobalAlpha));
		const uint32_t* colors = &black;
		uint32_t numColors = 1;

		if (aa)
		{
			if (isThin)
			{
				strokerPolylineStrokeAAThin(stroker, &mesh, vtx, numPathVertices, isClosed, vg::Colors::Black, lineCap, lineJoin);
			}
			else
			{
				strokerPolylineStrokeAA(stroker, &mesh, vtx, numPathVertices, isClosed, vg::Colors::Black, strokeWidth, lineCap, lineJoin);
			}

			colors = mesh.colorBuffer;
			numColors = mesh.numVertices;
		}
		else
		{
			strokerPolylineStroke(stroker, &mesh, vtx, numPathVertices, isClosed, strokeWidth, lineCap, lineJoin);
		}

#if VG_CONFIG_ENABLE_SHAPE_CACHING
		if (hasCache)
		{
			addCachedCommand(this, mesh.posBuffer, mesh.numVertices, colors, numColors, mesh.indexBuffer, mesh.numIndices);
		}
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING

		createDrawCommand_ColorGradient(this, _gradientHandle, mesh.posBuffer, mesh.numVertices, colors, numColors, mesh.indexBuffer, mesh.numIndices);
	}

#if VG_CONFIG_ENABLE_SHAPE_CACHING
	if (hasCache)
	{
		endCachedCommand(this);
	}
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING
}

void Context::strokePathImagePattern(ImagePatternHandle _imgPatternHandle, Color _color, float _width, uint32_t _flags)
{
	VG_CHECK(!m_RecordClipCommands, "Only strokePath(Color) is supported inside BeginClip()/EndClip()");
	VG_CHECK(isValid(_imgPatternHandle), "Invalid image pattern handle");
	VG_CHECK(!isLocal(_imgPatternHandle), "Invalid gradient handle");

#if VG_CONFIG_ENABLE_SHAPE_CACHING
	const bool hasCache = getCommandListCacheStackTop(this) != NULL;
#else
	const bool hasCache = false;
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING

	const State* state = getState(this);
	const float avgScale = state->m_AvgScale;
	const float globalAlpha = hasCache ? 1.0f : state->m_GlobalAlpha;
	const float fringeWidth = m_FringeWidth;

	const float scaledStrokeWidth = ((_flags & StrokeFlags::FixedWidth) != 0) ? _width : bx::clamp<float>(_width * avgScale, 0.0f, 200.0f);
	const bool isThin = scaledStrokeWidth <= fringeWidth;

	const float alphaScale = isThin ? globalAlpha : globalAlpha * bx::square(bx::clamp<float>(scaledStrokeWidth, 0.0f, fringeWidth));
	const Color col = colorSetAlpha(_color, (uint8_t)(alphaScale * colorGetAlpha(_color)));
	if (!hasCache && colorGetAlpha(col) == 0)
	{
		return;
	}

	const LineJoin::Enum lineJoin = (LineJoin::Enum)((_flags & VG_STROKE_FLAGS_LINE_JOIN_Msk) >> VG_STROKE_FLAGS_LINE_JOIN_Pos);
	const LineCap::Enum lineCap = (LineCap::Enum)((_flags & VG_STROKE_FLAGS_LINE_CAP_Msk) >> VG_STROKE_FLAGS_LINE_CAP_Pos);
#if VG_CONFIG_FORCE_AA_OFF
	const bool aa = false;
#else
	const bool aa = (bool)((_flags & VG_STROKE_FLAGS_AA_Msk) >> VG_STROKE_FLAGS_AA_Pos);
#endif // VG_CONFIG_FORCE_AA_OFF

	const float strokeWidth = isThin ? fringeWidth : scaledStrokeWidth;

	const float* pathVertices = transformPath(this);

	Stroker* stroker = m_Stroker;
	const Path* path = m_Path;
	const uint32_t numSubPaths = pathGetNumSubPaths(path);
	const SubPath* subPaths = pathGetSubPaths(path);

#if VG_CONFIG_ENABLE_SHAPE_CACHING
	if (hasCache)
	{
		beginCachedCommand(this);
	}
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING

	for (uint32_t iSubPath = 0; iSubPath < numSubPaths; ++iSubPath)
	{
		const SubPath* subPath = &subPaths[iSubPath];
		if (subPath->m_NumVertices < 2)
		{
			continue;
		}

		const float* vtx = &pathVertices[subPath->m_FirstVertexID << 1];
		const uint32_t numPathVertices = subPath->m_NumVertices;
		const bool isClosed = subPath->m_IsClosed;

		Mesh mesh;
		const uint32_t* colors = &col;
		uint32_t numColors = 1;

		if (aa)
		{
			if (isThin)
			{
				strokerPolylineStrokeAAThin(stroker, &mesh, vtx, numPathVertices, isClosed, col, lineCap, lineJoin);
			}
			else
			{
				strokerPolylineStrokeAA(stroker, &mesh, vtx, numPathVertices, isClosed, col, strokeWidth, lineCap, lineJoin);
			}

			colors = mesh.colorBuffer;
			numColors = mesh.numVertices;
		}
		else
		{
			strokerPolylineStroke(stroker, &mesh, vtx, numPathVertices, isClosed, strokeWidth, lineCap, lineJoin);
		}

#if VG_CONFIG_ENABLE_SHAPE_CACHING
		if (hasCache)
		{
			addCachedCommand(this, mesh.posBuffer, mesh.numVertices, colors, numColors, mesh.indexBuffer, mesh.numIndices);
		}
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING

		createDrawCommand_ImagePattern(this, _imgPatternHandle, mesh.posBuffer, mesh.numVertices, colors, numColors, mesh.indexBuffer, mesh.numIndices);
	}

#if VG_CONFIG_ENABLE_SHAPE_CACHING
	if (hasCache)
	{
		endCachedCommand(this);
	}
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING
}

void Context::beginClip(ClipRule::Enum _rule)
{
	VG_CHECK(!m_RecordClipCommands, "Already inside beginClip()/endClip() block");

	ClipState* clipState = &m_ClipState;
	const uint32_t nextClipCmdID = m_NumClipCommands;

	clipState->m_Rule = _rule;
	clipState->m_FirstCmdID = nextClipCmdID;
	clipState->m_NumCmds = 0;

	m_RecordClipCommands = true;
	m_ForceNewClipCommand = true;
}

void Context::endClip()
{
	VG_CHECK(m_RecordClipCommands, "Must be called once after beginClip()");

	ClipState* clipState = &m_ClipState;
	const uint32_t nextClipCmdID = m_NumClipCommands;

	clipState->m_NumCmds = nextClipCmdID - clipState->m_FirstCmdID;

	m_RecordClipCommands = false;
	m_ForceNewDrawCommand = true;
}

void Context::resetClip()
{
	VG_CHECK(!m_RecordClipCommands, "Must be called outside beginClip()/endClip() pair.");

	ClipState* clipState = &m_ClipState;

	if (clipState->m_FirstCmdID != ~0u)
	{
		clipState->m_FirstCmdID = ~0u;
		clipState->m_NumCmds = 0;

		m_ForceNewDrawCommand = true;
	}
}

GradientHandle Context::createLinearGradient(float _sx, float _sy, float _ex, float _ey, Color _icol, Color _ocol)
{
	if (m_NextGradientID >= m_Config.maxGradients)
	{
		return VG_INVALID_HANDLE32;
	}

	GradientHandle handle = { (uint16_t)m_NextGradientID++, 0 };

	const float large = 1e5;
	float dx = _ex - _sx;
	float dy = _ey - _sy;
	float dd = bx::sqrt(dx * dx + dy * dy);
	if (dd > 0.0001f)
	{
		dx /= dd;
		dy /= dd;
	}
	else
	{
		dx = 0;
		dy = 1;
	}

	float gradientMatrix[6];
	gradientMatrix[0] = dy;
	gradientMatrix[1] = -dx;
	gradientMatrix[2] = dx;
	gradientMatrix[3] = dy;
	gradientMatrix[4] = _sx - dx * large;
	gradientMatrix[5] = _sy - dy * large;

	const State* state = getState(this);
	const float* stateTransform = state->m_TransformMtx;

	float patternMatrix[6];
	vgutil::multiplyMatrix3(stateTransform, gradientMatrix, patternMatrix);

	float inversePatternMatrix[6];
	vgutil::invertMatrix3(patternMatrix, inversePatternMatrix);

	Gradient* grad = &m_Gradients[handle.idx];
	grad->m_Matrix[0] = inversePatternMatrix[0];
	grad->m_Matrix[1] = inversePatternMatrix[1];
	grad->m_Matrix[2] = 0.0f;
	grad->m_Matrix[3] = inversePatternMatrix[2];
	grad->m_Matrix[4] = inversePatternMatrix[3];
	grad->m_Matrix[5] = 0.0f;
	grad->m_Matrix[6] = inversePatternMatrix[4];
	grad->m_Matrix[7] = inversePatternMatrix[5];
	grad->m_Matrix[8] = 1.0f;
	grad->m_Params[0] = large;
	grad->m_Params[1] = large + dd * 0.5f;
	grad->m_Params[2] = 0.0f;
	grad->m_Params[3] = bx::max<float>(1.0f, dd);
	grad->m_InnerColor[0] = colorGetRed(_icol) / 255.0f;
	grad->m_InnerColor[1] = colorGetGreen(_icol) / 255.0f;
	grad->m_InnerColor[2] = colorGetBlue(_icol) / 255.0f;
	grad->m_InnerColor[3] = colorGetAlpha(_icol) / 255.0f;
	grad->m_OuterColor[0] = colorGetRed(_ocol) / 255.0f;
	grad->m_OuterColor[1] = colorGetGreen(_ocol) / 255.0f;
	grad->m_OuterColor[2] = colorGetBlue(_ocol) / 255.0f;
	grad->m_OuterColor[3] = colorGetAlpha(_ocol) / 255.0f;

	return handle;
}

GradientHandle Context::createBoxGradient(float _x, float _y, float _w, float _h, float _r, float _f, Color _icol, Color _ocol)
{
	if (m_NextGradientID >= m_Config.maxGradients)
	{
		return VG_INVALID_HANDLE32;
	}

	GradientHandle handle = { (uint16_t)m_NextGradientID++, 0 };

	float gradientMatrix[6];
	gradientMatrix[0] = 1.0f;
	gradientMatrix[1] = 0.0f;
	gradientMatrix[2] = 0.0f;
	gradientMatrix[3] = 1.0f;
	gradientMatrix[4] = _x + _w * 0.5f;
	gradientMatrix[5] = _y + _h * 0.5f;

	const State* state = getState(this);
	const float* stateTransform = state->m_TransformMtx;

	float patternMatrix[6];
	vgutil::multiplyMatrix3(stateTransform, gradientMatrix, patternMatrix);

	float inversePatternMatrix[6];
	vgutil::invertMatrix3(patternMatrix, inversePatternMatrix);

	Gradient* grad = &m_Gradients[handle.idx];
	grad->m_Matrix[0] = inversePatternMatrix[0];
	grad->m_Matrix[1] = inversePatternMatrix[1];
	grad->m_Matrix[2] = 0.0f;
	grad->m_Matrix[3] = inversePatternMatrix[2];
	grad->m_Matrix[4] = inversePatternMatrix[3];
	grad->m_Matrix[5] = 0.0f;
	grad->m_Matrix[6] = inversePatternMatrix[4];
	grad->m_Matrix[7] = inversePatternMatrix[5];
	grad->m_Matrix[8] = 1.0f;
	grad->m_Params[0] = _w * 0.5f;
	grad->m_Params[1] = _h * 0.5f;
	grad->m_Params[2] = _r;
	grad->m_Params[3] = bx::max<float>(1.0f, _f);
	grad->m_InnerColor[0] = colorGetRed(_icol) / 255.0f;
	grad->m_InnerColor[1] = colorGetGreen(_icol) / 255.0f;
	grad->m_InnerColor[2] = colorGetBlue(_icol) / 255.0f;
	grad->m_InnerColor[3] = colorGetAlpha(_icol) / 255.0f;
	grad->m_OuterColor[0] = colorGetRed(_ocol) / 255.0f;
	grad->m_OuterColor[1] = colorGetGreen(_ocol) / 255.0f;
	grad->m_OuterColor[2] = colorGetBlue(_ocol) / 255.0f;
	grad->m_OuterColor[3] = colorGetAlpha(_ocol) / 255.0f;

	return handle;
}

GradientHandle Context::createRadialGradient(float _cx, float _cy, float _inr, float _outr, Color _icol, Color _ocol)
{
	if (m_NextGradientID >= m_Config.maxGradients)
	{
		return VG_INVALID_HANDLE32;
	}

	GradientHandle handle = { (uint16_t)m_NextGradientID++, 0 };

	float gradientMatrix[6];
	gradientMatrix[0] = 1.0f;
	gradientMatrix[1] = 0.0f;
	gradientMatrix[2] = 0.0f;
	gradientMatrix[3] = 1.0f;
	gradientMatrix[4] = _cx;
	gradientMatrix[5] = _cy;

	const State* state = getState(this);
	const float* stateTransform = state->m_TransformMtx;

	float patternMatrix[6];
	vgutil::multiplyMatrix3(stateTransform, gradientMatrix, patternMatrix);

	float inversePatternMatrix[6];
	vgutil::invertMatrix3(patternMatrix, inversePatternMatrix);

	const float rr = (_inr + _outr) * 0.5f;
	const float ff = (_outr - _inr);

	Gradient* grad = &m_Gradients[handle.idx];
	grad->m_Matrix[0] = inversePatternMatrix[0];
	grad->m_Matrix[1] = inversePatternMatrix[1];
	grad->m_Matrix[2] = 0.0f;
	grad->m_Matrix[3] = inversePatternMatrix[2];
	grad->m_Matrix[4] = inversePatternMatrix[3];
	grad->m_Matrix[5] = 0.0f;
	grad->m_Matrix[6] = inversePatternMatrix[4];
	grad->m_Matrix[7] = inversePatternMatrix[5];
	grad->m_Matrix[8] = 1.0f;
	grad->m_Params[0] = rr;
	grad->m_Params[1] = rr;
	grad->m_Params[2] = rr;
	grad->m_Params[3] = bx::max<float>(1.0f, ff);
	grad->m_InnerColor[0] = colorGetRed(_icol) / 255.0f;
	grad->m_InnerColor[1] = colorGetGreen(_icol) / 255.0f;
	grad->m_InnerColor[2] = colorGetBlue(_icol) / 255.0f;
	grad->m_InnerColor[3] = colorGetAlpha(_icol) / 255.0f;
	grad->m_OuterColor[0] = colorGetRed(_ocol) / 255.0f;
	grad->m_OuterColor[1] = colorGetGreen(_ocol) / 255.0f;
	grad->m_OuterColor[2] = colorGetBlue(_ocol) / 255.0f;
	grad->m_OuterColor[3] = colorGetAlpha(_ocol) / 255.0f;

	return handle;
}

ImagePatternHandle Context::createImagePattern(float _cx, float _cy, float _w, float _h, float _angle, ImageHandle _image)
{
	if (!isValid(_image))
	{
		return VG_INVALID_HANDLE32;
	}

	if (m_NextImagePatternID >= m_Config.maxImagePatterns)
	{
		return VG_INVALID_HANDLE32;
	}

	ImagePatternHandle handle = { (uint16_t)m_NextImagePatternID++, 0 };

	const float cs = bx::cos(_angle);
	const float sn = bx::sin(_angle);

	float mtx[6];
	mtx[0] = cs;
	mtx[1] = sn;
	mtx[2] = -sn;
	mtx[3] = cs;
	mtx[4] = _cx;
	mtx[5] = _cy;

	const State* state = getState(this);
	const float* stateTransform = state->m_TransformMtx;

	float patternMatrix[6];
	vgutil::multiplyMatrix3(stateTransform, mtx, patternMatrix);

	float inversePatternMatrix[6];
	vgutil::invertMatrix3(patternMatrix, inversePatternMatrix);

	inversePatternMatrix[0] /= _w;
	inversePatternMatrix[1] /= _h;
	inversePatternMatrix[2] /= _w;
	inversePatternMatrix[3] /= _h;
	inversePatternMatrix[4] /= _w;
	inversePatternMatrix[5] /= _h;

	ImagePattern* pattern = &m_ImagePatterns[handle.idx];
	pattern->m_Matrix[0] = inversePatternMatrix[0];
	pattern->m_Matrix[1] = inversePatternMatrix[1];
	pattern->m_Matrix[2] = 0.0f;
	pattern->m_Matrix[3] = inversePatternMatrix[2];
	pattern->m_Matrix[4] = inversePatternMatrix[3];
	pattern->m_Matrix[5] = 0.0f;
	pattern->m_Matrix[6] = inversePatternMatrix[4];
	pattern->m_Matrix[7] = inversePatternMatrix[5];
	pattern->m_Matrix[8] = 1.0f;
	pattern->m_ImageHandle = _image;

	return handle;
}

void Context::pushState()
{
	VG_CHECK(m_StateStackTop < (uint32_t)(m_Config.maxStateStackSize - 1), "State stack overflow");

	const uint32_t top = m_StateStackTop;
	const State* curState = &m_StateStack[top];
	State* newState = &m_StateStack[top + 1];
	bx::memCopy(newState, curState, sizeof(State));
	++m_StateStackTop;
}

void Context::popState()
{
	VG_CHECK(m_StateStackTop > 0, "State stack underflow");
	--m_StateStackTop;

	// If the new state has a different scissor rect than the last draw command
	// force creating a new command.
	const uint32_t numDrawCommands = m_NumDrawCommands;
	if (numDrawCommands != 0)
	{
		const State* state = getState(this);
		const DrawCommand* lastDrawCommand = &m_DrawCommands[numDrawCommands - 1];
		const uint16_t* lastScissor = &lastDrawCommand->m_ScissorRect[0];
		const float* stateScissor = &state->m_ScissorRect[0];
		if (lastScissor[0] != (uint16_t)stateScissor[0] ||
			lastScissor[1] != (uint16_t)stateScissor[1] ||
			lastScissor[2] != (uint16_t)stateScissor[2] ||
			lastScissor[3] != (uint16_t)stateScissor[3])
			{
			m_ForceNewDrawCommand = true;
			m_ForceNewClipCommand = true;
		}
	}
}

void Context::resetScissor()
{
	State* state = getState(this);
	state->m_ScissorRect[0] = state->m_ScissorRect[1] = 0.0f;
	state->m_ScissorRect[2] = (float)m_CanvasWidth;
	state->m_ScissorRect[3] = (float)m_CanvasHeight;
	m_ForceNewDrawCommand = true;
	m_ForceNewClipCommand = true;
}

void Context::setScissor(float _x, float _y, float _w, float _h)
{
	State* state = getState(this);
	const float* stateTransform = state->m_TransformMtx;
	const float canvasWidth = (float)m_CanvasWidth;
	const float canvasHeight = (float)m_CanvasHeight;

	float pos[2], size[2];
	vgutil::transformPos2D(_x, _y, stateTransform, &pos[0]);
	vgutil::transformVec2D(_w, _h, stateTransform, &size[0]);

	const float minx = bx::clamp<float>(pos[0], 0.0f, canvasWidth);
	const float miny = bx::clamp<float>(pos[1], 0.0f, canvasHeight);
	const float maxx = bx::clamp<float>(pos[0] + size[0], 0.0f, canvasWidth);
	const float maxy = bx::clamp<float>(pos[1] + size[1], 0.0f, canvasHeight);

	state->m_ScissorRect[0] = minx;
	state->m_ScissorRect[1] = miny;
	state->m_ScissorRect[2] = maxx - minx;
	state->m_ScissorRect[3] = maxy - miny;
	m_ForceNewDrawCommand = true;
	m_ForceNewClipCommand = true;
}

bool Context::intersectScissor(float _x, float _y, float _w, float _h)
{
	State* state = getState(this);
	const float* stateTransform = state->m_TransformMtx;
	const float* scissorRect = state->m_ScissorRect;

	float pos[2], size[2];
	vgutil::transformPos2D(_x, _y, stateTransform, &pos[0]);
	vgutil::transformVec2D(_w, _h, stateTransform, &size[0]);

	const float minx = bx::max<float>(pos[0], scissorRect[0]);
	const float miny = bx::max<float>(pos[1], scissorRect[1]);
	const float maxx = bx::min<float>(pos[0] + size[0], scissorRect[0] + scissorRect[2]);
	const float maxy = bx::min<float>(pos[1] + size[1], scissorRect[1] + scissorRect[3]);

	const float newRectWidth = bx::max<float>(0.0f, maxx - minx);
	const float newRectHeight = bx::max<float>(0.0f, maxy - miny);

	state->m_ScissorRect[0] = minx;
	state->m_ScissorRect[1] = miny;
	state->m_ScissorRect[2] = newRectWidth;
	state->m_ScissorRect[3] = newRectHeight;

	m_ForceNewDrawCommand = true;
	m_ForceNewClipCommand = true;

	return newRectWidth >= 1.0f && newRectHeight >= 1.0f;
}

void Context::transformIdentity()
{
	State* state = getState(this);
	state->m_TransformMtx[0] = 1.0f;
	state->m_TransformMtx[1] = 0.0f;
	state->m_TransformMtx[2] = 0.0f;
	state->m_TransformMtx[3] = 1.0f;
	state->m_TransformMtx[4] = 0.0f;
	state->m_TransformMtx[5] = 0.0f;

	updateState(state);
}

void Context::transformScale(float _x, float _y)
{
	State* state = getState(this);
	state->m_TransformMtx[0] = _x * state->m_TransformMtx[0];
	state->m_TransformMtx[1] = _x * state->m_TransformMtx[1];
	state->m_TransformMtx[2] = _y * state->m_TransformMtx[2];
	state->m_TransformMtx[3] = _y * state->m_TransformMtx[3];

	updateState(state);
}

void Context::transformTranslate(float _x, float _y)
{
	State* state = getState(this);
	state->m_TransformMtx[4] += state->m_TransformMtx[0] * _x + state->m_TransformMtx[2] * _y;
	state->m_TransformMtx[5] += state->m_TransformMtx[1] * _x + state->m_TransformMtx[3] * _y;

	updateState(state);
}

void Context::transformRotate(float _ang_rad)
{
	const float cc = bx::cos(_ang_rad);
	const float ss = bx::sin(_ang_rad);

	State* state = getState(this);
	const float* stateTransform = state->m_TransformMtx;

	float mtx[6];
	mtx[0] = cc * stateTransform[0] + ss * stateTransform[2];
	mtx[1] = cc * stateTransform[1] + ss * stateTransform[3];
	mtx[2] = -ss * stateTransform[0] + cc * stateTransform[2];
	mtx[3] = -ss * stateTransform[1] + cc * stateTransform[3];
	mtx[4] = stateTransform[4];
	mtx[5] = stateTransform[5];
	bx::memCopy(state->m_TransformMtx, mtx, sizeof(float) * 6);

	updateState(state);
}

void Context::transformMult(const float* _mtx, TransformOrder::Enum _order)
{
	State* state = getState(this);
	const float* stateTransform = state->m_TransformMtx;

	float res[6];
	if (_order == TransformOrder::Post)
	{
		vgutil::multiplyMatrix3(stateTransform, _mtx, res);
	}
	else
	{
		VG_CHECK(_order == TransformOrder::Pre, "Unknown TransformOrder::Enum");
		vgutil::multiplyMatrix3(_mtx, stateTransform, res);
	}

	bx::memCopy(state->m_TransformMtx, res, sizeof(float) * 6);

	updateState(state);
}

void Context::setViewBox(float _x, float _y, float _w, float _h)
{
	const float scaleX = (float)m_CanvasWidth / _w;
	const float scaleY = (float)m_CanvasHeight / _h;

	State* state = getState(this);
	float* stateTransform = &state->m_TransformMtx[0];

	// transformScale(scaleX, scaleY);
	stateTransform[0] = scaleX * stateTransform[0];
	stateTransform[1] = scaleX * stateTransform[1];
	stateTransform[2] = scaleY * stateTransform[2];
	stateTransform[3] = scaleY * stateTransform[3];

	// transformTranslate(-x, -y);
	stateTransform[4] -= stateTransform[0] * _x + stateTransform[2] * _y;
	stateTransform[5] -= stateTransform[1] * _x + stateTransform[3] * _y;

	updateState(state);
}

void Context::setGlobalAlpha(float _alpha)
{
	State* state = getState(this);
	state->m_GlobalAlpha = _alpha;
}

void Context::indexedTriList(const float* _pos, const uv_t* _uv, uint32_t _numVertices, const Color* _colors, uint32_t _numColors, const uint16_t* _indices, uint32_t _numIndices, ImageHandle _img)
{
	if (!isValid(_img))
	{
		_img = fsGetFontAtlasImage(m_FontSystem);
	}

	const State* state = getState(this);
	const float* stateTransform = state->m_TransformMtx;

	DrawCommand* cmd = allocDrawCommand(this, _numVertices, _numIndices, DrawCommand::Type::Textured, _img.idx);

	// Vertex buffer
	VertexBuffer* vb = &m_VertexBuffers[cmd->m_VertexBufferID];
	const uint32_t vbOffset = cmd->m_FirstVertexID + cmd->m_NumVertices;

	float* dstPos = &vb->m_Pos[vbOffset << 1];
	vgutil::batchTransformPositions(_pos, _numVertices, dstPos, stateTransform);

	uv_t* dstUV = &vb->m_UV[vbOffset << 1];
	if (_uv)
	{
		bx::memCopy(dstUV, _uv, sizeof(uv_t) * 2 * _numVertices);
	}
	else
	{
		const uv_t* whiteRectUV = fsGetWhitePixelUV(m_FontSystem);

#if VG_CONFIG_UV_INT16
		vgutil::memset32(dstUV, _numVertices, &whiteRectUV[0]);
#else
		vgutil::memset64(dstUV, _numVertices, &whiteRectUV[0]);
#endif // VG_CONFIG_UV_INT16
	}

	uint32_t* dstColor = &vb->m_Color[vbOffset];
	if (_numColors == _numVertices)
	{
		bx::memCopy(dstColor, _colors, sizeof(uint32_t) * _numVertices);
	}
	else
	{
		VG_CHECK(_numColors == 1, "Invalid size of color array passed.");
		vgutil::memset32(dstColor, _numVertices, _colors);
	}

	// Index buffer
	IndexBuffer* ib = &m_IndexBuffers[m_ActiveIndexBufferID];
	uint16_t* dstIndex = &ib->m_Indices[cmd->m_FirstIndexID + cmd->m_NumIndices];
	vgutil::batchTransformDrawIndices(_indices, _numIndices, dstIndex, (uint16_t)cmd->m_NumVertices);

	cmd->m_NumVertices += _numVertices;
	cmd->m_NumIndices += _numIndices;
}

void Context::text(const TextConfig& _cfg, float _x, float _y, const char* _str, const char* _end)
{
	const State* state = getState(this);
	const float scale = state->m_FontScale * m_DevicePixelRatio;

	const uint32_t cc = colorSetAlpha(_cfg.color, (uint8_t)(state->m_GlobalAlpha * colorGetAlpha(_cfg.color)));
	if (colorGetAlpha(cc) == 0)
	{
		return;
	}

	const float scaledFontSize = _cfg.fontSize * scale;
	const uint32_t len = _end
		? (uint32_t)(_end - _str)
		: bx::strLen(_str)
		;

	const TextConfig newCfg = makeTextConfig(_cfg.fontHandle, scaledFontSize, _cfg.alignment, cc, _cfg.blur * scale, _cfg.spacing * scale);

	TextMesh mesh;
	bx::memSet(&mesh, 0, sizeof(TextMesh));
	if (!fsText(m_FontSystem, this, newCfg, _str, len, TextFlags::BuildBitmaps, &mesh))
	{
		return;
	}

	pushState();
	transformTranslate(_x + mesh.m_Alignment[0] / scale, _y + mesh.m_Alignment[1] / scale);
	renderTextQuads(this, mesh.m_Quads, mesh.m_Size, newCfg.color, fsGetFontAtlasImage(m_FontSystem));
	popState();
}

void Context::textBox(const TextConfig& _cfg, float _x, float _y, float _breakWidth, const char* _str, const char* _end, uint32_t _textBreakFlags)
{
	_end = _end
		? _end
		: _str + bx::strLen(_str)
		;

	const float lineHeight = fsGetLineHeight(m_FontSystem, _cfg);
	const TextAlignHor::Enum halign = (TextAlignHor::Enum)((_cfg.alignment & VG_TEXT_ALIGN_HOR_Msk) >> VG_TEXT_ALIGN_HOR_Pos);
	const TextAlignVer::Enum valign = (TextAlignVer::Enum)((_cfg.alignment & VG_TEXT_ALIGN_VER_Msk) >> VG_TEXT_ALIGN_VER_Pos);

	const TextConfig newCfg = makeTextConfig(_cfg.fontHandle, _cfg.fontSize, VG_TEXT_ALIGN(vg::TextAlignHor::Left, valign), _cfg.color, _cfg.blur, _cfg.spacing);

	TextRow rows[4];
	uint32_t numRows = 0;
	while ((numRows = fsTextBreakLines(m_FontSystem, _cfg, _str, _end, _breakWidth, &rows[0], BX_COUNTOF(rows), _textBreakFlags)) != 0)
	{
		for (uint32_t ii = 0; ii < numRows; ++ii)
		{
			if (halign == TextAlignHor::Left)
			{
				text(newCfg, _x, _y, rows[ii].start, rows[ii].end);
			}
			else if (halign == TextAlignHor::Center)
			{
				text(newCfg, _x + (_breakWidth - rows[ii].width) * 0.5f, _y, rows[ii].start, rows[ii].end);
			}
			else if (halign == TextAlignHor::Right)
			{
				text(newCfg, _x + _breakWidth - rows[ii].width, _y, rows[ii].start, rows[ii].end);
			}

			_y += lineHeight;
		}

		_str = rows[numRows - 1].next;
	}
}

void Context::submitCommandList(CommandListHandle _handle)
{
	VG_CHECK(isCommandListHandleValid(this, _handle), "Invalid command list handle");
	CommandList* cl = &m_CmdLists[_handle.idx];

	if (m_SubmitCmdListRecursionDepth >= m_Config.maxCommandListDepth)
	{
		VG_CHECK(false, "SubmitCommandList recursion depth limit reached.");
		return;
	}
	++m_SubmitCmdListRecursionDepth;

#if VG_CONFIG_ENABLE_SHAPE_CACHING
	CommandListCache* clCache = clGetCache(this, cl);
	if(clCache)
	{
		const State* state = getState(this);

		const float cachedScale = clCache->m_AvgScale;
		const float stateScale = state->m_AvgScale;

		const bool scaleInvariant = 0 != (cl->m_Flags & CommandListFlags::CacheScaleInvariant);
		const bool reuse = scaleInvariant ? (cachedScale != 0.0f) : (cachedScale == stateScale);

		if (reuse)
		{
			clCacheRender(this, cl);
			--m_SubmitCmdListRecursionDepth;
			return;
		}
		else
		{
			clCacheReset(this, clCache);

			clCache->m_AvgScale = stateScale;
		}
	}
#else
	CommandListCache* clCache = NULL;
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING

	// Don't cull commands during caching.
	const uint32_t clFlags = cl->m_Flags;
	const bool cullCmds = !clCache && ((clFlags & CommandListFlags::AllowCommandCulling) != 0);

	const uint16_t firstGradientID = (uint16_t)m_NextGradientID;
	const uint16_t firstImagePatternID = (uint16_t)m_NextImagePatternID;
	VG_CHECK(firstGradientID + cl->m_NumGradients <= m_Config.maxGradients, "Not enough free gradients for command list. Increase Init::maxGradients");
	VG_CHECK(firstImagePatternID + cl->m_NumImagePatterns <= m_Config.maxImagePatterns, "Not enough free image patterns for command list. Increase Init::maxImagePatterns");

	const uint8_t* cmd = cl->m_CommandBuffer;
	const uint8_t* cmdListEnd = cl->m_CommandBuffer + cl->m_CommandBufferPos;
	if (cmd == cmdListEnd)
	{
		--m_SubmitCmdListRecursionDepth;
		return;
	}

	const char* stringBuffer = cl->m_StringBuffer;

#if VG_CONFIG_ENABLE_SHAPE_CACHING
	pushCommandListCache(this, clCache);
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING

	bool skipCmds = false;
#if VG_CONFIG_COMMAND_LIST_PRESERVE_STATE
	pushState();
#endif // VG_CONFIG_COMMAND_LIST_PRESERVE_STATE

	while (cmd < cmdListEnd)
	{
		const CommandHeader* cmdHeader = (CommandHeader*)cmd;
		cmd += kAlignedCommandHeaderSize;

		const uint8_t* nextCmd = cmd + cmdHeader->m_Size;

		if (skipCmds && cmdHeader->m_Type >= CommandType::FirstStrokerCommand && cmdHeader->m_Type <= CommandType::LastStrokerCommand)
		{
			cmd = nextCmd;
			continue;
		}

		switch (cmdHeader->m_Type)
		{
		case CommandType::BeginPath:
		{
			beginPath();
		} break;
		case CommandType::ClosePath:
		{
			closePath();
		} break;
		case CommandType::MoveTo:
		{
			const float* coords = (float*)cmd;
			cmd += sizeof(float) * 2;
			moveTo(coords[0], coords[1]);
		} break;
		case CommandType::LineTo:
		{
			const float* coords = (float*)cmd;
			cmd += sizeof(float) * 2;
			lineTo(coords[0], coords[1]);
		} break;
		case CommandType::CubicTo:
		{
			const float* coords = (float*)cmd;
			cmd += sizeof(float) * 6;
			cubicTo(coords[0], coords[1], coords[2], coords[3], coords[4], coords[5]);
		} break;
		case CommandType::QuadraticTo:
		{
			const float* coords = (float*)cmd;
			cmd += sizeof(float) * 4;
			quadraticTo(coords[0], coords[1], coords[2], coords[3]);
		} break;
		case CommandType::Arc:
		{
			const float* coords = (float*)cmd;
			cmd += sizeof(float) * 5;
			const Winding::Enum dir = CMD_READ(cmd, Winding::Enum);
			arc(coords[0], coords[1], coords[2], coords[3], coords[4], dir);
		} break;
		case CommandType::ArcTo:
		{
			const float* coords = (float*)cmd;
			cmd += sizeof(float) * 5;
			arcTo(coords[0], coords[1], coords[2], coords[3], coords[4]);
		} break;
		case CommandType::Rect:
		{
			const float* coords = (float*)cmd;
			cmd += sizeof(float) * 4;
			rect(coords[0], coords[1], coords[2], coords[3]);
		} break;
		case CommandType::RoundedRect:
		{
			const float* coords = (float*)cmd;
			cmd += sizeof(float) * 5;
			roundedRect(coords[0], coords[1], coords[2], coords[3], coords[4]);
		} break;
		case CommandType::RoundedRectVarying:
		{
			const float* coords = (float*)cmd;
			cmd += sizeof(float) * 8;
			roundedRectVarying(coords[0], coords[1], coords[2], coords[3], coords[4], coords[5], coords[6], coords[7]);
		} break;
		case CommandType::Circle:
		{
			const float* coords = (float*)cmd;
			cmd += sizeof(float) * 3;
			circle(coords[0], coords[1], coords[2]);
		} break;
		case CommandType::Ellipse:
		{
			const float* coords = (float*)cmd;
			cmd += sizeof(float) * 4;
			ellipse(coords[0], coords[1], coords[2], coords[3]);
		} break;
		case CommandType::Polyline:
		{
			const uint32_t numPoints = CMD_READ(cmd, uint32_t);
			const float* coords = (float*)cmd;
			cmd += sizeof(float) * 2 * numPoints;
			polyline(coords, numPoints);
		} break;
		case CommandType::FillPathColor:
		{
			const uint32_t flags = CMD_READ(cmd, uint32_t);
			const Color color = CMD_READ(cmd, Color);

			fillPathColor(color, flags);
		} break;
		case CommandType::FillPathGradient:
		{
			const uint32_t flags = CMD_READ(cmd, uint32_t);
			const uint16_t gradientHandle = CMD_READ(cmd, uint16_t);
			const uint16_t gradientFlags = CMD_READ(cmd, uint16_t);

			const GradientHandle gradient = { isLocal(gradientFlags) ? (uint16_t)(gradientHandle + firstGradientID) : gradientHandle, 0 };
			fillPathGradient(gradient, flags);
		} break;
		case CommandType::FillPathImagePattern:
		{
			const uint32_t flags = CMD_READ(cmd, uint32_t);
			const Color color = CMD_READ(cmd, Color);
			const uint16_t imgPatternHandle = CMD_READ(cmd, uint16_t);
			const uint16_t imgPatternFlags = CMD_READ(cmd, uint16_t);

			const ImagePatternHandle imgPattern = { isLocal(imgPatternFlags) ? (uint16_t)(imgPatternHandle + firstImagePatternID) : imgPatternHandle, 0 };
			fillPathImagePattern(imgPattern, color, flags);
		} break;
		case CommandType::StrokePathColor:
		{
			const float width = CMD_READ(cmd, float);
			const uint32_t flags = CMD_READ(cmd, uint32_t);
			const Color color = CMD_READ(cmd, Color);

			strokePathColor(color, width, flags);
		} break;
		case CommandType::StrokePathGradient:
		{
			const float width = CMD_READ(cmd, float);
			const uint32_t flags = CMD_READ(cmd, uint32_t);
			const uint16_t gradientHandle = CMD_READ(cmd, uint16_t);
			const uint16_t gradientFlags = CMD_READ(cmd, uint16_t);

			const GradientHandle gradient = { isLocal(gradientFlags) ? (uint16_t)(gradientHandle + firstGradientID) : gradientHandle, 0 };
			strokePathGradient(gradient, width, flags);
		} break;
		case CommandType::StrokePathImagePattern:
		{
			const float width = CMD_READ(cmd, float);
			const uint32_t flags = CMD_READ(cmd, uint32_t);
			const Color color = CMD_READ(cmd, Color);
			const uint16_t imgPatternHandle = CMD_READ(cmd, uint16_t);
			const uint16_t imgPatternFlags = CMD_READ(cmd, uint16_t);

			const ImagePatternHandle imgPattern = { isLocal(imgPatternFlags) ? (uint16_t)(imgPatternHandle + firstImagePatternID) : imgPatternHandle, 0 };
			strokePathImagePattern(imgPattern, color, width, flags);
		} break;
		case CommandType::IndexedTriList:
		{
			const uint32_t numVertices = CMD_READ(cmd, uint32_t);
			const float* positions = (float*)cmd;
			cmd += sizeof(float) * 2 * numVertices;
			const uint32_t numUVs = CMD_READ(cmd, uint32_t);
			const uv_t* uv = (uv_t*)cmd;
			cmd += sizeof(uv_t) * 2 * numUVs;
			const uint32_t numColors = CMD_READ(cmd, uint32_t);
			const Color* colors = (Color*)cmd;
			cmd += sizeof(Color) * numColors;
			const uint32_t numIndices = CMD_READ(cmd, uint32_t);
			const uint16_t* indices = (uint16_t*)cmd;
			cmd += sizeof(uint16_t) * numIndices;
			const uint16_t imgHandle = CMD_READ(cmd, uint16_t);

			indexedTriList(positions, numUVs ? uv : NULL, numVertices, colors, numColors, indices, numIndices, { imgHandle });
		} break;
		case CommandType::CreateLinearGradient:
		{
			const float* params = (float*)cmd;
			cmd += sizeof(float) * 4;
			const Color* colors = (Color*)cmd;
			cmd += sizeof(Color) * 2;
			createLinearGradient(params[0], params[1], params[2], params[3], colors[0], colors[1]);
		} break;
		case CommandType::CreateBoxGradient:
		{
			const float* params = (float*)cmd;
			cmd += sizeof(float) * 6;
			const Color* colors = (Color*)cmd;
			cmd += sizeof(Color) * 2;
			createBoxGradient(params[0], params[1], params[2], params[3], params[4], params[5], colors[0], colors[1]);
		} break;
		case CommandType::CreateRadialGradient:
		{
			const float* params = (float*)cmd;
			cmd += sizeof(float) * 4;
			const Color* colors = (Color*)cmd;
			cmd += sizeof(Color) * 2;
			createRadialGradient(params[0], params[1], params[2], params[3], colors[0], colors[1]);
		} break;
		case CommandType::CreateImagePattern:
		{
			const float* params = (float*)cmd;
			cmd += sizeof(float) * 5;
			const ImageHandle img = CMD_READ(cmd, ImageHandle);
			createImagePattern(params[0], params[1], params[2], params[3], params[4], img);
		} break;
		case CommandType::Text:
		{
			const TextConfig* txtCfg = (TextConfig*)cmd;
			cmd += sizeof(TextConfig);
			const float* coords = (float*)cmd;
			cmd += sizeof(float) * 2;
			const uint32_t stringOffset = CMD_READ(cmd, uint32_t);
			const uint32_t stringLen = CMD_READ(cmd, uint32_t);
			VG_CHECK(stringOffset < cl->m_StringBufferPos, "Invalid string offset");
			VG_CHECK(stringOffset + stringLen <= cl->m_StringBufferPos, "Invalid string length");

			const char* str = stringBuffer + stringOffset;
			const char* end = str + stringLen;
			text(*txtCfg, coords[0], coords[1], str, end);
		} break;
		case CommandType::TextBox:
		{
			const TextConfig* txtCfg = (TextConfig*)cmd;
			cmd += sizeof(TextConfig);
			const float* coords = (float*)cmd;
			cmd += sizeof(float) * 3; // x, y, breakWidth
			const uint32_t stringOffset = CMD_READ(cmd, uint32_t);
			const uint32_t stringLen = CMD_READ(cmd, uint32_t);
			const uint32_t textboxFlags = CMD_READ(cmd, uint32_t);
			VG_CHECK(stringOffset < cl->m_StringBufferPos, "Invalid string offset");
			VG_CHECK(stringOffset + stringLen <= cl->m_StringBufferPos, "Invalid string length");

			const char* str = stringBuffer + stringOffset;
			const char* end = str + stringLen;
			textBox(*txtCfg, coords[0], coords[1], coords[2], str, end, textboxFlags);
		} break;
		case CommandType::ResetScissor:
		{
			resetScissor();
			skipCmds = false;
		} break;
		case CommandType::SetScissor:
		{
			const float* rect = (float*)cmd;
			cmd += sizeof(float) * 4;
			setScissor(rect[0], rect[1], rect[2], rect[3]);

			if (cullCmds)
			{
				const State* state = getState(this);
				const float* scissorRect = &state->m_ScissorRect[0];
				skipCmds = (scissorRect[2] < 1.0f) || (scissorRect[3] < 1.0f);
			}
		} break;
		case CommandType::IntersectScissor:
		{
			const float* rect = (float*)cmd;
			cmd += sizeof(float) * 4;

			const bool zeroRect = !intersectScissor(rect[0], rect[1], rect[2], rect[3]);
			if (cullCmds)
			{
				skipCmds = zeroRect;
			}
		} break;
		case CommandType::PushState:
		{
			pushState();
		} break;
		case CommandType::PopState:
		{
			popState();
			if (cullCmds)
			{
				const State* state = getState(this);
				const float* scissorRect = &state->m_ScissorRect[0];
				skipCmds = (scissorRect[2] < 1.0f) || (scissorRect[3] < 1.0f);
			}
		} break;
		case CommandType::TransformIdentity:
		{
			transformIdentity();
		} break;
		case CommandType::TransformRotate:
		{
			const float ang_rad = CMD_READ(cmd, float);
			transformRotate(ang_rad);
		} break;
		case CommandType::TransformTranslate:
		{
			const float* coords = (float*)cmd;
			cmd += sizeof(float) * 2;
			transformTranslate(coords[0], coords[1]);
		} break;
		case CommandType::TransformScale:
		{
			const float* coords = (float*)cmd;
			cmd += sizeof(float) * 2;
			transformScale(coords[0], coords[1]);
		} break;
		case CommandType::TransformMult:
		{
			const float* mtx = (float*)cmd;
			cmd += sizeof(float) * 6;
			const TransformOrder::Enum order = CMD_READ(cmd, TransformOrder::Enum);
			transformMult(mtx, order);
		} break;
		case CommandType::SetViewBox:
		{
			const float* viewBox = (float*)cmd;
			cmd += sizeof(float) * 4;
			setViewBox(viewBox[0], viewBox[1], viewBox[2], viewBox[3]);
		} break;
		case CommandType::SetGlobalAlpha:
		{
			const float alpha = CMD_READ(cmd, float);
			setGlobalAlpha(alpha);
		} break;
		case CommandType::BeginClip:
		{
			const ClipRule::Enum rule = CMD_READ(cmd, ClipRule::Enum);
			beginClip(rule);
		} break;
		case CommandType::EndClip:
		{
			endClip();
		} break;
		case CommandType::ResetClip:
		{
			resetClip();
		} break;
		case CommandType::SubmitCommandList:
		{
			const uint16_t cmdListID = CMD_READ(cmd, uint16_t);
			const CommandListHandle cmdListHandle = { cmdListID };

			if (isCommandListHandleValid(this, cmdListHandle))
			{
				submitCommandList(cmdListHandle);
			}
		} break;
		default:
		{
			VG_CHECK(false, "Unknown command");
		} break;
		}

		cmd = nextCmd;
	}

#if VG_CONFIG_COMMAND_LIST_PRESERVE_STATE
	popState();
	resetClip();
#endif // VG_CONFIG_COMMAND_LIST_PRESERVE_STATE

#if VG_CONFIG_ENABLE_SHAPE_CACHING
	popCommandListCache(this);
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING

	--m_SubmitCmdListRecursionDepth;
}

// Internal
static State* getState(Context* _ctx)
{
	const uint32_t top = _ctx->m_StateStackTop;
	return &_ctx->m_StateStack[top];
}

static void updateState(State* _state)
{
	const float* stateTransform = _state->m_TransformMtx;

	const float sx = bx::sqrt(stateTransform[0] * stateTransform[0] + stateTransform[2] * stateTransform[2]);
	const float sy = bx::sqrt(stateTransform[1] * stateTransform[1] + stateTransform[3] * stateTransform[3]);
	const float avgScale = (sx + sy) * 0.5f;

	_state->m_AvgScale = avgScale;

	const float quantFactor = 0.1f;
	const float quantScale = (bx::floor((avgScale / quantFactor) + 0.5f)) * quantFactor;
	_state->m_FontScale = quantScale;
}

static float* allocTransformedVertices(Context* _ctx, uint32_t _numVertices)
{
	if (_numVertices > _ctx->m_TransformedVertexCapacity)
	{
		bx::AllocatorI* allocator = _ctx->m_Allocator;
		_ctx->m_TransformedVertices = (float*)bx::alignedRealloc(allocator, _ctx->m_TransformedVertices, sizeof(float) * 2 * _numVertices, 16);
		_ctx->m_TransformedVertexCapacity = _numVertices;
	}

	return _ctx->m_TransformedVertices;
}

static const float* transformPath(Context* _ctx)
{
	if (_ctx->m_PathTransformed)
	{
		return _ctx->m_TransformedVertices;
	}

	Path* path = _ctx->m_Path;

	const uint32_t numPathVertices = pathGetNumVertices(path);
	float* transformedVertices = allocTransformedVertices(_ctx, numPathVertices);

	const State* state = getState(_ctx);
	const float* stateTransform = state->m_TransformMtx;
	const float* pathVertices = pathGetVertices(path);
	vgutil::batchTransformPositions(pathVertices, numPathVertices, transformedVertices, stateTransform);
	_ctx->m_PathTransformed = true;

	return transformedVertices;
}

static VertexBuffer* allocVertexBuffer(Context* _ctx)
{
	if (_ctx->m_NumVertexBuffers + 1 > _ctx->m_VertexBufferCapacity)
	{
		_ctx->m_VertexBufferCapacity++;
		_ctx->m_VertexBuffers = (VertexBuffer*)bx::realloc(_ctx->m_Allocator, _ctx->m_VertexBuffers, sizeof(VertexBuffer) * _ctx->m_VertexBufferCapacity);
		_ctx->m_GPUVertexBuffers = (GPUVertexBuffer*)bx::realloc(_ctx->m_Allocator, _ctx->m_GPUVertexBuffers, sizeof(GPUVertexBuffer) * _ctx->m_VertexBufferCapacity);

		GPUVertexBuffer* gpuvb = &_ctx->m_GPUVertexBuffers[_ctx->m_VertexBufferCapacity - 1];

		gpuvb->m_PosBufferHandle = BGFX_INVALID_HANDLE;
		gpuvb->m_UVBufferHandle = BGFX_INVALID_HANDLE;
		gpuvb->m_ColorBufferHandle = BGFX_INVALID_HANDLE;
	}

#if BX_CONFIG_SUPPORTS_THREADING
	bx::MutexScope ms(*_ctx->m_DataPoolMutex);
#endif // BX_CONFIG_SUPPORTS_THREADING

	VertexBuffer* vb = &_ctx->m_VertexBuffers[_ctx->m_NumVertexBuffers++];
	vb->m_Pos = (float*)bx::alloc(_ctx->m_PosBufferPool, sizeof(float) * 2 * _ctx->m_Config.maxVBVertices);
	vb->m_Color = (uint32_t*)bx::alloc(_ctx->m_ColorBufferPool, sizeof(uint32_t) * _ctx->m_Config.maxVBVertices);
	vb->m_UV = (uv_t*)bx::alloc(_ctx->m_UVBufferPool, sizeof(uv_t) * 2 * _ctx->m_Config.maxVBVertices);
	vb->m_Count = 0;

	return vb;
}

static uint16_t allocIndexBuffer(Context* _ctx)
{
#if BX_CONFIG_SUPPORTS_THREADING
	bx::MutexScope ms(*_ctx->m_DataPoolMutex);
#endif // BX_CONFIG_SUPPORTS_THREADING

	uint16_t ibID = UINT16_MAX;
	const uint32_t numIB = _ctx->m_NumIndexBuffers;
	for (uint32_t ii = 0; ii < numIB; ++ii)
	{
		if (_ctx->m_IndexBuffers[ii].m_Count == 0)
		{
			ibID = (uint16_t)ii;
			break;
		}
	}

	if (ibID == UINT16_MAX)
	{
		_ctx->m_NumIndexBuffers++;
		_ctx->m_IndexBuffers = (IndexBuffer*)bx::realloc(_ctx->m_Allocator, _ctx->m_IndexBuffers, sizeof(IndexBuffer) * _ctx->m_NumIndexBuffers);
		_ctx->m_GPUIndexBuffers = (GPUIndexBuffer*)bx::realloc(_ctx->m_Allocator, _ctx->m_GPUIndexBuffers, sizeof(GPUIndexBuffer) * _ctx->m_NumIndexBuffers);

		ibID = (uint16_t)(_ctx->m_NumIndexBuffers - 1);

		IndexBuffer* ib = &_ctx->m_IndexBuffers[ibID];
		ib->m_Capacity = 0;
		ib->m_Count = 0;
		ib->m_Indices = NULL;

		GPUIndexBuffer* gpuib = &_ctx->m_GPUIndexBuffers[ibID];
		gpuib->m_bgfxHandle = BGFX_INVALID_HANDLE;
	}

	return ibID;
}

static void releaseIndexBuffer(Context* _ctx, uint16_t* _data)
{
#if BX_CONFIG_SUPPORTS_THREADING
	bx::MutexScope ms(*_ctx->m_DataPoolMutex);
#endif // BX_CONFIG_SUPPORTS_THREADING

	VG_CHECK(_data != NULL, "Tried to release a null vertex buffer");
	const uint32_t numIB = _ctx->m_NumIndexBuffers;
	for (uint32_t ii = 0; ii < numIB; ++ii)
	{
		if (_ctx->m_IndexBuffers[ii].m_Indices == _data)
		{
			// Reset the ib for reuse.
			_ctx->m_IndexBuffers[ii].m_Count = 0;
			break;
		}
	}
}

static void createDrawCommand_VertexColor(Context* _ctx, const float* _vtx, uint32_t _numVertices, const uint32_t* _colors, uint32_t _numColors, const uint16_t* _indices, uint32_t _numIndices)
{
	// Allocate the draw command
	const ImageHandle fontImg = fsGetFontAtlasImage(_ctx->m_FontSystem);
	DrawCommand* cmd = allocDrawCommand(_ctx, _numVertices, _numIndices, DrawCommand::Type::Textured, fontImg.idx);

	// Vertex buffer
	VertexBuffer* vb = &_ctx->m_VertexBuffers[cmd->m_VertexBufferID];
	const uint32_t vbOffset = cmd->m_FirstVertexID + cmd->m_NumVertices;

	float* dstPos = &vb->m_Pos[vbOffset << 1];
	bx::memCopy(dstPos, _vtx, sizeof(float) * 2 * _numVertices);

	const uv_t* uv = fsGetWhitePixelUV(_ctx->m_FontSystem);

	uv_t* dstUV = &vb->m_UV[vbOffset << 1];
#if VG_CONFIG_UV_INT16
	vgutil::memset32(dstUV, _numVertices, &uv[0]);
#else
	vgutil::memset64(dstUV, _numVertices, &uv[0]);
#endif // VG_CONFIG_UV_INT16

	uint32_t* dstColor = &vb->m_Color[vbOffset];
	if (_numColors == _numVertices)
	{
		bx::memCopy(dstColor, _colors, sizeof(uint32_t) * _numVertices);
	}
	else
	{
		VG_CHECK(_numColors == 1, "Invalid size of color array passed.");
		vgutil::memset32(dstColor, _numVertices, _colors);
	}

	// Index buffer
	IndexBuffer* ib = &_ctx->m_IndexBuffers[_ctx->m_ActiveIndexBufferID];
	uint16_t* dstIndex = &ib->m_Indices[cmd->m_FirstIndexID + cmd->m_NumIndices];
	vgutil::batchTransformDrawIndices(_indices, _numIndices, dstIndex, (uint16_t)cmd->m_NumVertices);

	cmd->m_NumVertices += _numVertices;
	cmd->m_NumIndices += _numIndices;
}

static void createDrawCommand_ImagePattern(Context* _ctx, ImagePatternHandle _imgPatternHandle, const float* _vtx, uint32_t _numVertices, const uint32_t* _colors, uint32_t _numColors, const uint16_t* _indices, uint32_t _numIndices)
{
	DrawCommand* cmd = allocDrawCommand(_ctx, _numVertices, _numIndices, DrawCommand::Type::ImagePattern, _imgPatternHandle.idx);

	VertexBuffer* vb = &_ctx->m_VertexBuffers[cmd->m_VertexBufferID];
	const uint32_t vbOffset = cmd->m_FirstVertexID + cmd->m_NumVertices;

	float* dstPos = &vb->m_Pos[vbOffset << 1];
	bx::memCopy(dstPos, _vtx, sizeof(float) * 2 * _numVertices);

	uint32_t* dstColor = &vb->m_Color[vbOffset];
	if (_numColors == _numVertices)
	{
		bx::memCopy(dstColor, _colors, sizeof(uint32_t) * _numVertices);
	}
	else
	{
		VG_CHECK(_numColors == 1, "Invalid size of color array passed.");
		vgutil::memset32(dstColor, _numVertices, _colors);
	}

	IndexBuffer* ib = &_ctx->m_IndexBuffers[_ctx->m_ActiveIndexBufferID];
	uint16_t* dstIndex = &ib->m_Indices[cmd->m_FirstIndexID + cmd->m_NumIndices];
	vgutil::batchTransformDrawIndices(_indices, _numIndices, dstIndex, (uint16_t)cmd->m_NumVertices);

	cmd->m_NumVertices += _numVertices;
	cmd->m_NumIndices += _numIndices;
}

static void createDrawCommand_ColorGradient(Context* _ctx, GradientHandle _gradientHandle, const float* _vtx, uint32_t _numVertices, const uint32_t* _colors, uint32_t _numColors, const uint16_t* _indices, uint32_t _numIndices)
{
	DrawCommand* cmd = allocDrawCommand(_ctx, _numVertices, _numIndices, DrawCommand::Type::ColorGradient, _gradientHandle.idx);

	VertexBuffer* vb = &_ctx->m_VertexBuffers[cmd->m_VertexBufferID];
	const uint32_t vbOffset = cmd->m_FirstVertexID + cmd->m_NumVertices;

	float* dstPos = &vb->m_Pos[vbOffset << 1];
	bx::memCopy(dstPos, _vtx, sizeof(float) * 2 * _numVertices);

	uint32_t* dstColor = &vb->m_Color[vbOffset];
	if (_numColors == _numVertices)
	{
		bx::memCopy(dstColor, _colors, sizeof(uint32_t) * _numVertices);
	}
	else
	{
		VG_CHECK(_numColors == 1, "Invalid size of color array passed.");
		vgutil::memset32(dstColor, _numVertices, _colors);
	}

	IndexBuffer* ib = &_ctx->m_IndexBuffers[_ctx->m_ActiveIndexBufferID];
	uint16_t* dstIndex = &ib->m_Indices[cmd->m_FirstIndexID + cmd->m_NumIndices];
	vgutil::batchTransformDrawIndices(_indices, _numIndices, dstIndex, (uint16_t)cmd->m_NumVertices);

	cmd->m_NumVertices += _numVertices;
	cmd->m_NumIndices += _numIndices;
}

static void createDrawCommand_Clip(Context* _ctx, const float* _vtx, uint32_t _numVertices, const uint16_t* _indices, uint32_t _numIndices)
{
	// Allocate the draw command
	DrawCommand* cmd = allocClipCommand(_ctx, _numVertices, _numIndices);

	// Vertex buffer
	VertexBuffer* vb = &_ctx->m_VertexBuffers[cmd->m_VertexBufferID];
	const uint32_t vbOffset = cmd->m_FirstVertexID + cmd->m_NumVertices;

	float* dstPos = &vb->m_Pos[vbOffset << 1];
	bx::memCopy(dstPos, _vtx, sizeof(float) * 2 * _numVertices);

	// Index buffer
	IndexBuffer* ib = &_ctx->m_IndexBuffers[_ctx->m_ActiveIndexBufferID];
	uint16_t* dstIndex = &ib->m_Indices[cmd->m_FirstIndexID + cmd->m_NumIndices];
	vgutil::batchTransformDrawIndices(_indices, _numIndices, dstIndex, (uint16_t)cmd->m_NumVertices);

	cmd->m_NumVertices += _numVertices;
	cmd->m_NumIndices += _numIndices;
}

// NOTE: Side effect: Resets m_ForceNewDrawCommand and m_ForceNewClipCommand if the current
// vertex buffer cannot hold the specified amount of vertices.
static uint32_t allocVertices(Context* _ctx, uint32_t _numVertices, uint32_t* _vbID)
{
	VG_CHECK(_numVertices < _ctx->m_Config.maxVBVertices, "A single draw call cannot have more than %d vertices", _ctx->m_Config.maxVBVertices);

	// Check if the current vertex buffer can hold the specified amount of vertices
	VertexBuffer* vb = &_ctx->m_VertexBuffers[_ctx->m_NumVertexBuffers - 1];
	if (vb->m_Count + _numVertices > _ctx->m_Config.maxVBVertices)
	{
		// It cannot. Allocate a new vb.
		vb = allocVertexBuffer(_ctx);
		VG_CHECK(vb, "Failed to allocate new Vertex Buffer");

		// The currently active vertex buffer has changed so force a new draw command.
		_ctx->m_ForceNewDrawCommand = true;
		_ctx->m_ForceNewClipCommand = true;
	}

	*_vbID = (uint32_t)(vb - _ctx->m_VertexBuffers);

	const uint32_t firstVertexID = vb->m_Count;
	vb->m_Count += _numVertices;
	return firstVertexID;
}

static uint32_t allocIndices(Context* _ctx, uint32_t _numIndices)
{
	IndexBuffer* ib = &_ctx->m_IndexBuffers[_ctx->m_ActiveIndexBufferID];
	if (ib->m_Count + _numIndices > ib->m_Capacity)
	{
		const uint32_t nextCapacity = ib->m_Capacity != 0 ? (ib->m_Capacity * 3) / 2 : 32;

		ib->m_Capacity = bx::max(nextCapacity, ib->m_Count + _numIndices);
		ib->m_Indices = (uint16_t*)bx::alignedRealloc(_ctx->m_Allocator, ib->m_Indices, sizeof(uint16_t) * ib->m_Capacity, 16);
	}

	const uint32_t firstIndexID = ib->m_Count;
	ib->m_Count += _numIndices;
	return firstIndexID;
}

static DrawCommand* allocDrawCommand(Context* _ctx, uint32_t _numVertices, uint32_t _numIndices, DrawCommand::Type::Enum _type, uint16_t _handle)
{
	uint32_t vertexBufferID;
	const uint32_t firstVertexID = allocVertices(_ctx, _numVertices, &vertexBufferID);
	const uint32_t firstIndexID = allocIndices(_ctx, _numIndices);

	const State* state = getState(_ctx);
	const float* scissor = state->m_ScissorRect;

	if (!_ctx->m_ForceNewDrawCommand && _ctx->m_NumDrawCommands != 0)
	{
		DrawCommand* prevCmd = &_ctx->m_DrawCommands[_ctx->m_NumDrawCommands - 1];

		VG_CHECK(prevCmd->m_VertexBufferID == vertexBufferID, "Cannot merge draw commands with different vertex buffers");
		VG_CHECK(prevCmd->m_ScissorRect[0] == (uint16_t)scissor[0]
		      && prevCmd->m_ScissorRect[1] == (uint16_t)scissor[1]
		      && prevCmd->m_ScissorRect[2] == (uint16_t)scissor[2]
		      && prevCmd->m_ScissorRect[3] == (uint16_t)scissor[3], "Invalid scissor rect");

		if (prevCmd->m_Type == _type && prevCmd->m_HandleID == _handle)
		{
			return prevCmd;
		}
	}

	// The new draw command cannot be combined with the previous one. Create a new one.
	if (_ctx->m_NumDrawCommands == _ctx->m_DrawCommandCapacity)
	{
		_ctx->m_DrawCommandCapacity = _ctx->m_DrawCommandCapacity + 32;
		_ctx->m_DrawCommands = (DrawCommand*)bx::realloc(_ctx->m_Allocator, _ctx->m_DrawCommands, sizeof(DrawCommand) * _ctx->m_DrawCommandCapacity);
	}

	DrawCommand* cmd = &_ctx->m_DrawCommands[_ctx->m_NumDrawCommands];
	_ctx->m_NumDrawCommands++;

	cmd->m_VertexBufferID = vertexBufferID;
	cmd->m_FirstVertexID = firstVertexID;
	cmd->m_FirstIndexID = firstIndexID;
	cmd->m_NumVertices = 0;
	cmd->m_NumIndices = 0;
	cmd->m_Type = _type;
	cmd->m_HandleID = _handle;
	cmd->m_ScissorRect[0] = (uint16_t)scissor[0];
	cmd->m_ScissorRect[1] = (uint16_t)scissor[1];
	cmd->m_ScissorRect[2] = (uint16_t)scissor[2];
	cmd->m_ScissorRect[3] = (uint16_t)scissor[3];
	bx::memCopy(&cmd->m_ClipState, &_ctx->m_ClipState, sizeof(ClipState));

	_ctx->m_ForceNewDrawCommand = false;

	return cmd;
}

static DrawCommand* allocClipCommand(Context* _ctx, uint32_t _numVertices, uint32_t _numIndices)
{
	uint32_t vertexBufferID;
	const uint32_t firstVertexID = allocVertices(_ctx, _numVertices, &vertexBufferID);
	const uint32_t firstIndexID = allocIndices(_ctx, _numIndices);

	const State* state = getState(_ctx);
	const float* scissor = state->m_ScissorRect;

	if (!_ctx->m_ForceNewClipCommand && _ctx->m_NumClipCommands != 0)
	{
		DrawCommand* prevCmd = &_ctx->m_ClipCommands[_ctx->m_NumClipCommands - 1];

		VG_CHECK(prevCmd->m_VertexBufferID == vertexBufferID, "Cannot merge clip commands with different vertex buffers");
		VG_CHECK(prevCmd->m_ScissorRect[0] == (uint16_t)scissor[0]
		      && prevCmd->m_ScissorRect[1] == (uint16_t)scissor[1]
		      && prevCmd->m_ScissorRect[2] == (uint16_t)scissor[2]
		      && prevCmd->m_ScissorRect[3] == (uint16_t)scissor[3], "Invalid scissor rect");
		VG_CHECK(prevCmd->m_Type == DrawCommand::Type::Clip, "Invalid draw command type");

		return prevCmd;
	}

	// The new clip command cannot be combined with the previous one. Create a new one.
	if (_ctx->m_NumClipCommands == _ctx->m_ClipCommandCapacity)
	{
		_ctx->m_ClipCommandCapacity = _ctx->m_ClipCommandCapacity + 32;
		_ctx->m_ClipCommands = (DrawCommand*)bx::realloc(_ctx->m_Allocator, _ctx->m_ClipCommands, sizeof(DrawCommand) * _ctx->m_ClipCommandCapacity);
	}

	DrawCommand* cmd = &_ctx->m_ClipCommands[_ctx->m_NumClipCommands];
	_ctx->m_NumClipCommands++;

	cmd->m_VertexBufferID = vertexBufferID;
	cmd->m_FirstVertexID = firstVertexID;
	cmd->m_FirstIndexID = firstIndexID;
	cmd->m_NumVertices = 0;
	cmd->m_NumIndices = 0;
	cmd->m_Type = DrawCommand::Type::Clip;
	cmd->m_HandleID = UINT16_MAX;
	cmd->m_ScissorRect[0] = (uint16_t)scissor[0];
	cmd->m_ScissorRect[1] = (uint16_t)scissor[1];
	cmd->m_ScissorRect[2] = (uint16_t)scissor[2];
	cmd->m_ScissorRect[3] = (uint16_t)scissor[3];
	cmd->m_ClipState.m_FirstCmdID = ~0u;
	cmd->m_ClipState.m_NumCmds = 0;

	_ctx->m_ForceNewClipCommand = false;

	return cmd;
}

static void resetImage(Image* _img)
{
	_img->m_bgfxHandle = BGFX_INVALID_HANDLE;
	_img->m_Width = 0;
	_img->m_Height = 0;
	_img->m_Flags = 0;
	_img->m_Owned = false;
}

static ImageHandle allocImage(Context* _ctx)
{
	ImageHandle handle = { _ctx->m_ImageHandleAlloc->alloc() };
	if (!isValid(handle))
	{
		return VG_INVALID_HANDLE;
	}

	if (handle.idx >= _ctx->m_ImageCapacity)
	{
		const uint32_t oldCapacity = _ctx->m_ImageCapacity;

		_ctx->m_ImageCapacity = bx::min(bx::max(_ctx->m_ImageCapacity + 4, handle.idx + 1), _ctx->m_Config.maxImages);
		_ctx->m_Images = (Image*)bx::realloc(_ctx->m_Allocator, _ctx->m_Images, sizeof(Image) * _ctx->m_ImageCapacity);
		if (!_ctx->m_Images)
		{
			return VG_INVALID_HANDLE;
		}

		// Reset all new textures...
		const uint32_t capacity = _ctx->m_ImageCapacity;
		for (uint32_t ii = oldCapacity; ii < capacity; ++ii)
		{
			resetImage(&_ctx->m_Images[ii]);
		}
	}

	VG_CHECK(handle.idx < _ctx->m_ImageCapacity, "Allocated invalid image handle");
	Image* tex = &_ctx->m_Images[handle.idx];
	VG_CHECK(!bgfx::isValid(tex->m_bgfxHandle), "Allocated texture is already in use");
	resetImage(tex);

	return handle;
}

static void renderTextQuads(Context* _ctx, const TextQuad* _quads, uint32_t _numQuads, Color _color, ImageHandle _img)
{
	const uint32_t numDrawVertices = _numQuads * 4;
	const uint32_t numDrawIndices = _numQuads * 6;

	if (_ctx->m_TextVertexCapacity < numDrawVertices)
	{
		_ctx->m_TextVertices = (float*)bx::alignedRealloc(_ctx->m_Allocator, _ctx->m_TextVertices, sizeof(float) * 2 * numDrawVertices, 16);
		_ctx->m_TextVertexCapacity = numDrawVertices;
	}

	const State* state = getState(_ctx);
	const float scale = state->m_FontScale * _ctx->m_DevicePixelRatio;
	const float invscale = 1.0f / scale;

	float mtx[6];
	mtx[0] = state->m_TransformMtx[0] * invscale;
	mtx[1] = state->m_TransformMtx[1] * invscale;
	mtx[2] = state->m_TransformMtx[2] * invscale;
	mtx[3] = state->m_TransformMtx[3] * invscale;
	mtx[4] = state->m_TransformMtx[4];
	mtx[5] = state->m_TransformMtx[5];

	// TODO: Calculate bounding rect of the quads.
	vgutil::batchTransformTextQuads(&_quads->m_Pos[0], _numQuads, mtx, _ctx->m_TextVertices);

	DrawCommand* cmd = allocDrawCommand(_ctx, numDrawVertices, numDrawIndices, DrawCommand::Type::Textured, _img.idx);

	VertexBuffer* vb = &_ctx->m_VertexBuffers[cmd->m_VertexBufferID];
	const uint32_t vbOffset = cmd->m_FirstVertexID + cmd->m_NumVertices;

	float* dstPos = &vb->m_Pos[vbOffset << 1];
	bx::memCopy(dstPos, _ctx->m_TextVertices, sizeof(float) * 2 * numDrawVertices);

	uint32_t* dstColor = &vb->m_Color[vbOffset];
	vgutil::memset32(dstColor, numDrawVertices, &_color);

	uv_t* dstUV = &vb->m_UV[vbOffset << 1];
	const TextQuad* qq = _quads;
	uint32_t nq = _numQuads;
	while (nq-- > 0)
	{
		const uv_t s0 = qq->m_TexCoord[0];
		const uv_t t0 = qq->m_TexCoord[1];
		const uv_t s1 = qq->m_TexCoord[2];
		const uv_t t1 = qq->m_TexCoord[3];

		dstUV[0] = s0; dstUV[1] = t0;
		dstUV[2] = s1; dstUV[3] = t0;
		dstUV[4] = s1; dstUV[5] = t1;
		dstUV[6] = s0; dstUV[7] = t1;

		dstUV += 8;
		++qq;
	}

	IndexBuffer* ib = &_ctx->m_IndexBuffers[_ctx->m_ActiveIndexBufferID];
	uint16_t* dstIndex = &ib->m_Indices[cmd->m_FirstIndexID + cmd->m_NumIndices];
	vgutil::genQuadIndices_unaligned(dstIndex, _numQuads, (uint16_t)cmd->m_NumVertices);

	cmd->m_NumVertices += numDrawVertices;
	cmd->m_NumIndices += numDrawIndices;
}

static CommandListHandle allocCommandList(Context* _ctx)
{
	CommandListHandle handle = { _ctx->m_CmdListHandleAlloc->alloc() };
	if (!isValid(handle))
	{
		return VG_INVALID_HANDLE;
	}

	VG_CHECK(handle.idx < _ctx->m_Config.maxCommandLists, "Allocated invalid command list handle");
	CommandList* cl = &_ctx->m_CmdLists[handle.idx];
	bx::memSet(cl, 0, sizeof(CommandList));

	return handle;
}

static inline bool isCommandListHandleValid(Context* _ctx, CommandListHandle _handle)
{
	return isValid(_handle) && _ctx->m_CmdListHandleAlloc->isValid(_handle.idx);
}

#if VG_CONFIG_ENABLE_SHAPE_CACHING
static CommandListCache* allocCommandListCache(Context* _ctx)
{
	bx::AllocatorI* allocator = _ctx->m_Allocator;

	CommandListCache* cache = (CommandListCache*)bx::alloc(allocator, sizeof(CommandListCache));
	bx::memSet(cache, 0, sizeof(CommandListCache));

	return cache;
}

static void freeCommandListCache(Context* _ctx, CommandListCache* _cache)
{
	bx::AllocatorI* allocator = _ctx->m_Allocator;

	clCacheReset(_ctx, _cache);
	bx::free(allocator, _cache);
}
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING

static uint8_t* clAllocCommand(Context* _ctx, CommandList* _cl, CommandType::Enum _cmdType, uint32_t _dataSize)
{
	const uint32_t alignedDataSize = alignSize(_dataSize, VG_CONFIG_COMMAND_LIST_ALIGNMENT);
	const uint32_t totalSize = 0
		+ kAlignedCommandHeaderSize
		+ alignedDataSize;

	const uint32_t pos = _cl->m_CommandBufferPos;
	VG_CHECK(isAligned(pos, VG_CONFIG_COMMAND_LIST_ALIGNMENT), "Unaligned command buffer position");

	if (pos + totalSize > _cl->m_CommandBufferCapacity)
	{
		const uint32_t capacityDelta = bx::max<uint32_t>(totalSize, 256);
		_cl->m_CommandBufferCapacity += capacityDelta;
		_cl->m_CommandBuffer = (uint8_t*)bx::alignedRealloc(_ctx->m_Allocator, _cl->m_CommandBuffer, _cl->m_CommandBufferCapacity, VG_CONFIG_COMMAND_LIST_ALIGNMENT);

		_ctx->m_Stats.cmdListMemoryTotal += capacityDelta;
	}

	uint8_t* ptr = &_cl->m_CommandBuffer[pos];
	_cl->m_CommandBufferPos += totalSize;
	_ctx->m_Stats.cmdListMemoryUsed += totalSize;

	CommandHeader* hdr = (CommandHeader*)ptr;
	ptr += kAlignedCommandHeaderSize;

	hdr->m_Type = _cmdType;
	hdr->m_Size = alignedDataSize;

	return ptr;
}

static uint32_t clStoreString(Context* _ctx, CommandList* _cl, const char* _str, uint32_t _len)
{
	if (_cl->m_StringBufferPos + _len > _cl->m_StringBufferCapacity)
	{
		_cl->m_StringBufferCapacity += bx::max<uint32_t>(_len, 128);
		_cl->m_StringBuffer = (char*)bx::realloc(_ctx->m_Allocator, _cl->m_StringBuffer, _cl->m_StringBufferCapacity);
	}

	const uint32_t offset = _cl->m_StringBufferPos;
	bx::memCopy(_cl->m_StringBuffer + offset, _str, _len);
	_cl->m_StringBufferPos += _len;
	return offset;
}

#if VG_CONFIG_ENABLE_SHAPE_CACHING
static CommandListCache* clGetCache(Context* _ctx, CommandList* _cl)
{
	if ((_cl->m_Flags & CommandListFlags::Cacheable) == 0)
	{
		return NULL;
	}

	CommandListCache* cache = _cl->m_Cache;
	if (!cache)
	{
		cache = allocCommandListCache(_ctx);
		_cl->m_Cache = cache;
	}

	return cache;
}

static void pushCommandListCache(Context* _ctx, CommandListCache* _cache)
{
	VG_CHECK(_ctx->m_CmdListCacheStackTop + 1 < VG_CONFIG_COMMAND_LIST_CACHE_STACK_SIZE, "Command list cache stack overflow");
	++_ctx->m_CmdListCacheStackTop;
	_ctx->m_CmdListCacheStack[_ctx->m_CmdListCacheStackTop] = _cache;
}

static void popCommandListCache(Context* _ctx)
{
	VG_CHECK(_ctx->m_CmdListCacheStackTop != ~0u, "Command list cache stack underflow");
	--_ctx->m_CmdListCacheStackTop;
}

static CommandListCache* getCommandListCacheStackTop(Context* _ctx)
{
	const uint32_t top = _ctx->m_CmdListCacheStackTop;
	return top == ~0u ? NULL : _ctx->m_CmdListCacheStack[top];
}

static void beginCachedCommand(Context* _ctx)
{
	CommandListCache* cache = getCommandListCacheStackTop(_ctx);
	VG_CHECK(cache, "No bound CommandListCache");

	bx::AllocatorI* allocator = _ctx->m_Allocator;

	cache->m_NumCommands++;
	if (cache->m_NumCommands > cache->m_MaxCommands)
	{
		cache->m_MaxCommands += kCommandGrowth;
		cache->m_Commands = (CachedCommand*)bx::realloc(allocator, cache->m_Commands, sizeof(CachedCommand) * cache->m_MaxCommands);
	}


	CachedCommand* lastCmd = &cache->m_Commands[cache->m_NumCommands - 1];
	lastCmd->m_FirstMeshID = (uint16_t)cache->m_NumMeshes;
	lastCmd->m_NumMeshes = 0;

	const State* state = getState(_ctx);
	vgutil::invertMatrix3(state->m_TransformMtx, lastCmd->m_InvTransformMtx);
}

static void endCachedCommand(Context* _ctx)
{
	CommandListCache* cache = getCommandListCacheStackTop(_ctx);
	VG_CHECK(cache, "No bound CommandListCache");

	VG_CHECK(cache->m_NumCommands != 0, "beginCachedCommand() hasn't been called");

	CachedCommand* lastCmd = &cache->m_Commands[cache->m_NumCommands - 1];
	VG_CHECK(lastCmd->m_NumMeshes == 0, "endCachedCommand() called too many times");
	lastCmd->m_NumMeshes = (uint16_t)(cache->m_NumMeshes - lastCmd->m_FirstMeshID);
}

static void addCachedCommand(Context* _ctx, const float* _pos, uint32_t _numVertices, const uint32_t* _colors, uint32_t _numColors, const uint16_t* _indices, uint32_t _numIndices)
{
	CommandListCache* cache = getCommandListCacheStackTop(_ctx);
	VG_CHECK(cache, "No bound CommandListCache");

	bx::AllocatorI* allocator = _ctx->m_Allocator;

	cache->m_NumMeshes++;
	if (cache->m_NumMeshes > cache->m_MaxMeshes)
	{
		cache->m_MaxMeshes += kMeshGrowth;
		cache->m_Meshes = (CachedMesh*)bx::realloc(allocator, cache->m_Meshes, sizeof(CachedMesh) * cache->m_MaxMeshes);
	}

	CachedMesh* mesh = &cache->m_Meshes[cache->m_NumMeshes - 1];

	const uint32_t totalMem = 0
		+ alignSize(sizeof(float) * 2 * _numVertices, 16)
		+ ((_numColors != 1) ? alignSize(sizeof(uint32_t) * _numVertices, 16) : 0)
		+ alignSize(sizeof(uint16_t) * _numIndices, 16);

	uint8_t* mem = (uint8_t*)bx::alignedAlloc(allocator, totalMem, 16);
	mesh->m_Pos = (float*)mem;
	mem += alignSize(sizeof(float) * 2 * _numVertices, 16);

	const float* invMtx = cache->m_Commands[cache->m_NumCommands - 1].m_InvTransformMtx;
	vgutil::batchTransformPositions(_pos, _numVertices, mesh->m_Pos, invMtx);
	mesh->m_NumVertices = _numVertices;

	if (_numColors == 1)
	{
		mesh->m_Colors = NULL;
	}
	else
	{
		VG_CHECK(_numColors == _numVertices, "Invalid number of colors");
		mesh->m_Colors = (uint32_t*)mem;
		mem += alignSize(sizeof(uint32_t) * _numVertices, 16);

		bx::memCopy(mesh->m_Colors, _colors, sizeof(uint32_t) * _numColors);
	}

	mesh->m_Indices = (uint16_t*)mem;
	bx::memCopy(mesh->m_Indices, _indices, sizeof(uint16_t) * _numIndices);
	mesh->m_NumIndices = _numIndices;
}

// Walk the command list; avoid Path commands and use CachedMesh(es) on Stroker commands.
// Everything else (state, clip, text) is executed similarly to the uncached version (see submitCommandList).
static void clCacheRender(Context* _ctx, CommandList* _cl)
{
	const uint16_t numGradients = _cl->m_NumGradients;
	const uint16_t numImagePatterns = _cl->m_NumImagePatterns;
	const uint32_t clFlags = _cl->m_Flags;

	const bool cullCmds = (clFlags & CommandListFlags::AllowCommandCulling) != 0;

	CommandListCache* clCache = _cl->m_Cache;
	VG_CHECK(clCache != NULL, "No CommandListCache in CommandList; this function shouldn't have been called!");

	const uint16_t firstGradientID = (uint16_t)_ctx->m_NextGradientID;
	const uint16_t firstImagePatternID = (uint16_t)_ctx->m_NextImagePatternID;
	VG_CHECK(firstGradientID + numGradients <= _ctx->m_Config.maxGradients, "Not enough free gradients for command list. Increase Init::maxGradients");
	VG_CHECK(firstImagePatternID + numImagePatterns <= _ctx->m_Config.maxImagePatterns, "Not enough free image patterns for command list. Increase Init::maxImagePatterns");
	BX_UNUSED(numGradients, numImagePatterns); // For Release builds.

	const uint8_t* cmd = _cl->m_CommandBuffer;
	const uint8_t* cmdListEnd = _cl->m_CommandBuffer + _cl->m_CommandBufferPos;
	if (cmd == cmdListEnd)
	{
		return;
	}

	const char* stringBuffer = _cl->m_StringBuffer;
	CachedCommand* nextCachedCommand = &clCache->m_Commands[0];

	bool skipCmds = false;

#if VG_CONFIG_COMMAND_LIST_PRESERVE_STATE
	_ctx->pushState();
#endif // VG_CONFIG_COMMAND_LIST_PRESERVE_STATE

	while (cmd < cmdListEnd)
	{
		const CommandHeader* cmdHeader = (CommandHeader*)cmd;
		cmd += kAlignedCommandHeaderSize;

		const uint8_t* nextCmd = cmd + cmdHeader->m_Size;

		// Skip path commands.
		if (cmdHeader->m_Type >= CommandType::FirstPathCommand && cmdHeader->m_Type <= CommandType::LastPathCommand)
		{
			cmd = nextCmd;
			continue;
		}

		if (skipCmds && cmdHeader->m_Type >= CommandType::FirstStrokerCommand && cmdHeader->m_Type <= CommandType::LastStrokerCommand)
		{
			cmd = nextCmd;
			++nextCachedCommand;
			continue;
		}

		switch (cmdHeader->m_Type)
		{
		case CommandType::FillPathColor:
		{
			const uint32_t flags = CMD_READ(cmd, uint32_t);
			const Color color = CMD_READ(cmd, Color);
			BX_UNUSED(flags);
			submitCachedMesh(_ctx, color, &clCache->m_Meshes[nextCachedCommand->m_FirstMeshID], nextCachedCommand->m_NumMeshes);
			++nextCachedCommand;
		} break;
		case CommandType::FillPathGradient:
		{
			const uint32_t flags = CMD_READ(cmd, uint32_t);
			const uint16_t gradientHandle = CMD_READ(cmd, uint16_t);
			const uint16_t gradientFlags = CMD_READ(cmd, uint16_t);
			BX_UNUSED(flags);

			const GradientHandle gradient = { isLocal(gradientFlags) ? (uint16_t)(gradientHandle + firstGradientID) : gradientHandle, 0 };
			submitCachedMesh(_ctx, gradient, &clCache->m_Meshes[nextCachedCommand->m_FirstMeshID], nextCachedCommand->m_NumMeshes);
			++nextCachedCommand;
		} break;
		case CommandType::FillPathImagePattern:
		{
			const uint32_t flags = CMD_READ(cmd, uint32_t);
			const Color color = CMD_READ(cmd, Color);
			const uint16_t imgPatternHandle = CMD_READ(cmd, uint16_t);
			const uint16_t imgPatternFlags = CMD_READ(cmd, uint16_t);
			BX_UNUSED(flags);

			const ImagePatternHandle imgPattern = { isLocal(imgPatternFlags) ? (uint16_t)(imgPatternHandle + firstImagePatternID) : imgPatternHandle, 0 };
			submitCachedMesh(_ctx, imgPattern, color, &clCache->m_Meshes[nextCachedCommand->m_FirstMeshID], nextCachedCommand->m_NumMeshes);
			++nextCachedCommand;
		} break;
		case CommandType::StrokePathColor:
		{
			const float width = CMD_READ(cmd, float);
			const uint32_t flags = CMD_READ(cmd, uint32_t);
			const Color color = CMD_READ(cmd, Color);
			BX_UNUSED(flags, width);

			submitCachedMesh(_ctx, color, &clCache->m_Meshes[nextCachedCommand->m_FirstMeshID], nextCachedCommand->m_NumMeshes);
			++nextCachedCommand;
		} break;
		case CommandType::StrokePathGradient:
		{
			const float width = CMD_READ(cmd, float);
			const uint32_t flags = CMD_READ(cmd, uint32_t);
			const uint16_t gradientHandle = CMD_READ(cmd, uint16_t);
			const uint16_t gradientFlags = CMD_READ(cmd, uint16_t);
			BX_UNUSED(flags, width);

			const GradientHandle gradient = { isLocal(gradientFlags) ? (uint16_t)(gradientHandle + firstGradientID) : gradientHandle, 0 };
			submitCachedMesh(_ctx, gradient, &clCache->m_Meshes[nextCachedCommand->m_FirstMeshID], nextCachedCommand->m_NumMeshes);
			++nextCachedCommand;
		} break;
		case CommandType::StrokePathImagePattern:
		{
			const float width = CMD_READ(cmd, float);
			const uint32_t flags = CMD_READ(cmd, uint32_t);
			const Color color = CMD_READ(cmd, Color);
			const uint16_t imgPatternHandle = CMD_READ(cmd, uint16_t);
			const uint16_t imgPatternFlags = CMD_READ(cmd, uint16_t);
			BX_UNUSED(flags, width);

			const ImagePatternHandle imgPattern = { isLocal(imgPatternFlags) ? (uint16_t)(imgPatternHandle + firstImagePatternID) : imgPatternHandle, 0 };
			submitCachedMesh(_ctx, imgPattern, color, &clCache->m_Meshes[nextCachedCommand->m_FirstMeshID], nextCachedCommand->m_NumMeshes);
			++nextCachedCommand;
		} break;
		case CommandType::IndexedTriList:
		{
			const uint32_t numVertices = CMD_READ(cmd, uint32_t);
			const float* positions = (float*)cmd;
			cmd += sizeof(float) * 2 * numVertices;
			const uint32_t numUVs = CMD_READ(cmd, uint32_t);
			const uv_t* uv = (uv_t*)cmd;
			cmd += sizeof(uv_t) * 2 * numUVs;
			const uint32_t numColors = CMD_READ(cmd, uint32_t);
			const Color* colors = (Color*)cmd;
			cmd += sizeof(Color) * numColors;
			const uint32_t numIndices = CMD_READ(cmd, uint32_t);
			const uint16_t* indices = (uint16_t*)cmd;
			cmd += sizeof(uint16_t) * numIndices;
			const uint16_t imgHandle = CMD_READ(cmd, uint16_t);

			_ctx->indexedTriList(positions, numUVs ? uv : NULL, numVertices, colors, numColors, indices, numIndices, { imgHandle });
		} break;
		case CommandType::CreateLinearGradient:
		{
			const float* params = (float*)cmd;
			cmd += sizeof(float) * 4;
			const Color* colors = (Color*)cmd;
			cmd += sizeof(Color) * 2;
			_ctx->createLinearGradient(params[0], params[1], params[2], params[3], colors[0], colors[1]);
		} break;
		case CommandType::CreateBoxGradient:
		{
			const float* params = (float*)cmd;
			cmd += sizeof(float) * 6;
			const Color* colors = (Color*)cmd;
			cmd += sizeof(Color) * 2;
			_ctx->createBoxGradient(params[0], params[1], params[2], params[3], params[4], params[5], colors[0], colors[1]);
		} break;
		case CommandType::CreateRadialGradient:
		{
			const float* params = (float*)cmd;
			cmd += sizeof(float) * 4;
			const Color* colors = (Color*)cmd;
			cmd += sizeof(Color) * 2;
			_ctx->createRadialGradient(params[0], params[1], params[2], params[3], colors[0], colors[1]);
		} break;
		case CommandType::CreateImagePattern:
		{
			const float* params = (float*)cmd;
			cmd += sizeof(float) * 5;
			const ImageHandle img = CMD_READ(cmd, ImageHandle);
			_ctx->createImagePattern(params[0], params[1], params[2], params[3], params[4], img);
		} break;
		case CommandType::Text:
		{
			const TextConfig* txtCfg = (TextConfig*)cmd;
			cmd += sizeof(TextConfig);
			const float* coords = (float*)cmd;
			cmd += sizeof(float) * 2;
			const uint32_t stringOffset = CMD_READ(cmd, uint32_t);
			const uint32_t stringLen = CMD_READ(cmd, uint32_t);
			VG_CHECK(stringOffset < _cl->m_StringBufferPos, "Invalid string offset");
			VG_CHECK(stringOffset + stringLen <= _cl->m_StringBufferPos, "Invalid string length");

			const char* str = stringBuffer + stringOffset;
			const char* end = str + stringLen;
			_ctx->text(*txtCfg, coords[0], coords[1], str, end);
		} break;
		case CommandType::TextBox:
		{
			const TextConfig* txtCfg = (TextConfig*)cmd;
			cmd += sizeof(TextConfig);
			const float* coords = (float*)cmd;
			cmd += sizeof(float) * 3; // x, y, breakWidth
			const uint32_t stringOffset = CMD_READ(cmd, uint32_t);
			const uint32_t stringLen = CMD_READ(cmd, uint32_t);
			const uint32_t textboxFlags = CMD_READ(cmd, uint32_t);
			VG_CHECK(stringOffset < _cl->m_StringBufferPos, "Invalid string offset");
			VG_CHECK(stringOffset + stringLen <= _cl->m_StringBufferPos, "Invalid string length");

			const char* str = stringBuffer + stringOffset;
			const char* end = str + stringLen;
			_ctx->textBox(*txtCfg, coords[0], coords[1], coords[2], str, end, textboxFlags);
		} break;
		case CommandType::ResetScissor:
		{
			_ctx->resetScissor();
			skipCmds = false;
		} break;
		case CommandType::SetScissor:
		{
			const float* rect = (float*)cmd;
			cmd += sizeof(float) * 4;
			_ctx->setScissor(rect[0], rect[1], rect[2], rect[3]);

			if (cullCmds)
			{
				skipCmds = (rect[2] < 1.0f) || (rect[3] < 1.0f);
			}
		} break;
		case CommandType::IntersectScissor:
		{
			const float* rect = (float*)cmd;
			cmd += sizeof(float) * 4;

			const bool zeroRect = !_ctx->intersectScissor(rect[0], rect[1], rect[2], rect[3]);
			if (cullCmds)
			{
				skipCmds = zeroRect;
			}
		} break;
		case CommandType::PushState:
		{
			_ctx->pushState();
		} break;
		case CommandType::PopState:
		{
			_ctx->popState();
			if (cullCmds)
			{
				const State* state = getState(_ctx);
				const float* scissorRect = &state->m_ScissorRect[0];
				skipCmds = (scissorRect[2] < 1.0f) || (scissorRect[3] < 1.0f);
			}
		} break;
		case CommandType::TransformIdentity:
		{
			_ctx->transformIdentity();
		} break;
		case CommandType::TransformRotate:
		{
			const float ang_rad = CMD_READ(cmd, float);
			_ctx->transformRotate(ang_rad);
		} break;
		case CommandType::TransformTranslate:
		{
			const float* coords = (float*)cmd;
			cmd += sizeof(float) * 2;
			_ctx->transformTranslate(coords[0], coords[1]);
		} break;
		case CommandType::TransformScale:
		{
			const float* coords = (float*)cmd;
			cmd += sizeof(float) * 2;
			_ctx->transformScale(coords[0], coords[1]);
		} break;
		case CommandType::TransformMult:
		{
			const float* mtx = (float*)cmd;
			cmd += sizeof(float) * 6;
			const TransformOrder::Enum order = CMD_READ(cmd, TransformOrder::Enum);
			_ctx->transformMult(mtx, order);
		} break;
		case CommandType::SetViewBox:
		{
			const float* viewBox = (float*)cmd;
			cmd += sizeof(float) * 4;
			_ctx->setViewBox(viewBox[0], viewBox[1], viewBox[2], viewBox[3]);
		} break;
		case CommandType::BeginClip:
		{
			const ClipRule::Enum rule = CMD_READ(cmd, ClipRule::Enum);
			_ctx->beginClip(rule);
		} break;
		case CommandType::EndClip:
		{
			_ctx->endClip();
		} break;
		case CommandType::ResetClip:
		{
			_ctx->resetClip();
		} break;
		case CommandType::SubmitCommandList:
		{
			const uint16_t cmdListID = CMD_READ(cmd, uint16_t);
			const CommandListHandle cmdListHandle = { cmdListID };

			if (isCommandListHandleValid(_ctx, cmdListHandle))
			{
				_ctx->submitCommandList(cmdListHandle);
			}
		} break;
		default:
		{
			VG_CHECK(false, "Unknown cached command");
		} break;
		}

		cmd = nextCmd;
	}

#if VG_CONFIG_COMMAND_LIST_PRESERVE_STATE
	_ctx->popState();
	_ctx->resetClip();
#endif // VG_CONFIG_COMMAND_LIST_PRESERVE_STATE
}

static void clCacheReset(Context* _ctx, CommandListCache* _cache)
{
	bx::AllocatorI* allocator = _ctx->m_Allocator;

	const uint32_t numMeshes = _cache->m_NumMeshes;
	for (uint32_t ii = 0; ii < numMeshes; ++ii)
	{
		CachedMesh* mesh = &_cache->m_Meshes[ii];
		bx::alignedFree(allocator, mesh->m_Pos, 16);
	}
	bx::free(allocator, _cache->m_Meshes);
	bx::free(allocator, _cache->m_Commands);

	bx::memSet(_cache, 0, sizeof(CommandListCache));
}

static void submitCachedMesh(Context* _ctx, Color _col, const CachedMesh* _meshList, uint32_t _numMeshes)
{
	const bool recordClipCommands = _ctx->m_RecordClipCommands;

	const State* state = getState(_ctx);
	const float* mtx = state->m_TransformMtx;

	if (recordClipCommands)
	{
		for (uint32_t ii = 0; ii < _numMeshes; ++ii)
		{
			const CachedMesh* mesh = &_meshList[ii];
			const uint32_t numVertices = mesh->m_NumVertices;
			float* transformedVertices = allocTransformedVertices(_ctx, numVertices);

			vgutil::batchTransformPositions(mesh->m_Pos, numVertices, transformedVertices, mtx);
			createDrawCommand_Clip(_ctx, transformedVertices, numVertices, mesh->m_Indices, mesh->m_NumIndices);
		}
	}
	else
	{
		for (uint32_t ii = 0; ii < _numMeshes; ++ii)
		{
			const CachedMesh* mesh = &_meshList[ii];
			const uint32_t numVertices = mesh->m_NumVertices;
			float* transformedVertices = allocTransformedVertices(_ctx, numVertices);

			const uint32_t* colors = mesh->m_Colors ? mesh->m_Colors : &_col;
			const uint32_t numColors = mesh->m_Colors ? numVertices : 1;

			vgutil::batchTransformPositions(mesh->m_Pos, numVertices, transformedVertices, mtx);
			createDrawCommand_VertexColor(_ctx, transformedVertices, numVertices, colors, numColors, mesh->m_Indices, mesh->m_NumIndices);
		}
	}
}

static void submitCachedMesh(Context* _ctx, GradientHandle _gradientHandle, const CachedMesh* _meshList, uint32_t _numMeshes)
{
	VG_CHECK(!_ctx->m_RecordClipCommands, "Only submitCachedMesh(Color) is supported inside BeginClip()/EndClip()");
	VG_CHECK(isValid(_gradientHandle), "Invalid gradient handle");
	VG_CHECK(!isLocal(_gradientHandle), "Invalid gradient handle");

	const State* state = getState(_ctx);
	const float* mtx = state->m_TransformMtx;

	const uint32_t black = Colors::Black;
	for (uint32_t ii = 0; ii < _numMeshes; ++ii)
	{
		const CachedMesh* mesh = &_meshList[ii];
		const uint32_t numVertices = mesh->m_NumVertices;
		float* transformedVertices = allocTransformedVertices(_ctx, numVertices);

		const uint32_t* colors = mesh->m_Colors ? mesh->m_Colors : &black;
		const uint32_t numColors = mesh->m_Colors ? numVertices : 1;

		vgutil::batchTransformPositions(mesh->m_Pos, numVertices, transformedVertices, mtx);
		createDrawCommand_ColorGradient(_ctx, _gradientHandle, transformedVertices, numVertices, colors, numColors, mesh->m_Indices, mesh->m_NumIndices);
	}
}

static void submitCachedMesh(Context* _ctx, ImagePatternHandle _imgPattern, Color _col, const CachedMesh* _meshList, uint32_t _numMeshes)
{
	VG_CHECK(!_ctx->m_RecordClipCommands, "Only submitCachedMesh(Color) is supported inside BeginClip()/EndClip()");
	VG_CHECK(isValid(_imgPattern), "Invalid image pattern handle");
	VG_CHECK(!isLocal(_imgPattern), "Invalid image pattern handle");

	const State* state = getState(_ctx);
	const float* mtx = state->m_TransformMtx;

	for (uint32_t ii = 0; ii < _numMeshes; ++ii)
	{
		const CachedMesh* mesh = &_meshList[ii];
		const uint32_t numVertices = mesh->m_NumVertices;
		float* transformedVertices = allocTransformedVertices(_ctx, numVertices);

		const uint32_t* colors = mesh->m_Colors ? mesh->m_Colors : &_col;
		const uint32_t numColors = mesh->m_Colors ? numVertices : 1;

		vgutil::batchTransformPositions(mesh->m_Pos, numVertices, transformedVertices, mtx);
		createDrawCommand_ImagePattern(_ctx, _imgPattern, transformedVertices, numVertices, colors, numColors, mesh->m_Indices, mesh->m_NumIndices);
	}
}
#endif // VG_CONFIG_ENABLE_SHAPE_CACHING

static void releaseVertexBufferPosCallback(void* _ptr, void* _userData)
{
	Context* ctx = (Context*)_userData;
#if BX_CONFIG_SUPPORTS_THREADING
	bx::MutexScope ms(*ctx->m_DataPoolMutex);
#endif // BX_CONFIG_SUPPORTS_THREADING
	bx::free(ctx->m_PosBufferPool, _ptr);
}

static void releaseVertexBufferColorCallback(void* _ptr, void* _userData)
{
	Context* ctx = (Context*)_userData;
#if BX_CONFIG_SUPPORTS_THREADING
	bx::MutexScope ms(*ctx->m_DataPoolMutex);
#endif // BX_CONFIG_SUPPORTS_THREADING
	bx::free(ctx->m_ColorBufferPool, _ptr);
}

static void releaseVertexBufferUVCallback(void* _ptr, void* _userData)
{
	Context* ctx = (Context*)_userData;
#if BX_CONFIG_SUPPORTS_THREADING
	bx::MutexScope ms(*ctx->m_DataPoolMutex);
#endif // BX_CONFIG_SUPPORTS_THREADING
	bx::free(ctx->m_UVBufferPool, _ptr);
}

static void releaseIndexBufferCallback(void* _ptr, void* _userData)
{
	Context* ctx = (Context*)_userData;
	releaseIndexBuffer(ctx, (uint16_t*)_ptr);
}
} // namespace vg
