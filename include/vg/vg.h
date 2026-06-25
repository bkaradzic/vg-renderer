#ifndef VG_H
#define VG_H

#include <stdint.h>
// UV component type of the public vertex API (see uv_t below). This is the one
// build-time switch that affects the public ABI; all other tuning is internal
// (src/config.h).
#ifndef VG_CONFIG_UV_INT16
#	define VG_CONFIG_UV_INT16 1
#endif

#define VG_COLOR_RED_Pos     0
#define VG_COLOR_RED_Msk     (0xFFu << VG_COLOR_RED_Pos)
#define VG_COLOR_GREEN_Pos   8
#define VG_COLOR_GREEN_Msk   (0xFFu << VG_COLOR_GREEN_Pos)
#define VG_COLOR_BLUE_Pos    16
#define VG_COLOR_BLUE_Msk    (0xFFu << VG_COLOR_BLUE_Pos)
#define VG_COLOR_ALPHA_Pos   24
#define VG_COLOR_ALPHA_Msk   (0xFFu << VG_COLOR_ALPHA_Pos)
#define VG_COLOR_RGB_Msk     (VG_COLOR_RED_Msk | VG_COLOR_GREEN_Msk | VG_COLOR_BLUE_Msk)

#define VG_COLOR32(r, g, b, a) (0 \
	| (((uint32_t)(r) << VG_COLOR_RED_Pos) & VG_COLOR_RED_Msk) \
	| (((uint32_t)(g) << VG_COLOR_GREEN_Pos) & VG_COLOR_GREEN_Msk) \
	| (((uint32_t)(b) << VG_COLOR_BLUE_Pos) & VG_COLOR_BLUE_Msk) \
	| (((uint32_t)(a) << VG_COLOR_ALPHA_Pos) & VG_COLOR_ALPHA_Msk) \
)

#define VG_TEXT_ALIGN_VER_Pos 0
#define VG_TEXT_ALIGN_VER_Msk (0x03u << VG_TEXT_ALIGN_VER_Pos)
#define VG_TEXT_ALIGN_HOR_Pos 2
#define VG_TEXT_ALIGN_HOR_Msk (0x03u << VG_TEXT_ALIGN_HOR_Pos)
#define VG_TEXT_ALIGN(hor, ver)  (0 \
	| (((uint32_t)(hor) << VG_TEXT_ALIGN_HOR_Pos) & VG_TEXT_ALIGN_HOR_Msk) \
	| (((uint32_t)(ver) << VG_TEXT_ALIGN_VER_Pos) & VG_TEXT_ALIGN_VER_Msk) \
)

#define VG_STROKE_FLAGS_LINE_JOIN_Pos   0
#define VG_STROKE_FLAGS_LINE_JOIN_Msk   (0x03u << VG_STROKE_FLAGS_LINE_JOIN_Pos)
#define VG_STROKE_FLAGS_LINE_CAP_Pos    2
#define VG_STROKE_FLAGS_LINE_CAP_Msk    (0x03u << VG_STROKE_FLAGS_LINE_CAP_Pos)
#define VG_STROKE_FLAGS_AA_Pos          4
#define VG_STROKE_FLAGS_AA_Msk          (0x01u << VG_STROKE_FLAGS_AA_Pos)
#define VG_STROKE_FLAGS_FIXED_WIDTH_Pos 5
#define VG_STROKE_FLAGS_FIXED_WIDTH_Msk (0x01u << VG_STROKE_FLAGS_FIXED_WIDTH_Pos)
#define VG_STROKE_FLAGS(cap, join, aa) (0 \
	| (((uint32_t)(join) << VG_STROKE_FLAGS_LINE_JOIN_Pos) & VG_STROKE_FLAGS_LINE_JOIN_Msk) \
	| (((uint32_t)(cap) << VG_STROKE_FLAGS_LINE_CAP_Pos) & VG_STROKE_FLAGS_LINE_CAP_Msk) \
	| (((uint32_t)(aa) << VG_STROKE_FLAGS_AA_Pos) & VG_STROKE_FLAGS_AA_Msk) \
)

#define VG_FILL_FLAGS_PATH_TYPE_Pos 0
#define VG_FILL_FLAGS_PATH_TYPE_Msk (0x01u << VG_FILL_FLAGS_PATH_TYPE_Pos)
#define VG_FILL_FLAGS_FILL_RULE_Pos 1
#define VG_FILL_FLAGS_FILL_RULE_Msk (0x01u << VG_FILL_FLAGS_FILL_RULE_Pos)
#define VG_FILL_FLAGS_AA_Pos        2
#define VG_FILL_FLAGS_AA_Msk        (0x01u << VG_FILL_FLAGS_AA_Pos)
#define VG_FILL_FLAGS(type, rule, aa) (0 \
	| (((uint32_t)(type) << VG_FILL_FLAGS_PATH_TYPE_Pos) & VG_FILL_FLAGS_PATH_TYPE_Msk) \
	| (((uint32_t)(rule) << VG_FILL_FLAGS_FILL_RULE_Pos) & VG_FILL_FLAGS_FILL_RULE_Msk) \
	| (((uint32_t)(aa) << VG_FILL_FLAGS_AA_Pos) & VG_FILL_FLAGS_AA_Msk) \
)

#define VG_HANDLE(_name) struct _name { uint16_t idx; }
#define VG_HANDLE32(_name) struct _name { uint16_t idx; uint16_t flags; }
#define VG_INVALID_HANDLE { UINT16_MAX }
#define VG_INVALID_HANDLE32 { UINT16_MAX, 0 }

namespace bx
{
struct AllocatorI;
}

namespace bgfx
{
struct TextureHandle;
}

namespace vg
{
typedef uint32_t Color;

Color color4f(float _r, float _g, float _b, float _a);
Color color4ub(uint8_t _r, uint8_t _g, uint8_t _b, uint8_t _a);
Color colorHSL(float _h, float _s, float _l, float _alpha = 1.0f);
Color colorHSB(float _h, float _s, float _b, float _alpha = 1.0f);
Color colorSetAlpha(Color _c, uint8_t _a);
uint8_t colorGetRed(Color _c);
uint8_t colorGetGreen(Color _c);
uint8_t colorGetBlue(Color _c);
uint8_t colorGetAlpha(Color _c);

struct Colors
{
	enum Enum : uint32_t
	{
		Transparent = 0x00000000,
		Black       = 0xFF000000,
		Red         = 0xFF0000FF,
		Green       = 0xFF00FF00,
		Blue        = 0xFFFF0000,
		White       = 0xFFFFFFFF
	};
};

struct TextAlignHor
{
	enum Enum : uint32_t
	{
		Left   = 0,
		Center = 1,
		Right  = 2
	};
};

struct TextAlignVer
{
	enum Enum : uint32_t
	{
		Top      = 0,
		Middle   = 1,
		Baseline = 2,
		Bottom   = 3
	};
};

struct TextAlign
{
	enum Enum : uint32_t
	{
		TopLeft        = VG_TEXT_ALIGN(TextAlignHor::Left, TextAlignVer::Top),
		TopCenter      = VG_TEXT_ALIGN(TextAlignHor::Center, TextAlignVer::Top),
		TopRight       = VG_TEXT_ALIGN(TextAlignHor::Right, TextAlignVer::Top),
		MiddleLeft     = VG_TEXT_ALIGN(TextAlignHor::Left, TextAlignVer::Middle),
		MiddleCenter   = VG_TEXT_ALIGN(TextAlignHor::Center, TextAlignVer::Middle),
		MiddleRight    = VG_TEXT_ALIGN(TextAlignHor::Right, TextAlignVer::Middle),
		BaselineLeft   = VG_TEXT_ALIGN(TextAlignHor::Left, TextAlignVer::Baseline),
		BaselineCenter = VG_TEXT_ALIGN(TextAlignHor::Center, TextAlignVer::Baseline),
		BaselineRight  = VG_TEXT_ALIGN(TextAlignHor::Right, TextAlignVer::Baseline),
		BottomLeft     = VG_TEXT_ALIGN(TextAlignHor::Left, TextAlignVer::Bottom),
		BottomCenter   = VG_TEXT_ALIGN(TextAlignHor::Center, TextAlignVer::Bottom),
		BottomRight    = VG_TEXT_ALIGN(TextAlignHor::Right, TextAlignVer::Bottom),
	};
};

struct LineCap
{
	enum Enum : uint32_t
	{
		Butt   = 0,
		Round  = 1,
		Square = 2,
	};
};

struct LineJoin
{
	enum Enum : uint32_t
	{
		Miter = 0,
		Round = 1,
		Bevel = 2
	};
};

struct StrokeFlags
{
	enum Enum : uint32_t
	{
		// w/o AA
		ButtMiter   = VG_STROKE_FLAGS(LineCap::Butt, LineJoin::Miter, 0),
		ButtRound   = VG_STROKE_FLAGS(LineCap::Butt, LineJoin::Round, 0),
		ButtBevel   = VG_STROKE_FLAGS(LineCap::Butt, LineJoin::Bevel, 0),
		RoundMiter  = VG_STROKE_FLAGS(LineCap::Round, LineJoin::Miter, 0),
		RoundRound  = VG_STROKE_FLAGS(LineCap::Round, LineJoin::Round, 0),
		RoundBevel  = VG_STROKE_FLAGS(LineCap::Round, LineJoin::Bevel, 0),
		SquareMiter = VG_STROKE_FLAGS(LineCap::Square, LineJoin::Miter, 0),
		SquareRound = VG_STROKE_FLAGS(LineCap::Square, LineJoin::Round, 0),
		SquareBevel = VG_STROKE_FLAGS(LineCap::Square, LineJoin::Bevel, 0),

		// w/ AA
		ButtMiterAA   = VG_STROKE_FLAGS(LineCap::Butt, LineJoin::Miter, 1),
		ButtRoundAA   = VG_STROKE_FLAGS(LineCap::Butt, LineJoin::Round, 1),
		ButtBevelAA   = VG_STROKE_FLAGS(LineCap::Butt, LineJoin::Bevel, 1),
		RoundMiterAA  = VG_STROKE_FLAGS(LineCap::Round, LineJoin::Miter, 1),
		RoundRoundAA  = VG_STROKE_FLAGS(LineCap::Round, LineJoin::Round, 1),
		RoundBevelAA  = VG_STROKE_FLAGS(LineCap::Round, LineJoin::Bevel, 1),
		SquareMiterAA = VG_STROKE_FLAGS(LineCap::Square, LineJoin::Miter, 1),
		SquareRoundAA = VG_STROKE_FLAGS(LineCap::Square, LineJoin::Round, 1),
		SquareBevelAA = VG_STROKE_FLAGS(LineCap::Square, LineJoin::Bevel, 1),

		FixedWidth = VG_STROKE_FLAGS_FIXED_WIDTH_Msk // NOTE: Scale independent stroke width
	};
};

struct PathType
{
	enum Enum : uint32_t
	{
		Convex  = 0,
		Concave = 1,
	};
};

struct FillRule
{
	enum Enum : uint32_t
	{
		NonZero = 0,
		EvenOdd = 1,
	};
};

struct FillFlags
{
	enum Enum : uint32_t
	{
		Convex           = VG_FILL_FLAGS(PathType::Convex, FillRule::NonZero, 0),
		ConvexAA         = VG_FILL_FLAGS(PathType::Convex, FillRule::NonZero, 1),
		ConcaveNonZero   = VG_FILL_FLAGS(PathType::Concave, FillRule::NonZero, 0),
		ConcaveEvenOdd   = VG_FILL_FLAGS(PathType::Concave, FillRule::EvenOdd, 0),
		ConcaveNonZeroAA = VG_FILL_FLAGS(PathType::Concave, FillRule::NonZero, 1),
		ConcaveEvenOddAA = VG_FILL_FLAGS(PathType::Concave, FillRule::EvenOdd, 1),
	};
};

struct Winding
{
	enum Enum : uint32_t
	{
		CCW = 0,
		CW  = 1,
	};
};

struct TextBoxFlags
{
	enum Enum : uint32_t
	{
		None               = 0,
		KeepTrailingSpaces = 1 << 0,
	};
};

struct ImageFlags
{
	enum Enum : uint32_t
	{
		Filter_NearestUV = 1 << 0,
		Filter_NearestW  = 1 << 1,
		Filter_LinearUV  = 1 << 2,
		Filter_LinearW   = 1 << 3,
		Clamp_U          =  1 << 10,
		Clamp_V          =  1 << 11,

		// Shortcuts
		Filter_Nearest   = Filter_NearestUV | Filter_NearestW,
		Filter_Bilinear  = Filter_LinearUV | Filter_NearestW,
		Filter_Trilinear = Filter_LinearUV | Filter_LinearW,
		Clamp_UV         = Clamp_U | Clamp_V
	};
};

struct ClipRule
{
	enum Enum : uint32_t
	{
		In  = 0, // fillRule = "nonzero"?
		Out = 1, // fillRule = "evenodd"?
	};
};

struct TransformOrder
{
	enum Enum : uint32_t
	{
		Pre = 0,
		Post = 1,
	};
};

#if VG_CONFIG_UV_INT16
typedef int16_t uv_t;
#else
typedef float uv_t;
#endif

VG_HANDLE32(GradientHandle);
VG_HANDLE32(ImagePatternHandle);
VG_HANDLE(ImageHandle);
VG_HANDLE(FontHandle);
VG_HANDLE(CommandListHandle);

inline bool isValid(GradientHandle _handle)           { return UINT16_MAX != _handle.idx; };
inline bool isValid(ImagePatternHandle _handle)       { return UINT16_MAX != _handle.idx; };
inline bool isValid(ImageHandle _handle)              { return UINT16_MAX != _handle.idx; };
inline bool isValid(FontHandle _handle)               { return UINT16_MAX != _handle.idx; };
inline bool isValid(CommandListHandle _handle)        { return UINT16_MAX != _handle.idx; };

struct Init
{
	uint16_t maxGradients;        // default: 64
	uint16_t maxImagePatterns;    // default: 64
	uint16_t maxFonts;            // default: 8
	uint16_t maxStateStackSize;   // default: 32
	uint16_t maxImages;           // default: 16
	uint16_t maxCommandLists;     // default: 256
	uint32_t maxVBVertices;       // default: 65536
	uint32_t fontAtlasImageFlags; // default: ImageFlags::Filter_Bilinear
	uint32_t maxCommandListDepth; // default: 16
	bool resetViewTransformOnEnd; // default: true
};

struct Stats
{
	uint32_t cmdListMemoryTotal;
	uint32_t cmdListMemoryUsed;
};

struct TextConfig
{
	float fontSize;
	float blur;
	float spacing;
	uint32_t alignment;
	Color color;
	FontHandle fontHandle;
};

struct Mesh
{
	const float* posBuffer;
	const uint32_t* colorBuffer;
	const uint16_t* indexBuffer;
	uint32_t numVertices;
	uint32_t numIndices;
};

// NOTE: The following 2 structs are identical to NanoVG because the rest of the code uses
// them a lot. Until I find a reason to replace them, these will do.
struct TextRow
{
	const char* start;	// Pointer to the input text where the row starts.
	const char* end;	// Pointer to the input text where the row ends (one past the last character).
	const char* next;	// Pointer to the beginning of the next row.
	float width;		// Logical width of the row.
	float minx, maxx;	// Actual bounds of the row. Logical with and bounds can differ because of kerning and some parts over extending.
};

struct GlyphPosition
{
	const char* str;	// Position of the glyph in the input string.
	float x;			// The x-coordinate of the logical glyph position.
	float minx, maxx;	// The bounds of the glyph shape.
};

struct CommandListFlags
{
	enum Enum : uint32_t
	{
		None                = 0,
		Cacheable           = 1 << 0, // Cache the generated geometry in order to avoid retesselation every frame; uses extra memory
		AllowCommandCulling = 1 << 1, // If the scissor rect ends up being zero-sized, don't execute fill/stroke commands.
		CacheScaleInvariant = 1 << 2, // Build the cache once and reuse it at any scale, skipping the per-scale retesselation.
	};
};

struct FontFlags
{
	enum Enum : uint32_t
	{
		None         = 0,
		DontCopyData = 1 << 0, // The calling code will keep the font data alive for as long as the Context is alive so there's no need to copy the data internally.
	};
};

bool init(bx::AllocatorI* _allocator, const Init* _init = NULL);
void shutdown();

void begin(uint16_t _viewID, uint16_t _canvasWidth, uint16_t _canvasHeight, float _devicePixelRatio);
void end();
void frame();
const Stats* getStats();

void beginPath();
void moveTo(float _x, float _y);
void lineTo(float _x, float _y);
void cubicTo(float _c1x, float _c1y, float _c2x, float _c2y, float _x, float _y);
void quadraticTo(float _cx, float _cy, float _x, float _y);
void arcTo(float _x1, float _y1, float _x2, float _y2, float _r);
void arc(float _cx, float _cy, float _r, float _a0, float _a1, Winding::Enum _dir);
void rect(float _x, float _y, float _w, float _h);
void roundedRect(float _x, float _y, float _w, float _h, float _r);
void roundedRectVarying(float _x, float _y, float _w, float _h, float _rtl, float _rtr, float _rbr, float _rbl);
void circle(float _cx, float _cy, float _radius);
void ellipse(float _cx, float _cy, float _rx, float _ry);
void polyline(const float* _coords, uint32_t _numPoints);
void closePath();
void fillPath(Color _color, uint32_t _flags);
void fillPath(GradientHandle _gradient, uint32_t _flags);
void fillPath(ImagePatternHandle _img, Color _color, uint32_t _flags);
void strokePath(Color _color, float _width, uint32_t _flags);
void strokePath(GradientHandle _gradient, float _width, uint32_t _flags);
void strokePath(ImagePatternHandle _img, Color _color, float _width, uint32_t _flags);
void beginClip(ClipRule::Enum _rule);
void endClip();
void resetClip();

GradientHandle createLinearGradient(float _sx, float _sy, float _ex, float _ey, Color _icol, Color _ocol);
GradientHandle createBoxGradient(float _x, float _y, float _w, float _h, float _r, float _f, Color _icol, Color _ocol);
GradientHandle createRadialGradient(float _cx, float _cy, float _inr, float _outr, Color _icol, Color _ocol);
ImagePatternHandle createImagePattern(float _cx, float _cy, float _w, float _h, float _angle, ImageHandle _image);

void setGlobalAlpha(float _alpha);
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

void getTransform(float* _mtx);
void getScissor(float* _rect);

FontHandle createFont(const char* _name, uint8_t* _data, uint32_t _size, uint32_t _flags);
FontHandle getFontByName(const char* _name);
bool setFallbackFont(FontHandle _base, FontHandle _fallback);
void text(const TextConfig& _cfg, float _x, float _y, const char* _str, const char* _end);
void textBox(const TextConfig& _cfg, float _x, float _y, float _breakWidth, const char* _text, const char* _end, uint32_t _textboxFlags);
float measureText(const TextConfig& _cfg, float _x, float _y, const char* _str, const char* _end, float* _bounds);
void measureTextBox(const TextConfig& _cfg, float _x, float _y, float _breakWidth, const char* _text, const char* _end, float* _bounds, uint32_t _flags);
float getTextLineHeight(const TextConfig& _cfg);
int textBreakLines(const TextConfig& _cfg, const char* _str, const char* _end, float _breakRowWidth, TextRow* _rows, int _maxRows, uint32_t _flags);
int textGlyphPositions(const TextConfig& _cfg, float _x, float _y, const char* _text, const char* _end, GlyphPosition* _positions, int _maxPositions);

/*
 * pos: A list of 2D vertices (successive x,y pairs)
 * uv (optional): 1 UV pair for each position. If not specified (NULL), the white-rect UV from the font atlas is used.
 * numVertices: the number of vertices (if uv != NULL, the number of uv pairs is supposed to be equal to this)
 * color: Either a single color (which is replicated on all vertices) or a list of colors, 1 for each vertex.
 * numColors: The number of colors in the color array (can be either 1 for solid color fills or equal to numVertices for per-vertex gradients).
 * indices: A list of indices (triangle list)
 * numIndices: The number of indices
 * img (optional): The image to use for this draw call (created via createImage()) or VG_INVALID_HANDLE in case you don't have an image (colored tri-list).
 */
void indexedTriList(const float* _pos, const uv_t* _uv, uint32_t _numVertices, const Color* _color, uint32_t _numColors, const uint16_t* _indices, uint32_t _numIndices, ImageHandle _img);

bool getImageSize(ImageHandle _handle, uint16_t* _w, uint16_t* _h);
ImageHandle createImage(uint16_t _w, uint16_t _h, uint32_t _flags, const uint8_t* _data);
ImageHandle createImage(uint32_t _flags, const bgfx::TextureHandle& _bgfxTextureHandle);
bool updateImage(ImageHandle _image, uint16_t _x, uint16_t _y, uint16_t _w, uint16_t _h, const uint8_t* _data);
bool destroyImage(ImageHandle _img);
bool isImageValid(ImageHandle _img);

CommandListHandle createCommandList(uint32_t _flags);
void destroyCommandList(CommandListHandle _handle);
void resetCommandList(CommandListHandle _handle);
void submitCommandList(CommandListHandle _handle);

void clBeginPath(CommandListHandle _handle);
void clMoveTo(CommandListHandle _handle, float _x, float _y);
void clLineTo(CommandListHandle _handle, float _x, float _y);
void clCubicTo(CommandListHandle _handle, float _c1x, float _c1y, float _c2x, float _c2y, float _x, float _y);
void clQuadraticTo(CommandListHandle _handle, float _cx, float _cy, float _x, float _y);
void clArcTo(CommandListHandle _handle, float _x1, float _y1, float _x2, float _y2, float _r);
void clArc(CommandListHandle _handle, float _cx, float _cy, float _r, float _a0, float _a1, Winding::Enum _dir);
void clRect(CommandListHandle _handle, float _x, float _y, float _w, float _h);
void clRoundedRect(CommandListHandle _handle, float _x, float _y, float _w, float _h, float _r);
void clRoundedRectVarying(CommandListHandle _handle, float _x, float _y, float _w, float _h, float _rtl, float _rtr, float _rbr, float _rbl);
void clCircle(CommandListHandle _handle, float _cx, float _cy, float _radius);
void clEllipse(CommandListHandle _handle, float _cx, float _cy, float _rx, float _ry);
void clPolyline(CommandListHandle _handle, const float* _coords, uint32_t _numPoints);
void clClosePath(CommandListHandle _handle);
void clIndexedTriList(CommandListHandle _handle, const float* _pos, const uv_t* _uv, uint32_t _numVertices, const Color* _color, uint32_t _numColors, const uint16_t* _indices, uint32_t _numIndices, ImageHandle _img);
void clFillPath(CommandListHandle _handle, Color _color, uint32_t _flags);
void clFillPath(CommandListHandle _handle, GradientHandle _gradient, uint32_t _flags);
void clFillPath(CommandListHandle _handle, ImagePatternHandle _img, Color _color, uint32_t _flags);
void clStrokePath(CommandListHandle _handle, Color _color, float _width, uint32_t _flags);
void clStrokePath(CommandListHandle _handle, GradientHandle _gradient, float _width, uint32_t _flags);
void clStrokePath(CommandListHandle _handle, ImagePatternHandle _img, Color _color, float _width, uint32_t _flags);
void clBeginClip(CommandListHandle _handle, ClipRule::Enum _rule);
void clEndClip(CommandListHandle _handle);
void clResetClip(CommandListHandle _handle);

GradientHandle clCreateLinearGradient(CommandListHandle _handle, float _sx, float _sy, float _ex, float _ey, Color _icol, Color _ocol);
GradientHandle clCreateBoxGradient(CommandListHandle _handle, float _x, float _y, float _w, float _h, float _r, float _f, Color _icol, Color _ocol);
GradientHandle clCreateRadialGradient(CommandListHandle _handle, float _cx, float _cy, float _inr, float _outr, Color _icol, Color _ocol);
ImagePatternHandle clCreateImagePattern(CommandListHandle _handle, float _cx, float _cy, float _w, float _h, float _angle, ImageHandle _image);

void clPushState(CommandListHandle _handle);
void clPopState(CommandListHandle _handle);
void clResetScissor(CommandListHandle _handle);
void clSetScissor(CommandListHandle _handle, float _x, float _y, float _w, float _h);
void clIntersectScissor(CommandListHandle _handle, float _x, float _y, float _w, float _h);
void clTransformIdentity(CommandListHandle _handle);
void clTransformScale(CommandListHandle _handle, float _x, float _y);
void clTransformTranslate(CommandListHandle _handle, float _x, float _y);
void clTransformRotate(CommandListHandle _handle, float _ang_rad);
void clTransformMult(CommandListHandle _handle, const float* _mtx, TransformOrder::Enum _order);
void clSetViewBox(CommandListHandle _handle, float _x, float _y, float _w, float _h);
void clSetGlobalAlpha(CommandListHandle _handle, float _alpha);

void clText(CommandListHandle _handle, const TextConfig& _cfg, float _x, float _y, const char* _str, const char* _end);
void clTextBox(CommandListHandle _handle, const TextConfig& _cfg, float _x, float _y, float _breakWidth, const char* _str, const char* _end, uint32_t _textboxFlags);

void clSubmitCommandList(CommandListHandle _parent, CommandListHandle _child);

TextConfig makeTextConfig(const char* _fontName, float _fontSize, uint32_t _alignment, Color _color, float _blur = 0.0f, float _spacing = 0.0f);
TextConfig makeTextConfig(FontHandle _fontHandle, float _fontSize, uint32_t _alignment, Color _color, float _blur = 0.0f, float _spacing = 0.0f);
void text(FontHandle _fontHandle, float _fontSize, uint32_t _alignment, Color _color, float _x, float _y, const char* _str, const char* _end);
void textBox(FontHandle _fontHandle, float _fontSize, uint32_t _alignment, Color _color, float _x, float _y, float _breakWidth, const char* _str, const char* _end, uint32_t _textboxFlags);
float measureText(FontHandle _fontHandle, float _fontSize, uint32_t _alignment, float _x, float _y, const char* _str, const char* _end, float* _bounds);
void measureTextBox(FontHandle _fontHandle, float _fontSize, uint32_t _alignment, float _x, float _y, float _breakWidth, const char* _str, const char* _end, float* _bounds, uint32_t _textboxFlags);
float getTextLineHeight(FontHandle _fontHandle, float _fontSize, uint32_t _alignment);
int textBreakLines(FontHandle _fontHandle, float _fontSize, uint32_t _alignment, const char* _str, const char* _end, float _breakRowWidth, TextRow* _rows, int _maxRows, uint32_t _flags);
int textGlyphPositions(FontHandle _fontHandle, float _fontSize, uint32_t _alignment, float _x, float _y, const char* _str, const char* _end, GlyphPosition* _positions, int _maxPositions);
}

#include "inline/vg.inl"

#endif // VG_H
