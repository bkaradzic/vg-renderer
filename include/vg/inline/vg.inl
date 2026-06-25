#ifndef VG_H
#	error "Must be included from vg/vg.h"
#endif

#include <bx/math.h>

namespace vg
{
inline Color color4f(float _r, float _g, float _b, float _a)
{
	uint8_t rb = (uint8_t)(bx::clamp<float>(_r, 0.0f, 1.0f) * 255.0f);
	uint8_t gb = (uint8_t)(bx::clamp<float>(_g, 0.0f, 1.0f) * 255.0f);
	uint8_t bb = (uint8_t)(bx::clamp<float>(_b, 0.0f, 1.0f) * 255.0f);
	uint8_t ab = (uint8_t)(bx::clamp<float>(_a, 0.0f, 1.0f) * 255.0f);
	return VG_COLOR32(rb, gb, bb, ab);
}

inline Color color4ub(uint8_t _r, uint8_t _g, uint8_t _b, uint8_t _a)
{
	return VG_COLOR32(_r, _g, _b, _a);
}

inline Color colorHSB(float _hue, float _sat, float _brightness, float _alpha)
{
	const float d = 1.0f / 6.0f;

	const float C = _brightness * _sat;
	const float X = C * (1.0f - bx::abs(bx::mod((_hue * 6.0f), 2.0f) - 1.0f));
	const float m = _brightness - C;

	float fred = 0.0f;
	float fgreen = 0.0f;
	float fblue = 0.0f;
	if (_hue <= d) {
		fred = C;
		fgreen = X;
	} else if (_hue > d && _hue <= 2.0f * d) {
		fred = X;
		fgreen = C;
	} else if (_hue > 2.0f * d && _hue <= 3.0f * d) {
		fgreen = C;
		fblue = X;
	} else if (_hue > 3.0f * d && _hue <= 4.0f * d) {
		fgreen = X;
		fblue = C;
	} else if (_hue > 4.0f * d && _hue <= 5.0f * d) {
		fred = X;
		fblue = C;
	} else {
		fred = C;
		fblue = X;
	}

	uint32_t r = (uint32_t)bx::floor((fred + m) * 255.0f);
	uint32_t g = (uint32_t)bx::floor((fgreen + m) * 255.0f);
	uint32_t b = (uint32_t)bx::floor((fblue + m) * 255.0f);
	uint32_t a = (uint32_t)bx::floor(255.0f * _alpha);

	return (a << 24) | (b << 16) | (g << 8) | (r);
}

inline float _hue_helper(float _h, float _m1, float _m2)
{
	if (_h < 0) _h += 1;
	if (_h > 1) _h -= 1;
	if (_h < 1.0f/6.0f)
		return _m1 + (_m2 - _m1) * _h * 6.0f;
	else if (_h < 3.0f/6.0f)
		return _m2;
	else if (_h < 4.0f/6.0f)
		return _m1 + (_m2 - _m1) * (2.0f/3.0f - _h) * 6.0f;
	return _m1;
}

inline Color colorHSL(float _hue, float _sat, float _lightness, float _alpha)
{
	float m1, m2;
	_hue = bx::mod(_hue, 1.0f);
	if (_hue < 0.0f) _hue += 1.0f;
	_sat = bx::clamp<float>(_sat, 0.0f, 1.0f);
	_lightness = bx::clamp<float>(_lightness, 0.0f, 1.0f);
	m2 = _lightness <= 0.5f ? (_lightness * (1 + _sat)) : (_lightness + _sat - _lightness * _sat);
	m1 = 2 * _lightness - m2;
	float fr = bx::clamp<float>(_hue_helper(_hue + 1.0f/3.0f, m1, m2), 0.0f, 1.0f);
	float fg = bx::clamp<float>(_hue_helper(_hue, m1, m2), 0.0f, 1.0f);
	float fb = bx::clamp<float>(_hue_helper(_hue - 1.0f/3.0f, m1, m2), 0.0f, 1.0f);
	float fa = _alpha;

	uint32_t r = (uint32_t)bx::floor(fr * 255.0f);
	uint32_t g = (uint32_t)bx::floor(fg * 255.0f);
	uint32_t b = (uint32_t)bx::floor(fb * 255.0f);
	uint32_t a = (uint32_t)bx::floor(fa * 255.0f);

	return (a << 24) | (b << 16) | (g << 8) | (r);
}

inline Color colorSetAlpha(Color _c, uint8_t _a)
{
	return (_c & VG_COLOR_RGB_Msk) | (((uint32_t)_a << VG_COLOR_ALPHA_Pos) & VG_COLOR_ALPHA_Msk);
}

inline uint8_t colorGetAlpha(Color _c)
{
	return (uint8_t)((_c & VG_COLOR_ALPHA_Msk) >> VG_COLOR_ALPHA_Pos);
}

inline uint8_t colorGetRed(Color _c)
{
	return (uint8_t)((_c & VG_COLOR_RED_Msk) >> VG_COLOR_RED_Pos);
}

inline uint8_t colorGetGreen(Color _c)
{
	return (uint8_t)((_c & VG_COLOR_GREEN_Msk) >> VG_COLOR_GREEN_Pos);
}

inline uint8_t colorGetBlue(Color _c)
{
	return (uint8_t)((_c & VG_COLOR_BLUE_Msk) >> VG_COLOR_BLUE_Pos);
}

// Text helpers
inline TextConfig makeTextConfig(const char* _fontName, float _fontSize, uint32_t _alignment, Color _color, float _blur, float _spacing)
{
	TextConfig cfg;
	cfg.fontHandle = getFontByName(_fontName);
	cfg.fontSize = _fontSize;
	cfg.alignment = _alignment;
	cfg.color = _color;
	cfg.blur = _blur;
	cfg.spacing = _spacing;
	return cfg;
}

inline TextConfig makeTextConfig(FontHandle _fontHandle, float _fontSize, uint32_t _alignment, Color _color, float _blur, float _spacing)
{
	TextConfig cfg;
	cfg.fontHandle = _fontHandle;
	cfg.fontSize = _fontSize;
	cfg.alignment = _alignment;
	cfg.color = _color;
	cfg.blur = _blur;
	cfg.spacing = _spacing;
	return cfg;
}

inline void text(FontHandle _fontHandle, float _fontSize, uint32_t _alignment, Color _color, float _x, float _y, const char* _str, const char* _end)
{
	text(makeTextConfig(_fontHandle, _fontSize, _alignment, _color), _x, _y, _str, _end);
}

inline void textBox(FontHandle _fontHandle, float _fontSize, uint32_t _alignment, Color _color, float _x, float _y, float _breakWidth, const char* _str, const char* _end, uint32_t _textboxFlags)
{
	textBox(makeTextConfig(_fontHandle, _fontSize, _alignment, _color), _x, _y, _breakWidth, _str, _end, _textboxFlags);
}

inline float measureText(FontHandle _fontHandle, float _fontSize, uint32_t _alignment, float _x, float _y, const char* _str, const char* _end, float* _bounds)
{
	return measureText(makeTextConfig(_fontHandle, _fontSize, _alignment, Colors::Transparent), _x, _y, _str, _end, _bounds);
}

inline void measureTextBox(FontHandle _fontHandle, float _fontSize, uint32_t _alignment, float _x, float _y, float _breakWidth, const char* _str, const char* _end, float* _bounds, uint32_t _flags)
{
	measureTextBox(makeTextConfig(_fontHandle, _fontSize, _alignment, Colors::Transparent), _x, _y, _breakWidth, _str, _end, _bounds, _flags);
}

inline float getTextLineHeight(FontHandle _fontHandle, float _fontSize, uint32_t _alignment)
{
	return getTextLineHeight(makeTextConfig(_fontHandle, _fontSize, _alignment, Colors::Transparent));
}

inline int textBreakLines(FontHandle _fontHandle, float _fontSize, uint32_t _alignment, const char* _str, const char* _end, float _breakRowWidth, TextRow* _rows, int _maxRows, uint32_t _flags)
{
	return textBreakLines(makeTextConfig(_fontHandle, _fontSize, _alignment, Colors::Transparent), _str, _end, _breakRowWidth, _rows, _maxRows, _flags);
}

inline int textGlyphPositions(FontHandle _fontHandle, float _fontSize, uint32_t _alignment, float _x, float _y, const char* _str, const char* _end, GlyphPosition* _positions, int _maxPositions)
{
	return textGlyphPositions(makeTextConfig(_fontHandle, _fontSize, _alignment, Colors::Transparent), _x, _y, _str, _end, _positions, _maxPositions);
}
}
