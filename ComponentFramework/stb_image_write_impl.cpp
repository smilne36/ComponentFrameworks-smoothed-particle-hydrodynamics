// stb's HDR writer calls plain sprintf, which MSVC's /sdl flag elevates to an
// error (C4996). Suppress the CRT deprecation for this third-party TU only.
#define _CRT_SECURE_NO_WARNINGS
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
