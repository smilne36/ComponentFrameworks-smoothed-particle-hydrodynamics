#include "Window.h"
#include "Debug.h"
#include <SDL.h>

Window::Window() : window{ nullptr }, context{ nullptr }, width{ 0 }, height{ 0 } {}

Window::~Window() {
	OnDestroy();
}

bool Window::OnCreate(std::string name_, int width_, int height_) {
	if (SDL_Init(SDL_INIT_VIDEO) < 0) {
		Debug::FatalError("Failed to initialize SDL", __FILE__, __LINE__);
		return false;
	}

	this->width = width_;
	this->height = height_;
	window = SDL_CreateWindow(name_.c_str(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
		width, height, SDL_WINDOW_OPENGL);

	if (window == nullptr) {
		Debug::FatalError("Failed to create a window", __FILE__, __LINE__);
		return false;
	}

	// Set context attributes BEFORE creating context
	setAttributes(4, 5); // Target OpenGL 4.5 Core
	context = SDL_GL_CreateContext(window);

	if (!gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress)) {
		Debug::FatalError("Failed to initialize GLAD", __FILE__, __LINE__);
		return false;
	}

	getInstalledOpenGLInfo();
	glViewport(0, 0, width, height);
	return true;
}

void Window::OnDestroy() {
	SDL_GL_DeleteContext(context);
	SDL_DestroyWindow(window);
	window = nullptr;
}

void Window::getInstalledOpenGLInfo(int* major, int* minor) {
	const GLubyte* version = glGetString(GL_VERSION);
	const GLubyte* vendor = glGetString(GL_VENDOR);
	const GLubyte* renderer = glGetString(GL_RENDERER);
	const GLubyte* glslVersion = glGetString(GL_SHADING_LANGUAGE_VERSION);

	glGetIntegerv(GL_MAJOR_VERSION, major);
	glGetIntegerv(GL_MINOR_VERSION, minor);
	Debug::Info("OpenGL version: " + std::string((char*)version), __FILE__, __LINE__);
	Debug::Info("Graphics card vendor: " + std::string((char*)vendor), __FILE__, __LINE__);
	Debug::Info("Graphics card name: " + std::string((char*)renderer), __FILE__, __LINE__);
	Debug::Info("GLSL version: " + std::string((char*)glslVersion), __FILE__, __LINE__);
}

void Window::getInstalledOpenGLInfo() {
	int major, minor;
	getInstalledOpenGLInfo(&major, &minor);
}

void Window::setAttributes(int major_, int minor_) {
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, major_);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, minor_);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 32);
	SDL_GL_SetSwapInterval(1); // V-sync
}
