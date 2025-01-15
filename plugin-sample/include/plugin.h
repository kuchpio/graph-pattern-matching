#pragma once
#include "core.h"

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else // __WIN32
#define EXPORT
#endif

extern "C" EXPORT core::IPatternMatcher* GetPatternMatcher();
