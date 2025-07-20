#pragma once

#define STB_IMAGE_IMPLEMENTATION
#include"stb_image.h"

inline std::unique_ptr<uint8_t[]> png_to_raw(const char* filename) {
    int x,y,n;
    uint8_t* pixels = stbi_load(filename,&x,&y,&n,0);
    assert(pixels != nullptr && ("Could not read image: " + std::string(filename)).data());
    assert(n == 1 && "Only 8 bit grayscale images are supported");
    const size_t size = x * y;
    auto res = std::make_unique<uint8_t[]>(size);
    std::copy_n(pixels,size,res.get());
    return res;
}