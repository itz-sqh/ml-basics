cmake_minimum_required(VERSION 3.31)
project(basic_ml)

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable(basic_ml main.cpp
mli.h
        png_to_raw.h)
