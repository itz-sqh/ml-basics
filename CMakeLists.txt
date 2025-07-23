cmake_minimum_required(VERSION 3.31)
project(ml-basics)

set(CMAKE_CXX_STANDARD 23)

set(STB_IMAGE_URL "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h")
set(STB_IMAGE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/stb_image.h")

if(NOT EXISTS "${STB_IMAGE_FILE}")
    message(STATUS "stb_image.h not found, downloading stb_image.h...")
    file(DOWNLOAD
            ${STB_IMAGE_URL}
            ${STB_IMAGE_FILE}
            SHOW_PROGRESS
            STATUS DOWNLOAD_STATUS
    )
    list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
    if(STATUS_CODE)
        message(WARNING "Failed to download stb_image.h - please download manually")
    endif()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(${PROJECT_NAME} examples/digit_recognizer.cpp)
