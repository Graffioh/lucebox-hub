# JPEG and PNG decoders behind common/vision/image_decode: libjpeg-turbo from
# its pinned release archive, lodepng vendored. License texts are in
# ImageCodecs.NOTICES.md, deps/lodepng/LICENSE and the libjpeg-turbo archive.
include_guard(GLOBAL)

include(ExternalProject)

# DOWNLOAD_EXTRACT_TIMESTAMP exists from CMake 3.24; older releases would read
# it as part of URL_HASH.
set(IMAGE_CODEC_EXTRACT_TIMESTAMP)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(IMAGE_CODEC_EXTRACT_TIMESTAMP DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
endif()

set(IMAGE_CODEC_JPEG_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/libjpeg-turbo-prefix)
set(IMAGE_CODEC_JPEG_ARCHIVE_NAME jpeg)
if(MSVC OR CMAKE_C_SIMULATE_ID STREQUAL "MSVC")
    set(IMAGE_CODEC_JPEG_ARCHIVE_NAME jpeg-static)
endif()
set(IMAGE_CODEC_JPEG_ARCHIVE
    ${IMAGE_CODEC_JPEG_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${IMAGE_CODEC_JPEG_ARCHIVE_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX})
file(MAKE_DIRECTORY ${IMAGE_CODEC_JPEG_PREFIX}/include)
ExternalProject_Add(libjpeg_turbo_external
    URL https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.1.4.1/libjpeg-turbo-3.1.4.1.tar.gz
    URL_HASH SHA256=ecae8008e2cc9ade2f2c1bb9d5e6d4fb73e7c433866a056bd82980741571a022
    ${IMAGE_CODEC_EXTRACT_TIMESTAMP}
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_INSTALL_PREFIX=${IMAGE_CODEC_JPEG_PREFIX}
        -DCMAKE_INSTALL_LIBDIR=lib
        -DENABLE_SHARED=OFF
        -DENABLE_STATIC=ON
        -DWITH_TOOLS=OFF
        -DWITH_TESTS=OFF
        -DWITH_SIMD=OFF
        -DWITH_TURBOJPEG=OFF
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --parallel 2
    BUILD_BYPRODUCTS ${IMAGE_CODEC_JPEG_ARCHIVE})
add_library(image_codec_jpeg STATIC IMPORTED GLOBAL)
set_target_properties(image_codec_jpeg PROPERTIES
    IMPORTED_LOCATION ${IMAGE_CODEC_JPEG_ARCHIVE}
    INTERFACE_INCLUDE_DIRECTORIES ${IMAGE_CODEC_JPEG_PREFIX}/include)
add_dependencies(image_codec_jpeg libjpeg_turbo_external)

# lodepng is vendored (server/deps/lodepng): two source files, no release archives upstream.
set(IMAGE_CODEC_PNG_DIR "${CMAKE_CURRENT_LIST_DIR}/../deps/lodepng")
add_library(image_codec_png STATIC ${IMAGE_CODEC_PNG_DIR}/lodepng.cpp)
target_include_directories(image_codec_png PUBLIC ${IMAGE_CODEC_PNG_DIR})

# Expose the original JPEG notices for production installation after the
# dependency build has downloaded its hash-verified source archive.
ExternalProject_Get_Property(libjpeg_turbo_external SOURCE_DIR)
set(IMAGE_CODEC_JPEG_SOURCE_DIR "${SOURCE_DIR}")
unset(SOURCE_DIR)
