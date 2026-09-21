# Vendored lodepng

PNG decoder used by `server/src/common/vision/image_decode.cpp`.

- Source: https://github.com/lvandeve/lodepng
- Commit: `ed6fe5825c6a4fbb7f58ab35a4231c7543cd452a`
- Archive SHA256: `c2459a3f9145258f901d262576f7a56ca08087d3b3efeee3ae033c0952120803`
- Files: `lodepng.cpp`, `lodepng.h`, `LICENSE` (zlib), unmodified.

Vendored rather than downloaded: the project has no release archives, and
GitHub's on-the-fly commit archives fail often enough to break CI.
