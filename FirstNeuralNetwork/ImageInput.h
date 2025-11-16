#pragma once
#include <vector>
#include <string>
#include <stdexcept>
#include "stb_image.h"

// Nearest-Neighbor Resize
inline void resizeNeuralNet_8u(const unsigned char* src, int srcWidth, int srcHeight,
	unsigned char* dst, int dstWidth, int dstHeight) {
	for (int y = 0; y < dstHeight; ++y) {
		int sy = (y * srcHeight) / dstHeight;
		const unsigned char* srow = src + sy * srcWidth;
		unsigned char* drow = dst + y * dstWidth;
		for (int x = 0; x < dstWidth; ++x) {
			int sx = (x * srcWidth) / dstWidth;
			drow[x] = srow[sx];
		}
	}
}

// Load any image (png/jpg, etc.), force grayscale, resize to 400x400, 
// normalize to [0, 1], and flatten to vector<double> (size 160000).
inline std::vector<double> loadImage400x400(const std::string& path) {
	int w = 0, h = 0, comp = 0;
	// Force 1-channel grayscale
	unsigned char* img = stbi_load(path.c_str(), &w, &h, &comp, 1);
	if (!img) {
		std::cerr << "STB failed: " << stbi_failure_reason() << "\n";
		throw std::runtime_error("Failed to load image: " + path);
	}

	std::cout << "Loaded " << path << " ->w = " << w << " h = " << h <<
		" comp = " << comp << "\n";

	// Ensure 400 x 400
	const int DW = 400, DH = 400;
	std::vector<unsigned char> work(DW * DH);
	if (w == DW && h == DH)
		std::memcpy(work.data(), img, static_cast<size_t>(DW) * DH);
	else
		resizeNeuralNet_8u(img, w, h, work.data(), DW, DH);
	stbi_image_free(img);

	// Normalize to [0, 1] doubles and flattens
	std::vector<double> out;
	out.resize(DW * DH);
	for (size_t i = 0; i < out.size(); ++i)
		out[i] = static_cast<double>(work[i]) / 255.0;

	return out;
}