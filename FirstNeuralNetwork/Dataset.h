#pragma once
#include <filesystem>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include "ImageInput.h"

namespace fs = std::filesystem;

inline void addClassSamples(const fs::path& dir, int label,
	std::vector<std::vector<double>>& rows) {
	for (auto& p : fs::recursive_directory_iterator(dir)) {
		if (!p.is_regular_file()) continue;
		auto ext = p.path().extension().string();
		for (auto& c : ext) c = char(::tolower(unsigned char(c)));
		if (ext != ".png" && ext != ".jpg" && ext != ".jpeg" && ext != ".bmp") continue;

		auto x = loadImage400x400(p.path().string());
		x.push_back(double(label));
		rows.push_back(std::move(x));
	}
}

inline void shuffle_in_place(std::vector<std::vector<double>>& v) {
	std::mt19937 rng(std::random_device{}());
	std::shuffle(v.begin(), v.end(), rng);
 }

struct LabeledSet {
	std::vector<std::vector<double>> train;
	std::vector<std::vector<double>> test; 
};

inline LabeledSet buildDataSet2Class(const std::string& dogsDir, const std::string& catsDir,
	double trainRatio = 0.8) {
	std::vector<std::vector<double>> rows;
	addClassSamples(dogsDir, 0, rows);
	addClassSamples(catsDir, 1, rows);
	std::cout << "Total Samples: " << rows.size() << '\n';
	shuffle_in_place(rows);

	size_t ntrain = std::max<size_t>(1, size_t(rows.size() * trainRatio));
	LabeledSet z;
	ntrain = std::min(ntrain, rows.size());
	z.train.assign(rows.begin(), rows.begin() + ntrain);
	z.test.assign(rows.begin() + ntrain, rows.end());
	return z;
}