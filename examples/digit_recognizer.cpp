#include<iostream>
#include<memory>
#include<vector>
#include<filesystem>
#include"neural.h"
#include"png_to_raw.h"

using Mat = Matrix<float>;
using NN = Neural<float, Sigmoid<float>>;

// Path to MNIST dataset with pngs
const std::string training_base_path = "C:/Users/sq/Desktop/training";
const std::string testing_base_path = "C:/Users/sq/Desktop/testing";

constexpr int image_size = 28;
constexpr int digits = 10;

void load_dataset(const std::string &base_path, Mat& tin, Mat& tout) {
    assert(std::filesystem::exists(base_path) && ("Directory does not exist: " + base_path).data());

    std::vector<std::pair<std::unique_ptr<uint8_t[]>,int>> samples;

    for (int digit = 0; digit < digits; digit++) {
        // {base_path}/{digit}/{png}
        std::string dir = base_path+"/"+std::to_string(digit);

        assert(std::filesystem::exists(dir) && ("Directory does not exist: " + dir).data());

        for (const auto& entry : std::filesystem::directory_iterator(dir)) {
            if (entry.path().extension() == ".png") {
                auto pixels = png_to_raw(entry.path().string().c_str());
                samples.emplace_back(move(pixels), digit);
            }
        }
    }

    tin = Mat(samples.size(),image_size*image_size);
    tout = Mat(samples.size(),digits);

    for (size_t i = 0; i < samples.size(); ++i) {
        for (size_t j = 0; j < image_size*image_size; ++j) {
            tin.a[i][j] = static_cast<float>(samples[i].first[j]) / 255.0f;
        }
        tout.a[i][samples[i].second] = 1.0f;
    }
}

float check(NN& nn) {
    Mat tin, tout;

    load_dataset(testing_base_path,tin, tout);

    int cnt = 0;

    for (size_t i = 0; i < tin.n; i++) {
        nn.activation[0] = tin.row(i);
        nn.forward();
        auto prediction = nn.activation.back()[0];

        auto res = 0;
        for (int j = 0; j < digits; j++) {
            if (tout[i][j] == 1.0f) res = j;
        }

        for (int j = 0; j < digits; j++) {
            if (prediction[j] > 0.9f && res == j) {
                cnt++;
            }
        }
    }
    return static_cast<float>(cnt)/static_cast<float>(tin.n);
}


int main() {
    srand(time(nullptr));

    Mat tin,tout;
    load_dataset(training_base_path,tin,tout);

    const std::vector<size_t> layers{784, 16, 16, digits};
    NN nn(layers);
    NN grad(layers);

    nn.xavier_init();

    std::cout << "Initial cost = " << nn.cost(tin, tout) << std::endl;

    for (int i = 0; i < 50; i++) {
        constexpr float rate = 1e-1;
        nn.batch_process(grad, tin, tout, rate, 64, 10, true);
        std::cout << i + 1 << ": cost = " << nn.cost(tin, tout) << std::endl;
    }


    std::cout << "Precision = " << check(nn) << "%";

}
