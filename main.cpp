#include<bits/stdc++.h>
#include "mli.h"


// uses stb_image.h
#include "png_to_raw.h"

using Mat = Matrix<float>;
using NN = Neural<float>;

using namespace std;
namespace fs = std::filesystem;



//MNIST dataset
const string training_base_path = "C:/Users/Egor/Desktop/training";
const string testing_base_path = "C:/Users/Egor/Desktop/testing";


constexpr int image_size = 28;
constexpr int digits = 10;

void load_dataset(const string &base_path, Mat& tin, Mat& tout) {
    assert(fs::exists(base_path) && ("Directory does not exist: " + base_path).data());
    vector<pair<unique_ptr<uint8_t[]>,int>> samples;
    for (int digit = 0; digit < digits; digit++) {
        string dir = base_path+"/"+to_string(digit);
        assert(fs::exists(dir) && ("Directory does not exist: " + dir).data());
        for (const auto& entry : fs::directory_iterator(dir)) {
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
            tin.a[i][j] = samples[i].first[j] / 255.0f;
        }
        tout.a[i][samples[i].second] = 1.0f;
    }
}

float check(NN& nn) {

    Mat tin,tout;
    load_dataset(testing_base_path,tin, tout);
    int cnt = 0;
    for (size_t i = 0; i < tin.n; i++) {
        nn.activation[0] = tin.row(i);
        nn.forward();

        auto predict = nn.activation.back()[0];
        auto res = 0;
        for (int j = 0; j < digits; j++) {
            if (tout[i][j] == 1.0f) res = j;
        }
        for (int j = 0; j < digits; j++) {
            if (predict[j] > 0.9f && res == j)
                cnt++;
        }
    }
    return static_cast<float>(cnt)/static_cast<float>(tin.n);
}


int main() {
    srand(time(0));
    Mat tin,tout;
    load_dataset(training_base_path,tin,tout);

    constexpr float rate = 1e-1;
    const vector<size_t> layers{784,16,16,digits};
    NN nn(layers);
    NN grad(layers);

    nn.xavier_init();


    cout << "0: c = " << nn.cost(tin,tout) << endl;
    for (int i = 0; i < 50; i++) {
        nn.batch_process(grad,tin,tout,rate,64,10,true);
        cout << i + 1 << ": c = " << nn.cost(tin,tout) << endl;
    }
    cout << check(nn) << endl;
    cout << nn << endl;
}
