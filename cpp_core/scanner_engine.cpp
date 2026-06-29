#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>

namespace py = pybind11;

cv::Mat numpy_to_mat(py::array_t<uint8_t>& input) {
    py::buffer_info buf = input.request();
    int rows = buf.shape[0];
    int cols = buf.shape[1];
    int channels = buf.ndim == 2 ? 1 : buf.shape[2];
    int type = channels == 1 ? CV_8UC1 : CV_8UC3;
    return cv::Mat(rows, cols, type, buf.ptr);
}

py::array_t<uint8_t> mat_to_numpy(cv::Mat& img) {
    std::vector<ssize_t> shape;
    if (img.channels() == 1) {
        shape = {img.rows, img.cols};
    } else {
        shape = {img.rows, img.cols, img.channels()};
    }
    return py::array_t<uint8_t>(shape, img.data);
}

// FUNGIS UTAMA YANG DIPANGGIL PYTHON
py::array_t<uint8_t> enhance_for_ocr(py::array_t<uint8_t> input_array) {
    cv::Mat img = numpy_to_mat(input_array);
    cv::Mat gray;
    
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = img.clone();
    }

    // Upscale 2x
    cv::resize(gray, gray, cv::Size(), 2.0, 2.0, cv::INTER_CUBIC);

    // CLAHE
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(1.5, cv::Size(8, 8));
    cv::Mat clahe_img;
    clahe->apply(gray, clahe_img);

    // Denoise Ringan (FastNlMeansDenoising)
    cv::Mat denoised;
    cv::fastNlMeansDenoising(clahe_img, denoised, 12, 7, 21);

    // Otsu Threshold
    cv::Mat otsu;
    cv::threshold(denoised, otsu, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    return mat_to_numpy(otsu);
}

// Mendaftarkan modul ke Python
PYBIND11_MODULE(scanner_cpp, m) {
    m.doc() = "C++ Image Processing Core for Smart Scanner";
    m.def("enhance_for_ocr", &enhance_for_ocr, "Enhance image for OCR using OpenCV C++");
}