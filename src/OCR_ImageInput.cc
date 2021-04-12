/*
 * OCR_ImageInput.cc
 *
 *  Created on: Mar 31, 2021
 *      Author: sungho
 */

#include "OCR_ImageInput.hh"

ImageInput::ImageInput() {
}

ImageInput::ImageInput(int device) {
	capture.open(device);
	std::cout << ">>> OCR Camera specifications" << std::endl;
	std::cout << "    Camera FPS       : " << capture.get(cv::CAP_PROP_FPS) << std::endl;
	std::cout << "    Camera Resolution: " << capture.get(cv::CAP_PROP_FRAME_WIDTH) << "x" <<
										      capture.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl << std::endl;
}

ImageInput::~ImageInput() {
}

bool ImageInput::nextImage() {
	// read image from camera
	bool success = capture.read(img);
	return success;
}

cv::Mat& ImageInput::getImage() {
	return img;
}

void ImageInput::setImage(cv::Mat& _img) {
	img = _img;
}
