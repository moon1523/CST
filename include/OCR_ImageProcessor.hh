#ifndef INCLUDE_OCR_IMAGEPROCESSOR_HH_
#define INCLUDE_OCR_IMAGEPROCESSOR_HH_

#include <iostream>
#include <fstream>
#include <vector>

#include <opencv2/imgproc/imgproc.hpp>
#include "OCR_ImageInput.hh"
#include "OCR_Config.hh"

class ROIBox
{
public:
	ROIBox() : roiBox(0) { }
	std::vector<cv::Rect> getROIBox() { return roiBox; }
	void setROIBox(std::vector<cv::Rect>& _roiBox) {
		roiBox = _roiBox; std::cout << "    >> ROI box #: " << roiBox.size() << std::endl;
	}
private:
	std::vector<cv::Rect> roiBox;
};

class sortRectByX {
public:
	bool operator()(cv::Rect const& a, cv::Rect const& b) const {
		return a.x < b.x;
	}
};

class ImageProcessor {
public:
	ImageProcessor(const Config& config);
	virtual ~ImageProcessor();

	bool Process(ROIBox* roi);

	int ShowImage();
	void SetInput(cv::Mat& _img) { img = _img; }
	bool GetPowerOn() { return powerOn; }

	const std::vector<cv::Mat>& GetOutputkV()   { return digits_kV; };
	const std::vector<cv::Mat>& GetOutputmA()   { return digits_mA; };
	const std::vector<cv::Mat>& GetOutputDAP()  { return digits_DAP;};


	void DebugWindow(bool bval = true) { debugWindow = bval; }
	void DebugEdges (bool bval = true) { debugEdges  = bval; }
	void DebugDigits(bool bval = true) { debugDigits = bval; }
	void DebugPower (bool bval = true) { debugPower  = bval; }
	void DebugOCR   (bool bval = true) { debugOCR    = bval; }



private:
	cv::Mat BinaryFiltering();
	void FindAlignedBoxes(std::vector<cv::Rect>::const_iterator begin,
			std::vector<cv::Rect>::const_iterator end, std::vector<cv::Rect>& result);
	void FilterContours(std::vector<std::vector<cv::Point> >& contours,
			std::vector<cv::Rect>& boundingBoxes, std::vector<std::vector<cv::Point> >& filteredContours);
	void FindCounterDigits(ROIBox* _roi);



	cv::Mat img;
	cv::Mat imgGray;
	cv::Mat imgBin;

	std::vector<cv::Mat> digits;
	std::vector<cv::Mat> digits_kV;
	std::vector<cv::Mat> digits_mA;
	std::vector<cv::Mat> digits_DAP;

	Config config;

	bool debugWindow;
	bool debugEdges;
	bool debugDigits;
	bool debugPower;
	bool debugOCR;

	bool powerOn;


	int key;

};

#endif /* INCLUDE_OCR_IMAGEPROCESSOR_HH_ */
