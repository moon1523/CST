#include "OCR_ImageProcessor.hh"

ImageProcessor::ImageProcessor(const Config& _config)
: debugWindow(false), debugEdges(false), debugDigits(false), debugPower(false), debugOCR(false),
  config(_config), powerOn(false), key(0)
{


}

ImageProcessor::~ImageProcessor() {

}

bool ImageProcessor::Process(ROIBox* roi)
{
	digits.clear();
	digits_kV.clear();
	digits_mA.clear();
	digits_DAP.clear();

	// convert to gray
	cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

	// find and isolate counter digits
	FindCounterDigits(roi);

	if (debugWindow)
		ShowImage();


	return powerOn;
}

int ImageProcessor::ShowImage() {
	cv::imshow("ImageProcessor", img);
	key = cv::waitKey(1);
	return key;
}


cv::Mat ImageProcessor::BinaryFiltering() {
	cv::Mat bin;
	cv::threshold(imgGray, bin, config.getBinaryThreshold(), 255, cv::THRESH_BINARY);
	return bin;
}

void ImageProcessor::FindAlignedBoxes(std::vector<cv::Rect>::const_iterator begin,
		std::vector<cv::Rect>::const_iterator end, std::vector<cv::Rect>& result) {
	std::vector<cv::Rect>::const_iterator it = begin;

	cv::Rect start = *it;
	++it;
	result.push_back(start);

	for (; it != end; ++it) {
		if (abs(start.y - it->y) < config.getDigitYAlignment() && abs(start.height - it->height) < 5) {
			result.push_back(*it);
		}
	}
}

void ImageProcessor::FilterContours(std::vector<std::vector<cv::Point> >& contours,
        std::vector<cv::Rect>& boundingBoxes, std::vector<std::vector<cv::Point> >& filteredContours) {
	// filter contours by bounding rect size
	for (size_t i=0; i<contours.size(); i++) {
		cv::Rect bounds = cv::boundingRect(contours[i]);
		if (bounds.height > config.getDigitMinHeight() && bounds.height < config.getDigitMaxHeight()) {
			boundingBoxes.push_back(bounds);
			filteredContours.push_back(contours[i]);
		}
	}
}

void ImageProcessor::FindCounterDigits(ROIBox* _roi)
{
	cv::Mat edges = BinaryFiltering();
	if (debugEdges) cv::imshow("edges", edges);
	if (debugPower) {
		cv::Mat powerScreen = edges(_roi->getROIBox()[_roi->getROIBox().size()-1]);

		powerOn = false;
		for (int i=0; i<powerScreen.cols; i++) {
			for (int j=0; j<powerScreen.rows; j++) {
				uchar b = powerScreen.at<cv::Vec3b>(i,j)[0];
				uchar g = powerScreen.at<cv::Vec3b>(i,j)[1];
				uchar r = powerScreen.at<cv::Vec3b>(i,j)[2];
				if ( b > 100 && g > 100 && r > 100 ) { powerOn = true; break;}
			}
			if (powerOn) break;
		}
	}

	cv::Mat img_ret = edges.clone();

	// find contours in whole image
	std::vector<std::vector<cv::Point>> contours, filteredContours;
	std::vector<cv::Rect>boundingBoxes;
	cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	//filter contours by bounding rect size
	FilterContours(contours, boundingBoxes, filteredContours);

	// sort bounding boxes from left to right
	std::sort(boundingBoxes.begin(), boundingBoxes.end(), sortRectByX());

	if (debugEdges) {
		//drawContours
		cv::Mat cont = cv::Mat::zeros(edges.rows, edges.cols, CV_8UC1);
		cv::drawContours(cont, contours, -1, cv::Scalar(255));
		cv::imshow("contours", cont);
	}

	// cut out found rectangles from edged image
	for (int i = 0; i < boundingBoxes.size(); ++i) {
		cv::Rect roi = boundingBoxes[i];
		digits.push_back(img_ret(roi));

		if (debugDigits) {
			cv::rectangle(img, roi, cv::Scalar(0, 255, 0), 1);
		}
	}

	if (debugOCR) {
		cv::Mat voltScreen  = edges(_roi->getROIBox()[0]);
		cv::Mat currScreen  = edges(_roi->getROIBox()[1]);
		cv::Mat dapRScreen  = edges(_roi->getROIBox()[2]);

		std::vector<std::vector<cv::Point>> kVcontours, kVfilteredContours;
		std::vector<cv::Rect> kVboundingBoxes;
		cv::findContours(voltScreen, kVcontours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		FilterContours(kVcontours, kVboundingBoxes, kVfilteredContours);
		std::sort(kVboundingBoxes.begin(), kVboundingBoxes.end(), sortRectByX());

		for (int i=0; i < kVboundingBoxes.size(); ++i) {
			cv::Rect kVroi = kVboundingBoxes[i];
			digits_kV.push_back(voltScreen(kVroi));
		}

		std::vector<std::vector<cv::Point>> mAcontours, mAfilteredContours;
		std::vector<cv::Rect> mAboundingBoxes;
		cv::findContours(currScreen, mAcontours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		FilterContours(mAcontours, mAboundingBoxes, mAfilteredContours);
		std::sort(mAboundingBoxes.begin(), mAboundingBoxes.end(), sortRectByX());

		for (int i=0; i < mAboundingBoxes.size(); ++i) {
			cv::Rect mAroi = mAboundingBoxes[i];
			digits_mA.push_back(currScreen(mAroi));
		}

		std::vector<std::vector<cv::Point>> dapcontours, dapfilteredContours;
		std::vector<cv::Rect> dapboundingBoxes;
		cv::findContours(dapRScreen, dapcontours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		FilterContours(dapcontours, dapboundingBoxes, dapfilteredContours);
		std::sort(dapboundingBoxes.begin(), dapboundingBoxes.end(), sortRectByX());

		for (int i=0; i < dapboundingBoxes.size(); ++i) {
			cv::Rect daproi = dapboundingBoxes[i];
			digits_DAP.push_back(dapRScreen(daproi));
		}
	}



}
