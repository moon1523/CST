#ifndef INCLUDE_OCR_KNN_HH_
#define INCLUDE_OCR_KNN_HH_

#include "OCR_Config.hh"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <vector>
#include <list>
#include <string>

class KNearestOcr {
public:
	KNearestOcr(const Config& config);
	virtual ~KNearestOcr();

	int learn(const cv::Mat& img);
	int learn(const std::vector<cv::Mat>& images);
	bool hasTrainingData();
	void saveTrainingData();
	bool loadTrainingData();

	char recognize(const cv::Mat& img);
	std::string recognize(const std::vector<cv::Mat>& images);

private:
	cv::Mat prepareSample(const cv::Mat& img);
	void initModel();

	cv::Mat samples;
	cv::Mat responses;
	cv::Ptr<cv::ml::KNearest> pModel;
	Config config;
};

#endif /* INCLUDE_OCR_KNN_HH_ */
