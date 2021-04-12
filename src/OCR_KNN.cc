#include "OCR_KNN.hh"

KNearestOcr::KNearestOcr(const Config& _config)
: pModel(), config(_config){

}

KNearestOcr::~KNearestOcr() {
}

// Learn a single digit.
int KNearestOcr::learn(const cv::Mat& img) {
	cv::imshow("Learn", img);
	int key = cv::waitKey(0) & 255;
	if (key >= 176 && key <= 185) {
		key -= 128; // numeric keypad
	}
	if ((key >= '0' && key <= '9') || key == '.') {
		responses.push_back(cv::Mat(1, 1, CV_32F, (float) key - '0'));
		samples.push_back(prepareSample(img));
		std::cout << char(key) << std::flush;
	}
	return key;
}

// Learn a vector of digits.
int KNearestOcr::learn(const std::vector<cv::Mat>& images) {
	int key = 0;
	for (std::vector<cv::Mat>::const_iterator it = images.begin();
			it < images.end() && key != 's' && key != 'q'; ++it) {
		key = learn(*it);
	}
	return key;
}

bool KNearestOcr::hasTrainingData() {
	return !samples.empty() && !responses.empty();
}

// Save training data to file.
void KNearestOcr::saveTrainingData() {
	cv::FileStorage fs(config.getTrainingDataFilename(), cv::FileStorage::WRITE);
	fs << "samples" << samples;
	fs << "responses" << responses;
	fs.release();
}

// Load training data from file and init model.
bool KNearestOcr::loadTrainingData() {
	cv::FileStorage fs(config.getTrainingDataFilename(), cv::FileStorage::READ);
	if (fs.isOpened()) {
		fs["samples"] >> samples;
		fs["responses"] >> responses;
		fs.release();

		initModel();
	} else return false;
	return true;
}

// Recognize a single digit.
char KNearestOcr::recognize(const cv::Mat& img) {
	char cres = '?';
	int k_idx(3);

	if (pModel.empty()) {
		throw std::runtime_error("Model is not initialized");
	}

	cv::Mat results, neighborResponses, dists;
	float result = pModel->findNearest(prepareSample(img), k_idx, results, neighborResponses, dists);

	// Find majority character of neigborResponses set. (k_idx should be odd number to determine the character)
	std::vector<int> neighborResponsesCount; // 0,1,2,3,4,5,6,7,8,9,'.'
	neighborResponsesCount.clear();
	for (int i=0; i<11; i++) neighborResponsesCount.push_back(0);
	for (int i=0; i < neighborResponses.cols; i++) {
		if ((int)neighborResponses.at<float>(0,i) == -2) { neighborResponsesCount[10]++; continue; }
		neighborResponsesCount[(int)neighborResponses.at<float>(0,i)]++;
	}

	int maxResponse = *max_element(neighborResponsesCount.begin(), neighborResponsesCount.end());
	for (int i=0;i<neighborResponsesCount.size();i++) {
		if (neighborResponsesCount[i] == maxResponse) {
			if (i == 10) { results = -2; break; }
			else 		 { results =  i; break; }
		}
	}	// set value

	cres = '0' + (int) result;

//	std::cout << "results: " << results << std::endl;
//	std::cout << "neighborResponses: " << neighborResponses << std::endl;
//	std::cout << "dists: " << dists << std::endl;

	return cres;
}

// Recognize a vector of digits.
std::string KNearestOcr::recognize(const std::vector<cv::Mat>& images) {
	std::string result;
	for (std::vector<cv::Mat>::const_iterator it = images.begin();
			it != images.end(); ++it) {
		result += recognize(*it);
	}
	return result;
}

// Prepare an image of a digit to work as a sample for the model.
cv::Mat KNearestOcr::prepareSample(const cv::Mat& img) {
	cv::Mat roi, sample;
	cv::resize(img, roi, cv::Size(10, 10));
	roi.reshape(1,1).convertTo(sample, CV_32F);

	return sample;
}

// Initialize the model.
void KNearestOcr::initModel() {
	pModel = cv::ml::KNearest::create();
	// load persistent model
	cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(samples, cv::ml::ROW_SAMPLE, responses);
	pModel->train(trainData);
}
