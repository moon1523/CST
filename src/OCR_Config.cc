#include "OCR_Config.hh"

Config::Config() :
        _rotationDegrees(0), _ocrMaxDist(5e5), _digitMinHeight(20), _digitMaxHeight(
                90), _digitYAlignment(10), _cannyThreshold1(100), _cannyThreshold2(
                200), _trainingDataFilename("trainctr.yml"), _binaryThreshold(100) {
}

void Config::saveConfig() {
    cv::FileStorage fs("./data/config.yml", cv::FileStorage::WRITE);
    fs << "rotationDegrees" << _rotationDegrees;
    fs << "cannyThreshold1" << _cannyThreshold1;
    fs << "cannyThreshold2" << _cannyThreshold2;
    fs << "digitMinHeight" << _digitMinHeight;
    fs << "digitMaxHeight" << _digitMaxHeight;
    fs << "digitYAlignment" << _digitYAlignment;
    fs << "ocrMaxDist" << _ocrMaxDist;
    fs << "trainingDataFilename" << _trainingDataFilename;
    fs << "binaryThreshold" << _binaryThreshold;
    fs.release();
}

void Config::loadConfig() {
    cv::FileStorage fs("./data/config.yml", cv::FileStorage::READ);
    if (fs.isOpened()) {
        fs["rotationDegrees"] >> _rotationDegrees;
        fs["cannyThreshold1"] >> _cannyThreshold1;
        fs["cannyThreshold2"] >> _cannyThreshold2;
        fs["digitMinHeight"] >> _digitMinHeight;
        fs["digitMaxHeight"] >> _digitMaxHeight;
        fs["digitYAlignment"] >> _digitYAlignment;
        fs["ocrMaxDist"] >> _ocrMaxDist;
        fs["trainingDataFilename"] >> _trainingDataFilename;
        fs["binaryThreshold"] >> _binaryThreshold;
        fs.release();
    } else {
        // no config file - create an initial one with default values
        saveConfig();
    }
}
