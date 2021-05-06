#ifndef INCLUDE_OCR_FUNC_H_
#define INCLUDE_OCR_FUNC_H_

#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include "OCR_ImageInput.hh"
#include "OCR_ImageProcessor.hh"
#include "OCR_KNN.hh"

int DELAY = 1;

cv::Rect cropRect(0,0,0,0);
cv::Point P1(0,0), P2(0,0);
std::vector<cv::Rect> blackBox;
std::vector<cv::Rect> blackPoint;
bool clicked=false;
bool calc=false;

void onMouseCropImage(int event, int x, int y, int f, void *param);
ROIBox* setROIBOX(ImageInput* pImageInput);

void testOcr(ImageInput* pImageInput, int cam) {
	auto roi = setROIBOX(pImageInput);
	bool debugOCR(false);

	Config config;
    config.loadConfig();
    ImageProcessor proc(config);
    proc.DebugDigits();
    proc.DebugWindow();
    proc.DebugOCR();
    proc.DebugPower();

    KNearestOcr ocr(config);
    if (! ocr.loadTrainingData()) {
        std::cout << "Failed to load OCR training data\n";
        return;
    }
    std::cout << "OCR training data loaded.\n";
    std::cout << "<q> to quit.\n";

    cv::Mat imgNone = cv::Mat::zeros(pImageInput->getImage().rows, pImageInput->getImage().cols, CV_8UC3);

    int frameNo(0); int key = 0; bool processImage(true);
    while (pImageInput->nextImage()) {
    	std::cout << "Frame " << frameNo++ << std::endl;
		cv::Mat imgCopy = imgNone;

		for (int k=0; k<blackBox.size(); k++) {
			for (int i=blackBox[k].x; i< blackBox[k].x + blackBox[k].width; i++) {
				for (int j=blackBox[k].y; j<blackBox[k].y + blackBox[k].height; j++) {
					imgCopy.at<cv::Vec3b>(j,i) = pImageInput->getImage().at<cv::Vec3b>(j,i);
				}
			}
		}
    	proc.SetInput(imgCopy);
        proc.Process(roi);
        bool powerOn = proc.GetPowerOn();
        if (!powerOn) continue;

        // KNN results
        std::string voltage = ocr.recognize(proc.GetOutputkV());
        std::string current = ocr.recognize(proc.GetOutputmA());
		std::string dapRate = ocr.recognize(proc.GetOutputDAP());

        if (voltage.find('.') != std::string::npos || current.find('.') != std::string::npos
		 	||voltage.empty() || current.empty()) {

			std::cout << "Tube voltage (kV) : " << voltage << std::endl;
			std::cout << "Tube current (mA) : " << current << std::endl;
			std::cout << "DAP rate (Gycm2/s): " << dapRate << std::endl;
			std::cout << "!!WARNING!! point is recognized or character is not recognized" << std::endl;
        }
        else {
        	float currentF = (float)stoi(current) * 0.1;
			std::cout << "Tube voltage (kV) : " << stoi(voltage) << std::endl;
			std::cout << "Tube current (mA) : " << currentF << std::endl;
			std::cout << "DAP rate (Gycm2/s): " << dapRate << std::endl << std::endl;
        }

        int key = cv::waitKey(DELAY) & 255;

        if (key == 'q') {
            std::cout << "Quit\n";
            break;
        }
    }
}

void adjustCamera(ImageInput* pImageInput) {
	bool processImage(true);
	int key(0);

	auto roi = setROIBOX(pImageInput);

	Config config;
	config.loadConfig();
	ImageProcessor proc(config);
	proc.DebugDigits();
	proc.DebugEdges();
	proc.DebugWindow();
	std::cout <<"## ADJUST CAMERA ##\n";
	std::cout << "<r>, <p> to select raw or processed image, <s> to save config and quit, <q> to quit without saving.\n";

	cv::Mat imgNone = cv::Mat::zeros(pImageInput->getImage().rows, pImageInput->getImage().cols, CV_8UC3);
	while (pImageInput->nextImage()) {
		cv::Mat imgCopy = imgNone;

		for (int k=0; k<blackBox.size(); k++) {
			for (int i=blackBox[k].x; i< blackBox[k].x + blackBox[k].width; i++) {
				for (int j=blackBox[k].y; j<blackBox[k].y + blackBox[k].height; j++) {
					imgCopy.at<cv::Vec3b>(j,i) = pImageInput->getImage().at<cv::Vec3b>(j,i);
				}
			}
		}

		if (processImage) {
			proc.SetInput(imgCopy);
			proc.Process(roi);
		}
		else {
			proc.SetInput(pImageInput->getImage());
		}

		key = cv::waitKey(1) & 255;

		if (key == 'q' || key == 's') {
			std::cout << "Quit\n";
			break;
		} else if (key == 'r') {
			processImage = false;
		} else if (key == 'p') {
			processImage = true;
		}
	}

	if (key != 'q') {
		std::cout << "Saving config\n";
		config.saveConfig();
	}

}

void onMouseCropImage(int event, int x, int y, int f, void *param){
	switch (event) {
    case cv::EVENT_LBUTTONDOWN:
        clicked = true;
        P1.x = x;
        P1.y = y;
        P2.x = x;
        P2.y = y;
        break;
    case cv::EVENT_LBUTTONUP:
        P2.x=x;
        P2.y=y;
        clicked = false;
        break;
    case cv::EVENT_MOUSEMOVE:
        if(clicked){
        P2.x=x;
        P2.y=y;
        }
        break;
    default:
        break;
    }

    if(clicked){
        if(P1.x>P2.x){
            cropRect.x=P2.x;
            cropRect.width=P1.x-P2.x;
        }
        else{
            cropRect.x=P1.x;
            cropRect.width=P2.x-P1.x;
        }

        if(P1.y>P2.y){
            cropRect.y=P2.y;
            cropRect.height=P1.y=P2.y;
        }
        else{
            cropRect.y=P1.y;
            cropRect.height=P2.y-P1.y;
        }
    }
}

ROIBox* setROIBOX(ImageInput* pImageInput) {
	ROIBox* roi = new ROIBox();
	bool pushcrop = false;


	// Set ROI
	std::cout << ">> Select ROI box to OCR, Last ROI box will check the On/Off" << std::endl;
	while (pImageInput->nextImage()) {
		cv::Mat img, imgCopy, imgCrop;
		img = pImageInput->getImage();
		img.copyTo(imgCopy);

		cv::Rect bounds(0,0,img.cols,img.rows);

		cv::setMouseCallback("Select ROI", onMouseCropImage, &imgCopy);

		if (clicked) pushcrop = true;

		if (cropRect.width > 0 && clicked == false) {
			imgCrop = img(cropRect & bounds);
			if (pushcrop) {
				std::cout << "push" << std::endl;
				blackBox.push_back(cropRect);
				pushcrop = false;
			}
		}
		else img.copyTo(imgCrop);

		cv::rectangle(imgCopy, P1, P2, CV_RGB(255,255,0), 2);

		cv::putText(imgCopy, "Resolution: "+std::to_string(img.cols)+"x"+std::to_string(img.rows),
				cv::Point(10,20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(60000), 1);

		cv::imshow("crop", imgCrop);
		cv::imshow("Select ROI", imgCopy);

		int key = cv::waitKey(1);
		if (key == 'q') break;
	}
	cv::destroyAllWindows();
	roi->setROIBox(blackBox);

	return roi;
}

ROIBox* setROIBOX2(int box[4]) {
	ROIBox* roi = new ROIBox();
	bool pushcrop = false;
	ScreenShot screen(box[0],box[1],box[2],box[3]);

	// Set ROI
	std::cout << "    >> Select 4 ROI box to OCR, Last ROI box will check the On/Off" << std::endl;
	std::cout << "       1. Tube Voltage, 2. Tube Current, 3. DAP, 4. Power"          << std::endl;
	while (1) {
		cv::Mat img, imgCopy, imgCrop;
		screen(img);
		img.copyTo(imgCopy);

//		cv::Rect bounds(0,0,img.cols,img.rows);

		cv::setMouseCallback("Select ROI", onMouseCropImage, &imgCopy);

		if (clicked) pushcrop = true;

		if (cropRect.width > 0 && clicked == false) {
			imgCrop = img(cropRect);
			if (pushcrop) {
				std::cout << "      push ROI box" << std::endl;
				blackBox.push_back(cropRect);
				pushcrop = false;
			}
		}
		else img.copyTo(imgCrop);

		cv::rectangle(imgCopy, P1, P2, CV_RGB(255,255,0), 2);

		cv::putText(imgCopy, "Resolution: "+std::to_string(img.cols)+"x"+std::to_string(img.rows),
				cv::Point(10,20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(60000), 1);

		cv::imshow("crop", imgCrop);
		cv::imshow("Select ROI", imgCopy);

		int key = cv::waitKey(1);
		if (key == 'q') break;
	}
	cv::destroyAllWindows();
	roi->setROIBox(blackBox);

	return roi;
}

map<int, tuple<double,double,double,double>> Read_Recording_OCR(string fileName)
{
	cout << "    Read recording ocr file..." << endl;
	string ocrFile = fileName.substr(0,fileName.size()-8) + ".ocr";
	ifstream ifs(ocrFile);
	if(!ifs.is_open()) { cerr << ocrFile  << " was not opened" << endl; exit(1); }
	map<int, tuple<double,double,double,double>> onFrameMap;
	string dump;
	getline(ifs,dump);
	while(getline(ifs,dump)) {
		int frame; double frameT, tVolt, tCurr, dapRate;
		stringstream ss(dump);
		ss >> frame >> frameT >> tVolt >> tCurr >> dapRate;
		onFrameMap[frame] = make_tuple(frameT,tVolt,tCurr,dapRate);
	}
	return onFrameMap;
}

#endif /* INCLUDE_OCR_FUNC_H_ */
