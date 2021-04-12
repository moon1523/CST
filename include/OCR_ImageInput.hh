#ifndef INCLUDE_OCR_IMAGEINPUT_HH_
#define INCLUDE_OCR_IMAGEINPUT_HH_

#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <X11/Xlib.h>
#include <X11/Xutil.h>

class ImageInput {
public:
	ImageInput();
	ImageInput(int device);
	virtual ~ImageInput();

	virtual bool nextImage();

	virtual cv::Mat& getImage();
	virtual void setImage(cv::Mat& _img);

	cv::VideoCapture getVideoCapture() { return capture; }


private:
	cv::Mat img;
	cv::VideoCapture capture;
};

class ScreenShot {
public:
	ScreenShot(int _x, int _y, int _width, int _height):
		x(_x), y(_y), width(_width), height(_height) {
		display = XOpenDisplay(nullptr);
		root = DefaultRootWindow(display);
	}

	virtual ~ScreenShot()
	{
		if(img != nullptr) XDestroyImage(img);
		XCloseDisplay(display);
	}

	void operator() (cv::Mat& cvImg)
	{
		if(img != nullptr)
			XDestroyImage(img);
		img = XGetImage(display, root, x, y, width, height, AllPlanes, ZPixmap);
		cv::Mat bgraImg = cv::Mat(height, width, CV_8UC4, img->data);
		cv::cvtColor(bgraImg, cvImg, cv::COLOR_BGRA2BGR);
	}

    void SetDisplay(int _x, int _y, int _width, int _height) {
    	x = _x;
    	y = _y;
    	width = _width;
    	height = _height;
    }
    int GetX() { return x; }
    int GetY() { return y; }
    int GetWidth() { return width; }
    int GetHeight() { return height; }

private:
	Display* display;
	Window root;
	int x,y,width,height;
	XImage* img{nullptr};

};

#endif /* INCLUDE_OCR_IMAGEINPUT_HH_ */
