#ifndef INCLUDE_TRACKINGFUNC_H_
#define INCLUDE_TRACKINGFUNC_H_

#include <opencv2/rgbd.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <vtkSmartPointer.h>
#include <vtkTransform.h>

#include <set>

using namespace std;

void help()
{
    printf("Keys:\n"
           "\t h   -- This help page\n"
           "\t m   -- Toggle printing match result\n"
           "\t t   -- Toggle printing timings\n"
           "\t [ ] -- Adjust matching threshold: '[' down,  ']' up\n"
           "\t r   -- record till pressed again (for device-connected mode)\n"
           "\t q   -- Quit\n\n");
}

// Adapted from cv_timer in cv_utilities
class Timer
{
public:
    Timer() : start_(0), time_(0) {}

    void start()
    {
        start_ = cv::getTickCount();
    }

    void stop()
    {
        CV_Assert(start_ != 0);
        int64 end = cv::getTickCount();
        time_ += end - start_;
        start_ = 0;
    }

    double time()
    {
        double ret = time_ / cv::getTickFrequency();
        time_ = 0;
        return ret;
    }

private:
    int64 start_, time_;
};

cv::Ptr<cv::linemod::Detector> readLinemod(const std::string& filename)
{
    cv::Ptr<cv::linemod::Detector> detector = cv::makePtr<cv::linemod::Detector>();
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    detector->read(fs.root());

    cv::FileNode fn = fs["classes"];
    for (cv::FileNodeIterator i = fn.begin(), iend = fn.end(); i != iend; ++i)
        detector->readClass(*i);

    return detector;
}

pair<vector<cv::Mat>, vector<cv::Mat>> Read_OpenCV_XML(string recordFileName, int recordID)
{
	cout << "    Read recording xml file..." << endl;
	cv::FileStorage fs_d, fs_c;
	vector<cv::Mat> colorVec, depthVec;

	cout << "      color: " << recordFileName + "_c.xml" << endl;
	cout << "      depth: " << recordFileName + "_d.xml" << endl;
	fs_c.open(recordFileName + "_c.xml", cv::FileStorage::READ);
	fs_d.open(recordFileName + "_d.xml", cv::FileStorage::READ);
	string matName = "frame"+to_string(recordID);
	while(1){
		cv::Mat color1, depth1;
		fs_c[matName]>>color1;
		if(color1.empty())break;
		cout<<"\rReading frame "<<recordID<<flush;
		colorVec.push_back(color1);
		fs_d[matName]>>depth1;
		depthVec.push_back(depth1);
		matName = "frame"+to_string(++recordID);
	}
	fs_c.release();
	fs_d.release();
	cout<<"\r      Imported "<<--recordID<<" frames"<<endl;
	recordID = 0;

	return make_pair(colorVec, depthVec);
}

vector<pair<float,float>> Read_XY_Table(string dataPath, int width, int height)
{
	cout << "    Initialize XY Table" << flush;
	ifstream ifs(dataPath + "xytable.txt");
	string xxx, yyy;
	vector<pair<float, float>> xy_table;
	vector<tuple<float, float, float>> pcd_data;;
	for (int y=0; y<height; y++) {
		for (int x=0; x<width; x++) {
			ifs >> xxx >> yyy;
			if (xxx == "nan" && yyy == "nan") { xy_table.push_back({0,0}); continue; }
			xy_table.push_back({stof(xxx),stof(yyy)});
		}
	} cout << "...done" << endl;

	return xy_table;
}

void drawResponse(const std::vector<cv::linemod::Template>& templates,
                  int num_modalities, cv::Mat& dst, cv::Point offset, int T,
                  cv::Mat depth)
{
    static const cv::Scalar COLORS[5] = { CV_RGB(0, 0, 255),
								   CV_RGB(0, 255, 0),
								   CV_RGB(255, 255, 0),
								   CV_RGB(255, 140, 0),
								   CV_RGB(255, 0, 0) };
    for (int m = 0; m < num_modalities; ++m)
    {
        // NOTE: Original demo recalculated max response for each feature in the TxT
        // box around it and chose the display color based on that response. Here
        // the display color just depends on the modality.
        cv::Scalar color = COLORS[m];

        for (int i = 0; i < (int)templates[m].features.size(); ++i)
        {
            cv::linemod::Feature f = templates[m].features[i];
            cv::Point pt(f.x + offset.x, f.y + offset.y);
            cv::circle(dst, pt, T / 2, color);
        }
    }
}


void Print_CST_Result(string fileName,
					  LabelData label, cv::linemod::Match m,
					  vector<pair<float,float>> xy_table, cv::Mat depth, int frameNo,
					  string voltage, string current, string dapRate, Timer frame_time, double isocenter[3])
{
	double origin[3] = {0,0,0}; double source[3] = {0,0,-810};
	double origin_rot[3], source_rot[3];
	vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
	ofstream ofs(fileName + ".out",ios::app);

	transform->Identity();
	transform->SetMatrix(label.GetAffineTransformMatrix(stoi(m.class_id), m.template_id, isocenter));
	transform->TransformPoint(origin, origin_rot);
	transform->TransformPoint(source, source_rot);

	int iC = m.x + label.GetPosition(stoi(m.class_id), m.template_id).first;
	int jC = m.y + label.GetPosition(stoi(m.class_id), m.template_id).second;
	double kC = label.GetDistance(stoi(m.class_id), m.template_id);

	double posX = xy_table[depth.cols*jC + iC].first  * kC;
	double posY = xy_table[depth.cols*jC + iC].second * kC;
	double posZ = kC;

	origin_rot[0] += posX; origin_rot[1] += posY;
	source_rot[0] += posX; source_rot[1] += posY;

    // Print Results
    ofs << "Frame " << frameNo << endl;
    ofs << "Frame_Time: " << frame_time.time() << endl;
    ofs << "Index " << m.class_id << " " << m.template_id << " " << endl;
    ofs << "AffineT" << endl;
    ofs << label.GetAffineTransformMatrix(stoi(m.class_id), m.template_id)[0] << " "
        << label.GetAffineTransformMatrix(stoi(m.class_id), m.template_id)[1] << " "
        << label.GetAffineTransformMatrix(stoi(m.class_id), m.template_id)[2] << " "
        << posX << endl;
    ofs << label.GetAffineTransformMatrix(stoi(m.class_id), m.template_id)[4] << " "
        << label.GetAffineTransformMatrix(stoi(m.class_id), m.template_id)[5] << " "
        << label.GetAffineTransformMatrix(stoi(m.class_id), m.template_id)[6] << " "
        << posY << endl;
    ofs << label.GetAffineTransformMatrix(stoi(m.class_id), m.template_id)[8] << " "
        << label.GetAffineTransformMatrix(stoi(m.class_id), m.template_id)[9] << " "
        << label.GetAffineTransformMatrix(stoi(m.class_id), m.template_id)[10] << " "
        << label.GetAffineTransformMatrix(stoi(m.class_id), m.template_id)[11] << endl;
    ofs << label.GetAffineTransformMatrix(stoi(m.class_id), m.template_id)[12] << " "
        << label.GetAffineTransformMatrix(stoi(m.class_id), m.template_id)[13] << " "
        << label.GetAffineTransformMatrix(stoi(m.class_id), m.template_id)[14] << " "
        << label.GetAffineTransformMatrix(stoi(m.class_id), m.template_id)[15] << endl;

    ofs << "Isocenter " << origin_rot[0] << " " << origin_rot[1] << " " << origin_rot[2] << endl;
    ofs << "Source    " << source_rot[0] << " " << source_rot[1] << " " << source_rot[2] << endl;

    // OCR Results
    ofs << "T_Voltage " << voltage << endl;
	ofs << "T_Current " << current << endl;
	ofs << "DAP_rate  " << dapRate << endl << endl;
}

void Print_CST_Result2(string fileName, int frameNo, string voltage, string current, string dapRate, Timer frame_time)
{
	ofstream ofs(fileName + ".out",ios::app);
	ofs << "Frame " << frameNo << endl;
	ofs << "Frame_Time: " << frame_time.time() << endl;
    // OCR Results
    ofs << "T_Voltage " << voltage << endl;
    ofs << "T_Current " << current << endl;
    ofs << "DAP_rate  " << dapRate << endl << endl;
}



#endif /* INCLUDE_TRACKINGFUNC_H_ */
