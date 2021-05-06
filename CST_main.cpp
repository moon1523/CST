#include "AzureKinect.h"
#include "MultiDeviceCapturer.h"
#include "LabelData.h"
#include "TrackingFunc.h"
#include "OCR_Func.h"
#include "VTKFunc.h"

void PrintUsage() {
	cout << "=================================================" << endl;
	cout << "C-arm Source Tracker (CST) - 'LINEMOD + KNN-OCR' " << endl;
	cout << "# Recording Tracking Version                     " << endl;
	cout << "Author: Sungho Moon, Haegin Han                  " << endl;
	cout << "=================================================" << endl;
}

int main(int argc, char** argv)
{
	PrintUsage();
	// Arguments
	bool tracking(true); bool recording(false);
	bool rec(false);

	bool fromRecording(false);
	bool endXML(false);
	bool cutXML(false);
	bool print(false);

	int width(1280), height(720);
	string dataPath("./data/");
	string outputPath("./output/");
	string recordPath("./record/");
	string outputFileName("./output/result");
	string recordFileName("./record/record");
	string printFileName;
	LabelData label;
	Timer init_timer, match_timer;
	string printPath;


	for (int i=1; i<argc; i++) {
		if (string(argv[i]) == "-o") {
			outputFileName = outputPath + argv[i+1];
			printFileName = outputFileName;
			i++;
		}
		else if (string(argv[i]) == "-r") {
			tracking = false; rec = true;
			recordFileName = recordPath + argv[i+1];
			printFileName = recordFileName;
			i++;
		}
		else if (string(argv[i]) == "-R") {
			fromRecording = true;
			recordFileName = argv[i+1];
			printFileName = recordFileName;
			if (argv[i+2]) printPath = string(argv[i+2]);
			i++;
			label = LabelData(dataPath + "labels.txt");
		}
	}
	if (argc == 1) {
		printFileName = outputFileName;
		cout << "Default tracking mode" << endl;
	}
	init_timer.start();
	//read ply file
	viz::Viz3d myWindow("PLY viewer");
	WPoly poly;
	if(tracking){
		poly.Initialize(dataPath+"carm.ply",label);
		myWindow.setWindowSize(Size(1280, 720));
		myWindow.showWidget("Coordinate", viz::WCoordinateSystem(500.));
		myWindow.showWidget("model PLY", poly);
		Vec3f cam_pos(0,0,-3000), cam_focal_point(0,0,1), cam_y_dir(0,-1.,0);
		Affine3f cam_pose = viz::makeCameraPose(cam_pos, cam_focal_point, cam_y_dir);
		myWindow.setViewerPose(cam_pose);
		myWindow.spinOnce(1,true);
	}

	// Various settings and flags
	int num_classes = 0;
	int matching_threshold = 70;
	bool show_match_result = false;
	bool show_timings = false;
	bool show_aiming = false;

	pair<vector<cv::Mat>,vector<cv::Mat>> recVec;
	map<int, tuple<double,double,double,double>> onFrameMap;
	map<int, tuple<double,double,double,double>>::iterator itr;

	int initFrame = stoi(recordFileName.substr(recordFileName.size()-7,recordFileName.size()-4));
	int lastFrame = stoi(recordFileName.substr(recordFileName.size()-3,recordFileName.size()));

	int initOnFrame(0), lastOnFrame(0), onFrameNum(0);
	vector<cv::Mat> colorVec, depthVec;
	if(fromRecording) {
		onFrameMap = Read_Recording_OCR(printFileName);
		initOnFrame = onFrameMap.begin()->first;
		lastOnFrame = onFrameMap.rbegin()->first;
		if(cutXML) initOnFrame = initFrame;

		recVec = Read_OpenCV_XML(recordFileName, initFrame);
		colorVec = recVec.first;
		depthVec = recVec.second;
	}

	int recSize = recVec.first.size();

	vector<pair<float,float>> xy_table = Read_XY_Table(dataPath, width, height);

	// Initialize LINEMOD data structures
	cv::Ptr<cv::linemod::Detector> detector;
	if(tracking) {
		cout << "    Initialize LINEMOD data structures: ";
		detector = readLinemod(dataPath + "templates.yml");
		num_classes = detector->numClasses();
		printf("Loaded %d classes and %d templates\n",
				num_classes, detector->numTemplates());
	} else {
		detector = cv::linemod::getDefaultLINEMOD();
	}
	int num_modalities = (int)detector->getModalities().size();

	init_timer.stop();
	cout << "  >> Initialization Time(s): " << init_timer.time() << endl << endl;


	// Main Loop
	cv::Mat color, depth;
	cv::FileStorage fs_d, fs_c;
	if(print)
    printPLY = true;

    double source[3] = {0,0,-810};
    double origin[3] = {0,0,0};
    double isocenter[3] = {-4.09685,-19.691,2363};
    vtkSmartPointer<vtkTransform> transform  = vtkSmartPointer<vtkTransform>::New();

    cout << "Main Loop Start" << endl; help();
    cout << "initFrame: " << initFrame << endl;
	cout << "lastFrame: " << lastFrame << endl;
    cout << "initOnFrame: " << initOnFrame << endl;
    cout << "lastOnFrame: " << lastOnFrame << endl;
	for (int i=initFrame; i<=lastFrame; i++) {
		bool power(false);
		Timer onframe_time;
		onframe_time.start();

		// color/depth images from recording file
		if(fromRecording) {
			depth = depthVec[i-initFrame];
			color = colorVec[i-initFrame];
		}

		vector<cv::Mat> sources;
		cv::Mat depthOrigin = depth.clone();
		depth = (depth-label.GetMinDist())*label.GetDepthFactor();
		sources.push_back(color);
		sources.push_back(depth);

		cv::Mat display = color.clone();

		vector<cv::Mat> quantized_images;
		if (tracking) {
			// Perform matching
			vector<cv::linemod::Match> matches;
			vector<cv::String> class_ids;


			match_timer.start();
			detector->match(sources, (float)matching_threshold, matches, class_ids, quantized_images);
			match_timer.stop();

			int classes_visited = 0;
			set<string> visited;

			int maxID; double maxSim(-1); int match_count(0);
			for (int i=0; (i < (int)matches.size()) && (classes_visited < num_classes); ++i) {
				cv::linemod::Match m = matches[i];
				if (visited.insert(m.class_id).second) {
					++classes_visited;

					if (show_match_result) {
						printf("Similarity: %5.1f%%; x: %3d; y: %3d; class: %s; template: %3d\n",
							   m.similarity, m.x, m.y, m.class_id.c_str(), m.template_id);
							   match_count++;
					}
					if (m.similarity > maxSim) {
						maxSim = m.similarity;
						maxID = i;
					}
				}
			}

			if (maxSim > 0) {
				cout << "Frmae #" << i << endl;
				cv::linemod::Match m = matches[maxID];
				// Draw matching template
				const std::vector<cv::linemod::Template>& templates = detector->getTemplates(m.class_id, m.template_id);
				drawResponse(templates, num_modalities, display, cv::Point(m.x, m.y), detector->getT(0), depthOrigin);
				onframe_time.stop();
				string frameT  = to_string(get<0>(onFrameMap[i]));
				string tVolt   = to_string(get<1>(onFrameMap[i]));
				string tCurr   = to_string(get<2>(onFrameMap[i]));
				string dapRate = to_string(get<3>(onFrameMap[i]));
				if ( initOnFrame <= i && i<=lastOnFrame) power = true;
				Print_CST_Result(printFileName,label, m, xy_table, depth, i, tVolt, tCurr, dapRate, frameT, power, isocenter);
				printf("Similarity: %5.1f%%; x: %3d; y: %3d; class: %s; template: %3d\n",
														   maxSim, m.x, m.y, m.class_id.c_str(), m.template_id);

				int iC = m.x + label.GetPosition(stoi(m.class_id), m.template_id).first;
				int jC = m.y + label.GetPosition(stoi(m.class_id), m.template_id).second;
				double kC = label.GetDistance(stoi(m.class_id), m.template_id);

				double posX = xy_table[depth.cols*jC + iC].first  * kC;
				double posY = xy_table[depth.cols*jC + iC].second * kC;
				double posZ = kC;

				double source_rot[3], origin_rot[3];

				transform->Identity();
				transform->SetMatrix(label.GetAffineTransformMatrix(stoi(m.class_id), m.template_id, isocenter));
				transform->TransformPoint(origin, origin_rot);
				transform->TransformPoint(source, source_rot);

//				origin_rot[0] += posX; origin_rot[1] += posY;
//				source_rot[0] += posX; source_rot[1] += posY;


				poly.Transform(printPath, atoi(m.class_id.c_str()), m.template_id, origin_rot, i);
				myWindow.spinOnce(1, true);
			}

			if (show_match_result && matches.empty())
				printf("No matches found...\n");

			cout << "Match count: " << match_count << endl;
			printf("Matching Time: %.2fs\n", match_timer.time());
			printf("------------------------------------------------------------\n");

		} // tracking
		cv::imshow("color", display);
		cv::imshow("normals", quantized_images[1]);

		if(print) {
			cv::imwrite(printPath + "color/color" + to_string(i) +".png", display);
			cv::imwrite(printPath + "normal/normal" + to_string(i) +".png", quantized_images[1]);
		}


		char key = (char)cv::waitKey(1);
		if (key == 'q') { cout << endl; break; }

		switch (key)
		{
			case 'h':
				help();
				break;
			case 'm':
				// toggle printing match result
				show_match_result = !show_match_result;
				printf("Show match result %s\n", show_match_result ? "ON" : "OFF");
				break;
			case 't':
				// toggle printing timings
				show_timings = !show_timings;
				printf("Show timings %s\n", show_timings ? "ON" : "OFF");
				break;
			case '[':
				// decrement threshold
				matching_threshold = std::max(matching_threshold - 1, -100);
				printf("New threshold: %d\n", matching_threshold);
				break;
			case ']':
				// increment threshold
				matching_threshold = std::min(matching_threshold + 1, +100);
				printf("New threshold: %d\n", matching_threshold);
				break;
			case 'r':
				// record switch
				if(recording) cout << "Recording switch OFF, press 'e' to save recording file" << endl;
				else          cout << "Recording switch ON, press 'r' again to stop recording" << endl;
				recording = !recording;
				break;
			case 'e':
				// record end
				cout << "Recording end" << endl;
				endXML=true;
				break;
			case 'a':
				// +
				if(show_aiming) cout << endl << "Aiming switch OFF" << endl;
				else cout << "Aiming swithch ON" << endl;
				show_aiming = ! show_aiming;
				break;
			default:
				;
		} // switch
	} // Main Loop

	return 0;
}
