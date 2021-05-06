#include "AzureKinect.h"
#include "MultiDeviceCapturer.h"
#include "LabelData.h"
#include "TrackingFunc.h"
#include "OCR_Func.h"

void PrintUsage() {
	cout << "=================================================" << endl;
	cout << "C-arm Source Tracker (CST) - 'LINEMOD + KNN-OCR' " << endl;
	cout << "Author: Sungho Moon, Haegin Han                  " << endl;
	cout << "=================================================" << endl;
	cout << "  ./CST -o  [output] : Tracking                  " << endl;
	cout << "  ./CST -r  [record] : Recording without tracking" << endl;
	cout << "  ./CST -R  [record] : Tracking with record file " << endl;

}

int main(int argc, char** argv)
{
	PrintUsage();
	// Arguments
	bool tracking(true); bool recording(false);
	bool rec(false);

	bool fromRecording(false);
	bool endXML(false);

	int width(1280), height(720);
	string dataPath("./data/");
	string outputPath("./output/");
	string recordPath("./record/");
	string outputFileName("./output/result");
	string recordFileName("./record/record");
	string printFileName;
	LabelData label;
	Timer init_timer, match_timer;

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
			printFileName = outputFileName;
			recordFileName = recordPath + argv[i+1];
			i++;
		}
	}
	if (argc == 1) {
		printFileName = outputFileName;
		cout << "Default tracking mode" << endl;
	}

	// Various settings and flags
	int num_classes = 0;
	int matching_threshold = 70;
	bool show_match_result = true;
	bool show_timings = false;
	bool show_aiming = false;

	init_timer.start();
	// Initialize Azure Kinect
	cout << "    Initialize Azure Kinect" << flush;
	vector<uint32_t> device_indices{ 0 };
	int32_t color_exposure_usec = 8000;  // somewhat reasonable default exposure time
	int32_t powerline_freq = 2;          // default to a 60 Hz powerline
	MultiDeviceCapturer capturer;
	k4a_device_configuration_t main_config, secondary_config;
	k4a::transformation main_depth_to_main_color;

	if(!fromRecording){
		capturer = MultiDeviceCapturer(device_indices, color_exposure_usec, powerline_freq);
		// Create configurations for devices
		main_config = get_master_config();
		main_config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
		main_config.camera_fps = K4A_FRAMES_PER_SECOND_15;
		main_config.color_resolution = K4A_COLOR_RESOLUTION_720P;
		main_config.wired_sync_mode = K4A_WIRED_SYNC_MODE_STANDALONE;// no need to have a master cable if it's standalone
		secondary_config = get_subordinate_config(); // not used - currently standalone mode
		// Construct all the things that we'll need whether or not we are running with 1 or 2 cameras
		k4a::calibration main_calibration = capturer.get_master_device().get_calibration(main_config.depth_mode,main_config.color_resolution);
		// Set up a transformation. DO THIS OUTSIDE OF YOUR MAIN LOOP! Constructing transformations involves time-intensive
		// hardware setup and should not change once you have a rigid setup, so only call it once or it will run very
		// slowly.
		main_depth_to_main_color = k4a::transformation(main_calibration);
		capturer.start_devices(main_config, secondary_config);
	}
	cout << "...done" << endl;

	pair<vector<cv::Mat>,vector<cv::Mat>> recVec;
	vector<cv::Mat> colorVec, depthVec;
	if(fromRecording) {
		recVec = Read_OpenCV_XML(recordFileName, 95); // set the recording index
		colorVec = recVec.first;
		depthVec = recVec.second;
	}

	auto xy_table = Read_XY_Table(dataPath, width, height);

	// Initialize LINEMOD data structures
	cv::Ptr<cv::linemod::Detector> detector;
	if(tracking) {
		cout << "    Initialize LINEMOD data structures: ";
		detector = readLinemod(dataPath + "templates.yml");
		num_classes = detector->numClasses();
		printf("Loaded %d classes and %d templates\n",
				num_classes, detector->numTemplates());
//		label = LabelData(dataPath + "labels.txt");
	} else {
		detector = cv::linemod::getDefaultLINEMOD();
	}
	int num_modalities = (int)detector->getModalities().size();

	// Initialize OCR

	int box[4] = {0,100,1200,400};
	ScreenShot screen(box[0],box[1],box[2],box[3]);
	cv::Mat imgNone = cv::Mat::zeros(box[3],box[2],CV_8UC3);

	cout << "    Initialize Screen OCR" << endl;
	ROIBox* roi = setROIBOX2(box);

	Config config;
	config.loadConfig();
	ImageProcessor proc(config);
	proc.DebugWindow();
	proc.DebugPower();
	proc.DebugDigits();
	proc.DebugOCR();
	KNearestOcr ocr(config);
	if (!ocr.loadTrainingData()) { cout << "      Failed to load OCR training data" << endl; return 1; }
	cout << "      OCR training data loaded." << endl;


	init_timer.stop();
	cout << "  >> Initialization Time(s): " << init_timer.time() << endl << endl;


	// Main Loop
	int frameNo(0);
	cv::Mat color, depth;
	cv::FileStorage fs_d, fs_c;

	double isocenter[3] = {-4.09685, -19.691, 2363};

	cout << "Main Loop Start" << endl; help();
	while(1) {
		cv::Mat img; screen(img);
		cv::Mat imgCopy(imgNone);

		// Read ROI box screen
		for (int k=0; k<blackBox.size(); k++) {
			for (int i=blackBox[k].x; i< blackBox[k].x + blackBox[k].width; i++) {
				for (int j=blackBox[k].y; j<blackBox[k].y + blackBox[k].height; j++) {
					imgCopy.at<cv::Vec3b>(j,i) = img.at<cv::Vec3b>(j,i);
				}
			}
		}
		proc.SetInput(imgCopy);
		proc.Process(roi);
		bool powerOn = proc.GetPowerOn();

//		if(!fromRecording) {
			// color/depth images caputured by Azure Kinect DK
			// Be careful to use it
		    //
			vector<k4a::capture> captures;
			captures = capturer.get_synchronized_captures(secondary_config, true);
			k4a::image main_color_image = captures[0].get_color_image();
			k4a::image main_depth_image = captures[0].get_depth_image();

			k4a::image main_depth_in_main_color = create_depth_image_like(main_color_image);
			main_depth_to_main_color.depth_image_to_color_camera(main_depth_image, &main_depth_in_main_color);
			depth = depth_to_opencv(main_depth_in_main_color);
			color = color_to_opencv(main_color_image);
//		}
		// color/depth images from recording file
//		else {
//			if (frameNo == depthVec.size()) break;
//			depth = depthVec[frameNo];
//			color = colorVec[frameNo];
//		}

		vector<cv::Mat> sources;
		cv::Mat depthOrigin = depth.clone();
		depth = (depth-label.GetMinDist())*label.GetDepthFactor();
		sources.push_back(color);
		sources.push_back(depth);

		cv::Mat display = color.clone();

		if(recording) {
			if(!fs_d.isOpened()) {
				fs_c.open(recordFileName+"_c.xml", cv::FileStorage::WRITE);
				fs_d.open(recordFileName+"_d.xml", cv::FileStorage::WRITE);
			}
			fs_d << "frame" + to_string(frameNo) << depth;
			fs_c << "frame" + to_string(frameNo) << color;

			cout << "Recorded frame " << frameNo << endl;
		}
		if (endXML) {
			fs_d.release();
			fs_c.release();
		}

		if (show_aiming) {
			int aimSize(10);
			line(display, cv::Point(640,360-aimSize), cv::Point(640,360+aimSize), cv::Scalar(255,255,0), 2, 4, 0);
			line(display, cv::Point(640-aimSize,360), cv::Point(640+aimSize,360), cv::Scalar(255,255,0), 2, 4, 0);

			int aimPP(0);
			for (int y=0;y<aimSize*2;y++) {
				for (int x=0;x<aimSize*2;x++) {
					aimPP += depth.at<ushort>(360-aimSize+y,640-aimSize+x);
				}
			} aimPP = aimPP / (4*aimSize*aimSize);

			cout << "\rcenter depth[area|point]: " << aimPP << " | " << depth.at<ushort>(360,640) << "  " << flush;
		}

		bool fullprint(false);
		if (powerOn) {
			Timer onframe_time;
			onframe_time.start();
			string voltage = ocr.recognize(proc.GetOutputkV());
			string current = ocr.recognize(proc.GetOutputmA());
			string dapRate = ocr.recognize(proc.GetOutputDAP());

			if (tracking) {
				// Perform matching
				vector<cv::linemod::Match> matches;
				vector<cv::String> class_ids;
				vector<cv::Mat> quantized_images;


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
					fullprint = true;
					cv::linemod::Match m = matches[maxID];
					// Draw matching template
					const std::vector<cv::linemod::Template>& templates = detector->getTemplates(m.class_id, m.template_id);
					drawResponse(templates, num_modalities, display, cv::Point(m.x, m.y), detector->getT(0), depthOrigin);
					if(fullprint) {
						onframe_time.stop();
						Print_CST_Result(printFileName,label, m, xy_table, depth, frameNo, voltage, current, dapRate, onframe_time, isocenter);
					}

				}

				if (show_match_result && matches.empty())
					printf("No matches found...\n");

				cout << "Match count: " << match_count << endl;
				printf("Matching Time: %.2fs\n", match_timer.time());
				printf("------------------------------------------------------------\n");

				cv::imshow("normals", quantized_images[1]);
				tracking = false;
			} // tracking
			if(!fullprint && rec && recording) {
				onframe_time.stop();
				Print_CST_Result2(printFileName,frameNo,voltage,current,dapRate,onframe_time);
			}
			if(!fullprint && !rec) {
				onframe_time.stop();
				Print_CST_Result2(printFileName,frameNo,voltage,current,dapRate,onframe_time);
			}

		}
		else { // powerOff
			if (rec) tracking = false;
			else     tracking = true;
		}
		cv::imshow("color", display);


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


		frameNo++;
	} // Main Loop
	delete roi;


	return 0;
}
